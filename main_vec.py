"""
Entry point for vectorized PPO/SAC training and testing.
"""

import multiprocessing
import os
from argparse import Namespace
from datetime import datetime

import hydra
import wandb
from gymnasium.vector import AsyncVectorEnv
from omegaconf import DictConfig, OmegaConf

from config.config import Config
from crowd_nav.rl_policy_factory import get_rl_policy_class
from crowd_sim.utils import (
    build_env,
    dump_train_config,
)
from rl.vec_ppo import VecPPO
from rl.vec_sac import VecSAC
from trainer_common import (
    build_base_hyperparameters,
    build_sac_hyperparameters,
    ensure_unique_exp_name,
    load_checkpoints,
    resolve_needs_qp_relative,
    select_device,
    set_global_seeds,
)


ALGO_TO_MODEL = {
    "ppo": VecPPO,
    "sac": VecSAC,
}


MAIN_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env_fn(config, env_name):
    def _init():
        return build_env(env_name, render_mode=None, config=config)

    return _init


def get_policy_kwargs(method, config=None):
    kwargs = {}
    cvar_methods = {"rlcvarbetaradius", "rlcvarbetaradius_2nets"}
    if config is not None and method in cvar_methods:
        gmm_cfg = dict(config.human.get("gmm", {}))
        kwargs.update(
            {
                "gmm_weights": gmm_cfg.get("weights"),
                "gmm_stds": gmm_cfg.get("stds"),
                "gmm_lateral_ratio": gmm_cfg.get("lateral_ratio", 0.3),
            }
        )
    return kwargs


def build_ppo_hyperparameters(args, base_hyperparameters):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "n_updates_per_iteration": args.n_updates_per_iteration,
            "timesteps_per_batch": args.timesteps_per_batch,
            "clip": args.clip,
            "lr": args.lr,
            "gamma": args.gamma,
            "lam": args.lam,
            "ent_coef": args.ent_coef,
            "target_kl": args.target_kl,
            "max_grad_norm": args.max_grad_norm,
            "action_std_init": args.action_std_init,
            "eval_freq_timesteps": args.eval_freq_timesteps,
            "eval_episodes": args.eval_episodes,
            "alpha": base_hyperparameters["cbf_alpha"],
            "beta": base_hyperparameters["cvar_beta"],
        }
    )
    return hyperparameters


def build_train_exp_name(args, config, num_envs):
    base = str(getattr(args, "run_name", "") or "").strip()
    if not base:
        actor_model = str(getattr(args, "actor_model", "") or "").strip()
        actor_file = os.path.basename(actor_model)
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(actor_model))) if actor_model else ""
        if actor_file.startswith("bc_actor") and parent_name:
            base = f"{parent_name[:-3]}_ft" if parent_name.endswith("_bc") else f"{parent_name}_ft"
        else:
            base = datetime.now().strftime("%Y%m%d_%H%M%S")
    robot_type = str(config.robot["type"])
    method = str(args.method)
    algo = str(args.algo).lower().strip()

    if algo == "sac":
        name = (
            f"{base}-{robot_type}-{method}-sac-"
            f"bs{int(args.sac_batch_size)}-a{args.sac_alpha}-"
            f"up{int(args.sac_updates_per_step)}-env{int(num_envs)}"
        )
        if bool(args.sac_auto_alpha):
            name += "-auto"
        return name

    name = (
        f"{base}-{robot_type}-{method}-ppo-"
        f"bs{int(args.timesteps_per_batch)}-ep{int(args.n_updates_per_iteration)}-"
        f"clip{float(args.clip):g}-"
        f"ent{float(args.ent_coef):g}-mb{int(args.num_minibatches)}-env{int(num_envs)}"
    )
    return name


def train(env, num_envs, algo, hyperparameters, actor_model, critic_model, method, total_timesteps):
    print(f"Training with {num_envs} vectorized environments", flush=True)
    PolicyClass = get_rl_policy_class(method)
    print(f"Algorithm: {algo.upper()}, Policy: {PolicyClass.__name__}", flush=True)

    seeds = [hyperparameters["seed"] + i for i in range(num_envs)]
    env.reset(seed=seeds)

    model_cls = ALGO_TO_MODEL[algo]
    model = model_cls(policy_class=PolicyClass, env=env, num_envs=num_envs, **hyperparameters)
    load_checkpoints(
        model=model,
        actor_model=actor_model,
        critic_model=critic_model,
        device=hyperparameters["device"],
        algo=algo,
    )

    model.learn(total_timesteps=total_timesteps)


def main(args):
    set_global_seeds(args.seed)

    device = select_device(args.device)

    needs_qp_relative = resolve_needs_qp_relative(args.method)
    print(
        f"Observation pipeline: actor/critic=polar, qp_relative={needs_qp_relative} (method={args.method})",
        flush=True,
    )

    config = Config()
    # config.env.rl_xy_to_unicycle = bool(args.method == "rl" and config.robot.type == "unicycle")
    env_name = config.env.get("name", "social_nav_var_num")
    num_envs = int(args.num_envs) if int(args.num_envs) > 0 else max(1, multiprocessing.cpu_count())
    print(f"Requested num_envs={args.num_envs}; using num_envs={num_envs}", flush=True)

    train_root = os.path.join(".", "trained_models", args.model_folder)
    exp_name = build_train_exp_name(args, config, num_envs)
    exp_name = ensure_unique_exp_name(train_root, exp_name)
    save_dir = os.path.join(train_root, exp_name)

    base_hyperparameters = build_base_hyperparameters(
        args=args,
        config=config,
        env_name=env_name,
        save_dir=save_dir,
        device=device,
        needs_qp_relative=needs_qp_relative,
        eval_seed=(args.seed if args.eval_seed is None else args.eval_seed),
        include_gamma=False,
        include_controller=True,
    )
    if args.algo == "sac":
        hyperparameters = build_sac_hyperparameters(args, base_hyperparameters)
    else:
        hyperparameters = build_ppo_hyperparameters(args, base_hyperparameters)

    hyperparameters["policy_kwargs"] = get_policy_kwargs(args.method, config=config)

    os.makedirs(save_dir, exist_ok=True)
    dump_train_config(
        save_dir,
        args,
        config,
        hyperparameters,
        extra={
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "method": args.method,
            "algo": args.algo,
        },
    )
    print(f"Models will be saved to: {save_dir}", flush=True)
    wandb.init(project=f"rl_adaptive_cvar_cbf_{args.algo}", name=exp_name, config=hyperparameters)

    env_fns = [make_env_fn(config, env_name) for _ in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)
    eval_env = None

    if args.algo == "ppo" and args.eval_freq_timesteps > 0 and args.eval_episodes > 0:
        eval_env = build_env(env_name, render_mode=None, config=config)
        hyperparameters["eval_env"] = eval_env
    if args.algo == "sac" and args.sac_eval_freq_episodes > 0 and args.sac_eval_episodes > 0:
        eval_env = build_env(env_name, render_mode=None, config=config)
        hyperparameters["eval_env"] = eval_env

    try:
        train(
            env=vec_env,
            num_envs=num_envs,
            algo=args.algo,
            hyperparameters=hyperparameters,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            method=args.method,
            total_timesteps=args.total_timesteps,
        )
    finally:
        vec_env.close()
        if eval_env is not None and hasattr(eval_env, "close"):
            eval_env.close()


def _to_main_vec_args(cfg: DictConfig) -> Namespace:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Hydra config must resolve to a flat dict for main_vec.")
    cfg_dict.pop("hydra", None)
    return Namespace(**cfg_dict)


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="main_vec",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    args = _to_main_vec_args(cfg)
    main(args)


if __name__ == "__main__":
    hydra_main()
