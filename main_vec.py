"""
Entry point for vectorized PPO/SAC training and testing.
"""

import multiprocessing
import os
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
from trainer import (
    build_base_hyperparameters,
    build_ppo_hyperparameters,
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


def get_policy_kwargs(method, cfg: DictConfig | None = None, config=None):
    kwargs = {}
    if cfg is not None:
        if "nHidden1" in cfg:
            kwargs["nHidden1"] = int(cfg.nHidden1)
        if "nHidden21" in cfg:
            kwargs["nHidden21"] = int(cfg.nHidden21)
        if "nHidden22" in cfg:
            kwargs["nHidden22"] = int(cfg.nHidden22)
        if "alpha_hidden1" in cfg:
            kwargs["alpha_hidden1"] = int(cfg.alpha_hidden1)
        if "alpha_hidden2" in cfg:
            kwargs["alpha_hidden2"] = int(cfg.alpha_hidden2)

    cvar_methods = {"rlcvarbetaradius", "rlcvarbetaradius_2nets", "rlcvarbetaradius_2nets_risk"}
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





def build_train_exp_name(cfg: DictConfig, config, num_envs):
    base = str(cfg.get("run_name", "") or "").strip()
    if not base:
        actor_model = str(cfg.get("actor_model", "") or "").strip()
        actor_file = os.path.basename(actor_model)
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(actor_model))) if actor_model else ""
        if actor_file.startswith("bc_actor") and parent_name:
            base = f"{parent_name[:-3]}_ft" if parent_name.endswith("_bc") else f"{parent_name}_ft"
        else:
            base = datetime.now().strftime("%Y%m%d_%H%M%S")
    robot_type = str(config.robot["type"])
    method = str(cfg.method)
    algo = str(cfg.algo).lower().strip()

    if algo == "sac":
        name = (
            f"{base}-{robot_type}-{method}-sac-"
            f"bs{int(cfg.sac_batch_size)}-a{cfg.sac_alpha}-"
            f"up{int(cfg.sac_updates_per_step)}-k{int(cfg.obs_topk)}-env{int(num_envs)}"
        )
        if bool(cfg.sac_auto_alpha):
            name += "-auto"
        return name

    mb_size = float(int(cfg.timesteps_per_batch)) / float(max(1, int(cfg.num_minibatches)))
    mb_size_str = str(int(round(mb_size))) if abs(mb_size - round(mb_size)) < 1e-9 else f"{mb_size:g}"

    name = (
        f"{base}-{robot_type}-{method}-ppo-"
        f"bs{int(cfg.timesteps_per_batch)}-ep{int(cfg.n_updates_per_iteration)}-"
        f"mbsz{mb_size_str}-"
        f"k{int(cfg.obs_topk)}-env{int(num_envs)}"
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


def main(cfg: DictConfig):
    set_global_seeds(int(cfg.seed))

    device = select_device(str(cfg.device))

    needs_qp_relative = resolve_needs_qp_relative(str(cfg.method))
    print(
        f"Observation pipeline: actor/critic=polar, qp_relative={needs_qp_relative} (method={cfg.method})",
        flush=True,
    )

    config = Config()
    # config.env.rl_xy_to_unicycle = bool(cfg.method == "rl" and config.robot.type == "unicycle")
    env_name = config.env.get("name", "social_nav_var_num")
    num_envs = int(cfg.num_envs) if int(cfg.num_envs) > 0 else max(1, multiprocessing.cpu_count())
    print(f"Requested num_envs={cfg.num_envs}; using num_envs={num_envs}", flush=True)

    train_root = os.path.join(".", "trained_models", str(cfg.model_folder))
    exp_name = build_train_exp_name(cfg, config, num_envs)
    exp_name = ensure_unique_exp_name(train_root, exp_name)
    save_dir = os.path.join(train_root, exp_name)

    base_hyperparameters = build_base_hyperparameters(
        cfg=cfg,
        config=config,
        env_name=env_name,
        save_dir=save_dir,
        device=device,
        needs_qp_relative=needs_qp_relative,
        eval_seed=(cfg.seed if cfg.eval_seed is None else cfg.eval_seed),
        include_gamma=False,
        include_controller=True,
    )
    if str(cfg.algo) == "sac":
        hyperparameters = build_sac_hyperparameters(cfg, base_hyperparameters)
    else:
        hyperparameters = build_ppo_hyperparameters(cfg, base_hyperparameters)

    hyperparameters["policy_kwargs"] = get_policy_kwargs(str(cfg.method), cfg=cfg, config=config)

    os.makedirs(save_dir, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        cfg_dict.pop("hydra", None)
    dump_train_config(
        save_dir,
        cfg_dict,
        config,
        hyperparameters,
        extra={
            "seed": int(cfg.seed),
            "eval_seed": cfg.eval_seed,
            "method": str(cfg.method),
            "algo": str(cfg.algo),
        },
    )
    print(f"Models will be saved to: {save_dir}", flush=True)
    wandb.init(project=f"rl_adaptive_cvar_cbf_{cfg.algo}", name=exp_name, config=hyperparameters)

    env_fns = [make_env_fn(config, env_name) for _ in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)
    eval_env = None

    if str(cfg.algo) == "ppo" and int(cfg.eval_freq_timesteps) > 0 and int(cfg.eval_episodes) > 0:
        eval_env = build_env(env_name, render_mode=None, config=config)
        hyperparameters["eval_env"] = eval_env
    if str(cfg.algo) == "sac" and int(cfg.sac_eval_freq_episodes) > 0 and int(cfg.sac_eval_episodes) > 0:
        eval_env = build_env(env_name, render_mode=None, config=config)
        hyperparameters["eval_env"] = eval_env

    try:
        train(
            env=vec_env,
            num_envs=num_envs,
            algo=str(cfg.algo),
            hyperparameters=hyperparameters,
            actor_model=str(cfg.actor_model),
            critic_model=str(cfg.critic_model),
            method=str(cfg.method),
            total_timesteps=int(cfg.total_timesteps),
        )
    finally:
        vec_env.close()
        if eval_env is not None and hasattr(eval_env, "close"):
            eval_env.close()


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="main_vec",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
