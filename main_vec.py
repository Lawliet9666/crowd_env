"""
Entry point for vectorized PPO/SAC training and testing.
"""

import multiprocessing
import os
import random
import sys
from argparse import Namespace
from datetime import datetime

import hydra
import numpy as np
import torch
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


ALGO_TO_MODEL = {
    "ppo": VecPPO,
    "sac": VecSAC,
}


METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcbfgamma_2nets": True,
    "rlcvarbetaradius": True,
    "rlcvarbetaradius_2nets": True,
}


MAIN_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env_fn(config, env_name):
    def _init():
        return build_env(env_name, render_mode=None, config=config)

    return _init


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_needs_qp_relative(method):
    method_key = str(method).strip().lower()
    return bool(METHOD_NEEDS_QP_RELATIVE.get(method_key, False))


def get_policy_kwargs(method, config=None):
    kwargs = {}
    cvar_methods = {"rlcvarbetaradius", "rlcvarbetaradius_v3"}
    if config is not None and method in cvar_methods:
        gmm_cfg = dict(config.human_params.get("gmm", {}))
        kwargs.update(
            {
                "gmm_weights": gmm_cfg.get("weights"),
                "gmm_stds": gmm_cfg.get("stds"),
                "gmm_lateral_ratio": gmm_cfg.get("lateral_ratio", 0.3),
            }
        )
    return kwargs


def build_base_hyperparameters(args, config, env_name, save_dir, device, needs_qp_relative):
    return {
        "max_timesteps_per_episode": config.env.max_steps,
        "env_name": env_name,
        "render": args.render,
        "render_every_i": args.render_every_i,
        "save_after_timesteps": args.save_after_timesteps,
        "save_freq": args.save_freq,
        "seed": args.seed,
        "eval_seed": args.seed if args.eval_seed is None else args.eval_seed,
        "save_dir": save_dir,
        "device": device,
        "safe_dist": config.controller_params["safety_margin"]
        + config.human_params["radius"]
        + config.robot_params["radius"],
        "cbf_alpha": config.controller_params["cbf_alpha"],
        "cvar_beta": config.controller_params["cvar_beta"],
        "robot_type": config.robot_params["type"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
        "obs_topk": args.obs_topk,
        "obs_farest_dist": args.obs_farest_dist,
        "qp_start_timesteps": args.qp_start_timesteps,
        "needs_qp_relative": bool(needs_qp_relative),
    }


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


def build_sac_hyperparameters(args, base_hyperparameters):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "timesteps_per_batch": args.sac_timesteps_per_batch,
            "buffer_size": args.sac_buffer_size,
            "batch_size": args.sac_batch_size,
            "start_timesteps": args.sac_start_timesteps,
            "updates_per_step": args.sac_updates_per_step,
            "hidden_sizes": tuple(args.sac_hidden_sizes),
            "tau": args.sac_tau,
            "actor_lr": args.sac_actor_lr,
            "critic_lr": args.sac_critic_lr,
            "max_grad_norm": args.sac_max_grad_norm,
            "auto_alpha": args.sac_auto_alpha,
            "alpha": args.sac_alpha,
            "alpha_lr": args.sac_alpha_lr,
            "target_entropy": args.sac_target_entropy,
            "action_std_init": args.sac_action_std_init,
            "eval_freq_episodes": args.sac_eval_freq_episodes,
            "eval_episodes": args.sac_eval_episodes,
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
    robot_type = str(config.robot_params["type"])
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


def ensure_unique_exp_name(base_root, exp_name):
    candidate = exp_name
    idx = 1
    while os.path.exists(os.path.join(base_root, candidate)):
        candidate = f"{exp_name}_{idx}"
        idx += 1
    return candidate


def train(env, num_envs, algo, hyperparameters, actor_model, critic_model, method, total_timesteps):
    print(f"Training with {num_envs} vectorized environments", flush=True)
    PolicyClass = get_rl_policy_class(method)
    print(f"Algorithm: {algo.upper()}, Policy: {PolicyClass.__name__}", flush=True)

    seeds = [hyperparameters["seed"] + i for i in range(num_envs)]
    env.reset(seed=seeds)

    model_cls = ALGO_TO_MODEL[algo]
    model = model_cls(policy_class=PolicyClass, env=env, num_envs=num_envs, **hyperparameters)

    loaded_parts = []
    if actor_model != "":
        if not os.path.exists(actor_model):
            print(f"Actor checkpoint not found: {actor_model}", flush=True)
            sys.exit(0)
        model.actor.load_state_dict(torch.load(actor_model, map_location=hyperparameters["device"]))
        loaded_parts.append(f"actor={actor_model}")

    if critic_model != "":
        if not os.path.exists(critic_model):
            print(f"Critic checkpoint not found: {critic_model}", flush=True)
            sys.exit(0)
        critic_state = torch.load(critic_model, map_location=hyperparameters["device"])
        if algo == "sac":
            model.q1.load_state_dict(critic_state)
            model.q2.load_state_dict(critic_state, strict=False)
            model.q1_target.load_state_dict(model.q1.state_dict())
            model.q2_target.load_state_dict(model.q2.state_dict())
            loaded_parts.append(f"critic(q1/q2)={critic_model}")
        else:
            model.critic.load_state_dict(critic_state)
            loaded_parts.append(f"critic={critic_model}")

    if loaded_parts:
        print(f"Warm start loaded: {', '.join(loaded_parts)}", flush=True)
    else:
        print("Training from scratch.", flush=True)

    model.learn(total_timesteps=total_timesteps)


def main(args):
    set_global_seeds(args.seed)

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            device = torch.device("cpu")
            print("GPU requested but not available. Using CPU.", flush=True)
    else:
        device = torch.device("cpu")
        print("Using CPU.", flush=True)

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
