"""
    Entry point for training/testing SAC in crowd navigation environments.
"""

import os
import random
import sys
from argparse import Namespace
from datetime import datetime

import hydra
import numpy as np
import torch
import wandb
import inspect
from omegaconf import DictConfig, OmegaConf

from config.config import Config
from crowd_sim.utils import (
    build_env,
    dump_train_config,
)
from crowd_nav.rl_policy_factory import get_rl_policy_class
from rl.sac import SAC


METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcbfgamma_2nets": True,
    "rlcvarbetaradius": True,
    "rlcvarbetaradius_2nets": True,
}


MAIN_DIR = os.path.dirname(os.path.abspath(__file__))


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_needs_qp_relative(method):
    method_key = str(method).strip().lower()
    return bool(METHOD_NEEDS_QP_RELATIVE.get(method_key, False))


def build_sac_exp_name(args, config):
    base = str(getattr(args, "run_name", "") or "").strip()
    if not base:
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
    robot_type = str(config.robot_params["type"])
    method = str(args.method)
    name = (
        f"{base}-{robot_type}-{method}-sac-"
        f"bs{int(args.sac_batch_size)}-a{args.sac_alpha}-"
        f"up{int(args.sac_updates_per_step)}"
    )
    if bool(args.sac_auto_alpha):
        name += "-auto"
    return name


def build_base_hyperparameters(args, config, env_name, max_ep_steps, save_dir, device, needs_qp_relative):
    return {
        "max_timesteps_per_episode": max_ep_steps,
        "gamma": args.gamma,
        "env_name": env_name,
        "render": args.render,
        "render_every_i": args.render_every_i,
        "save_after_timesteps": args.save_after_timesteps,
        "save_freq": args.save_freq,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "save_dir": save_dir,
        "device": device,
        "safe_dist": config.controller_params["safety_margin"] + config.human_params["radius"] + config.robot_params["radius"],
        "robot_type": config.robot_params["type"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
        "obs_topk": args.obs_topk,
        "obs_farest_dist": args.obs_farest_dist,
        "qp_start_timesteps": args.qp_start_timesteps,
        "needs_qp_relative": bool(needs_qp_relative),
    }


def build_sac_hyperparameters(args, base_hyperparameters, config):
    eval_freq_episodes = args.sac_eval_freq_episodes
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
            "eval_freq_episodes": eval_freq_episodes,
            "eval_episodes": args.sac_eval_episodes,
            "cbf_alpha": config.controller_params["cbf_alpha"],
            "cvar_beta": config.controller_params["cvar_beta"],
        }
    )
    return hyperparameters


def _build_policy_kwargs(method, config):
    gmm_cfg = dict(config.human_params.get("gmm", {}))
    kwargs = {
        "robot_type": config.robot_params["type"],
        "safe_dist": config.controller_params["safety_margin"] + config.human_params["radius"] + config.robot_params["radius"],
        "alpha": config.controller_params["cbf_alpha"],
        "beta": config.controller_params["cvar_beta"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
        "gmm_weights": gmm_cfg.get("weights"),
        "gmm_stds": gmm_cfg.get("stds"),
        "gmm_lateral_ratio": gmm_cfg.get("lateral_ratio", 0.3),
    }
    return kwargs


def _filter_policy_kwargs(policy_class, policy_kwargs):
    try:
        sig = inspect.signature(policy_class.__init__)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in policy_kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        return {}


def train(env, method, policy_kwargs, hyperparameters, actor_model, critic_model, total_timesteps):
    print("Training (SAC)", flush=True)

    PolicyClass = get_rl_policy_class(method)
    filtered_policy_kwargs = _filter_policy_kwargs(PolicyClass, policy_kwargs)
    model_hyperparameters = dict(hyperparameters)
    model_hyperparameters["policy_kwargs"] = filtered_policy_kwargs
    print(f"Policy: {PolicyClass.__name__}, args: {filtered_policy_kwargs}", flush=True)
    model = SAC(policy_class=PolicyClass, env=env, **model_hyperparameters)

    loaded_parts = []
    if actor_model != "":
        if not os.path.exists(actor_model):
            print(f"Actor checkpoint not found: {actor_model}", flush=True)
            sys.exit(0)
        model.actor.load_state_dict(torch.load(actor_model, map_location=model_hyperparameters["device"]))
        loaded_parts.append(f"actor={actor_model}")

    if critic_model != "":
        if not os.path.exists(critic_model):
            print(f"Critic checkpoint not found: {critic_model}", flush=True)
            sys.exit(0)
        critic_state = torch.load(critic_model, map_location=model_hyperparameters["device"])
        model.q1.load_state_dict(critic_state)
        model.q2.load_state_dict(critic_state, strict=False)
        model.q1_target.load_state_dict(model.q1.state_dict())
        model.q2_target.load_state_dict(model.q2.state_dict())
        loaded_parts.append(f"critic(q1/q2)={critic_model}")

    if loaded_parts:
        print(f"Warm start loaded: {', '.join(loaded_parts)}", flush=True)
    else:
        print("Training from scratch.", flush=True)

    model.learn(total_timesteps=total_timesteps)


def main(args):
    set_global_seeds(args.seed)

    config = Config()
    needs_qp_relative = resolve_needs_qp_relative(args.method)
    policy_kwargs = _build_policy_kwargs(args.method, config)
    env_name = config.env.get("name", "social_nav_var_num")

    exp_name = build_sac_exp_name(args, config)
    save_dir = f"./trained_models/{args.model_folder}/{exp_name}"

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

    max_ep_steps = config.env.max_steps  
    base_hyperparameters = build_base_hyperparameters(
        args=args,
        config=config,
        env_name=env_name,
        max_ep_steps=max_ep_steps,
        save_dir=save_dir,
        device=device,
        needs_qp_relative=needs_qp_relative,
    )
    hyperparameters = build_sac_hyperparameters(args, base_hyperparameters, config)

    os.makedirs(save_dir, exist_ok=True)
    dump_train_config(
        save_dir,
        args,
        config,
        hyperparameters=hyperparameters,
        extra={"seed": args.seed, "eval_seed": args.eval_seed, "method": args.method},
    )
    print(f"Models will be saved to: {save_dir}")
    wandb.init(
        project="rl_adaptive_cvar_cbf_sac",
        name=exp_name,
        config=hyperparameters,
    )

    env = build_env(env_name, render_mode=None, config=config)

    model_hyperparameters = dict(hyperparameters)
    eval_env = None
    if args.sac_eval_freq_episodes > 0 and args.sac_eval_episodes > 0:
        eval_env = build_env(env_name, render_mode=None, config=config)
        model_hyperparameters["eval_env"] = eval_env

    try:
        train(
            env=env,
            method=args.method,
            policy_kwargs=policy_kwargs,
            hyperparameters=model_hyperparameters,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            total_timesteps=args.total_timesteps,
        )
    finally:
        if hasattr(env, "close"):
            env.close()
        if eval_env is not None and hasattr(eval_env, "close"):
            eval_env.close()


def _to_main_args(cfg: DictConfig) -> Namespace:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Hydra config must resolve to a flat dict for main.py.")
    cfg_dict.pop("hydra", None)
    return Namespace(**cfg_dict)


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="main",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    args = _to_main_args(cfg)
    main(args)


if __name__ == "__main__":
    hydra_main()
