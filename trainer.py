"""Shared training/runtime helpers used by training entrypoints."""

import inspect
import os
import random
import sys

import numpy as np
import torch


METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcbfgamma_2nets": True,
    "rlcbfgamma_2nets_risk": True,
    "rlcvarbetaradius": True,
    "rlcvarbetaradius_2nets": True,
    "rlcvarbetaradius_2nets_risk": True,
}


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_needs_qp_relative(method):
    method_key = str(method).strip().lower()
    return bool(METHOD_NEEDS_QP_RELATIVE.get(method_key, False))


def select_device(device_arg):
    if str(device_arg).lower() == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
            return device
        print("GPU requested but not available. Using CPU.", flush=True)
        return torch.device("cpu")

    print("Using CPU.", flush=True)
    return torch.device("cpu")


def filter_policy_kwargs(policy_class, policy_kwargs):
    kwargs = dict(policy_kwargs or {})
    try:
        sig = inspect.signature(policy_class.__init__)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        return {}


def load_checkpoints(model, actor_model, critic_model, device, algo):
    loaded_parts = []

    if actor_model != "":
        if not os.path.exists(actor_model):
            print(f"Actor checkpoint not found: {actor_model}", flush=True)
            sys.exit(0)
        model.actor.load_state_dict(torch.load(actor_model, map_location=device))
        loaded_parts.append(f"actor={actor_model}")

    if critic_model != "":
        if not os.path.exists(critic_model):
            print(f"Critic checkpoint not found: {critic_model}", flush=True)
            sys.exit(0)
        critic_state = torch.load(critic_model, map_location=device)
        algo_key = str(algo).strip().lower()
        if algo_key == "sac":
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

    return loaded_parts


def ensure_unique_exp_name(base_root, exp_name):
    candidate = exp_name
    idx = 1
    while os.path.exists(os.path.join(base_root, candidate)):
        candidate = f"{exp_name}_{idx}"
        idx += 1
    return candidate


def build_base_hyperparameters(
    cfg,
    config,
    *,
    env_name,
    save_dir,
    device,
    needs_qp_relative,
    max_timesteps_per_episode=None,
    eval_seed=None,
    include_gamma=False,
    include_controller=False,
):
    max_steps = (
        int(max_timesteps_per_episode)
        if max_timesteps_per_episode is not None
        else int(config.env.max_steps)
    )
    resolved_eval_seed = cfg.eval_seed if eval_seed is None else eval_seed

    hyperparameters = {
        "max_timesteps_per_episode": max_steps,
        "env_name": env_name,
        "render": cfg.render,
        "render_every_i": cfg.render_every_i,
        "save_after_timesteps": cfg.save_after_timesteps,
        "save_freq": cfg.save_freq,
        "seed": cfg.seed,
        "eval_seed": resolved_eval_seed,
        "save_dir": save_dir,
        "device": device,
        "safe_dist": config.controller["safety_margin"]
        + config.human["radius"]
        + config.robot["radius"],
        "robot_type": config.robot["type"],
        "vmax": config.robot["vmax"],
        "amax": config.robot["amax"],
        "omega_max": config.robot["omega_max"],
        "obs_topk": cfg.obs_topk,
        "obs_farest_dist": cfg.obs_farest_dist,
        "qp_start_timesteps": cfg.qp_start_timesteps,
        "needs_qp_relative": bool(needs_qp_relative),
    }
    if include_gamma:
        hyperparameters["gamma"] = cfg.gamma
    if include_controller:
        hyperparameters["cbf_alpha"] = config.controller["cbf_alpha"]
        hyperparameters["cvar_beta"] = config.controller["cvar_beta"]
    return hyperparameters

def build_ppo_hyperparameters(cfg, base_hyperparameters):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "n_updates_per_iteration": cfg.n_updates_per_iteration,
            "timesteps_per_batch": cfg.timesteps_per_batch,
            "clip": cfg.clip,
            "lr": cfg.lr,
            "gamma": cfg.gamma,
            "lam": cfg.lam,
            "ent_coef": cfg.ent_coef,
            "target_kl": cfg.target_kl,
            "max_grad_norm": cfg.max_grad_norm,
            "action_std_init": cfg.action_std_init,
            "eval_freq_timesteps": cfg.eval_freq_timesteps,
            "eval_episodes": cfg.eval_episodes,
            "alpha": base_hyperparameters["cbf_alpha"],
            "beta": base_hyperparameters["cvar_beta"],
        }
    )
    return hyperparameters
    
def build_sac_hyperparameters(cfg, base_hyperparameters, *, cbf_alpha=None, cvar_beta=None):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "timesteps_per_batch": cfg.sac_timesteps_per_batch,
            "buffer_size": cfg.sac_buffer_size,
            "batch_size": cfg.sac_batch_size,
            "start_timesteps": cfg.sac_start_timesteps,
            "updates_per_step": cfg.sac_updates_per_step,
            "hidden_sizes": tuple(cfg.sac_hidden_sizes),
            "tau": cfg.sac_tau,
            "actor_lr": cfg.sac_actor_lr,
            "critic_lr": cfg.sac_critic_lr,
            "max_grad_norm": cfg.sac_max_grad_norm,
            "auto_alpha": cfg.sac_auto_alpha,
            "alpha": cfg.sac_alpha,
            "alpha_lr": cfg.sac_alpha_lr,
            "target_entropy": cfg.sac_target_entropy,
            "action_std_init": cfg.sac_action_std_init,
            "eval_freq_episodes": cfg.sac_eval_freq_episodes,
            "eval_episodes": cfg.sac_eval_episodes,
        }
    )
    if cbf_alpha is not None:
        hyperparameters["cbf_alpha"] = cbf_alpha
    if cvar_beta is not None:
        hyperparameters["cvar_beta"] = cvar_beta
    return hyperparameters
