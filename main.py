"""
    Entry point for training/testing SAC in crowd navigation environments.
"""

import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import wandb
import inspect

from config.arguments import get_args
from config.config import Config
from crowd_sim.utils import (
    build_env,
    polar_obs_dim_from_env_dim,
    relative_obs_dim_from_env_dim,
    dump_train_config,
    dump_test_config,
)
from crowd_nav.rl_policy_factory import get_rl_policy_class
from eval_policy import RLEvalActorAdapter, eval_policy
from rl.sac import SAC


METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcvarbetaradius": True,
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


def build_base_hyperparameters(args, config, env_name, max_ep_steps, save_dir, device, needs_qp_relative):
    return {
        "max_timesteps_per_episode": max_ep_steps,
        "gamma": args.gamma,
        "test_ep": args.test_ep,
        "test_viz_ep": args.test_viz_ep,
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


def prepare_test_save_dir(actor_model):
    base_dir = os.path.dirname(actor_model)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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


def test(
    env,
    actor_model,
    device,
    method,
    policy_kwargs,
    obs_topk,
    obs_farest_dist,
    needs_qp_relative,
    test_episodes=50,
    base_seed=None,
    save_path=None,
):
    print(f"Testing {actor_model}", flush=True)

    if actor_model == "":
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    PolicyClass = get_rl_policy_class(method)
    filtered_policy_kwargs = _filter_policy_kwargs(PolicyClass, policy_kwargs)
    env_obs_dim = int(env.observation_space.shape[0])
    obs_dim = polar_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk)
    qp_obs_dim = int(relative_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk))
    act_dim = env.action_space.shape[0]
    if needs_qp_relative:
        filtered_policy_kwargs["qp_obs_dim"] = qp_obs_dim

    print(f"Policy: {PolicyClass.__name__}, args: {filtered_policy_kwargs}", flush=True)
    policy = PolicyClass(obs_dim, act_dim, **filtered_policy_kwargs).to(device)
    policy.load_state_dict(torch.load(actor_model, map_location=device))

    save_path = save_path or os.path.dirname(actor_model)
    policy = RLEvalActorAdapter(policy, env.action_space, device)

    eval_policy(
        policy=policy,
        env=env,
        max_episodes=test_episodes,
        save_path=save_path,
        base_seed=base_seed,
        obs_topk=obs_topk,
        obs_farest_dist=obs_farest_dist,
        needs_qp_relative=needs_qp_relative,
    )


def main(args):
    set_global_seeds(args.seed)

    config = Config()
    needs_qp_relative = resolve_needs_qp_relative(args.method)
    policy_kwargs = _build_policy_kwargs(args.method, config)
    env_name = config.env.get("name", "social_nav_var_num")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{config.robot_params['type']}_{args.method}_sac"
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

    if args.mode == "train":
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

    render_mode = "human" if args.mode == "test" else None
    env = build_env(env_name, render_mode=render_mode, config=config)

    if args.mode == "train":
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
            if eval_env is not None and hasattr(eval_env, "close"):
                eval_env.close()
    else:
        actor_model = args.actor_model
        if not actor_model or not os.path.exists(actor_model):
            print(f"Actor model not found: {actor_model}", flush=True)
            sys.exit(0)

        test_save_dir = prepare_test_save_dir(actor_model)
        test_hyperparameters = dict(hyperparameters)
        test_hyperparameters["test_save_dir"] = test_save_dir
        dump_test_config(
            test_save_dir,
            config,
            hyperparameters=test_hyperparameters,
            extra={"eval_seed": args.eval_seed, "method": args.method},
        )

        test(
            env=env,
            actor_model=actor_model,
            device=device,
            method=args.method,
            policy_kwargs=policy_kwargs,
            obs_topk=args.obs_topk,
            obs_farest_dist=args.obs_farest_dist,
            needs_qp_relative=needs_qp_relative,
            test_episodes=args.test_ep,
            base_seed=args.eval_seed,
            save_path=test_save_dir,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
