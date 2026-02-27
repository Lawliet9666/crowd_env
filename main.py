"""
    Entry point for training/testing SAC or PPO in crowd navigation environments.
"""

import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import wandb

from config.arguments import get_args
from config.config import Config
from crowd_sim.utils import build_env
from eval_policy_v2 import RLEvalActorAdapter, eval_policy
from rl.network import FCNet
from rl.ppo_optimized import PPO
from rl.sac import SAC


ALGO_TO_MODEL = {
    "sac": SAC,
    "ppo": PPO,
}


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_base_hyperparameters(args, config, env_name, max_ep_steps, save_dir, device):
    return {
        "timesteps_per_batch": args.timesteps_per_batch,
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
        "eval_seed": args.seed if args.eval_seed is None else args.eval_seed,
        "save_dir": save_dir,
        "device": device,
        "safe_dist": config.controller_params["safety_margin"] + config.human_params["radius"] + config.robot_params["radius"],
        "robot_type": config.robot_params["type"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
    }


def build_sac_hyperparameters(args, base_hyperparameters, config):
    eval_freq_episodes = args.sac_eval_freq_episodes
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "start_timesteps": args.start_timesteps,
            "updates_per_step": args.updates_per_step,
            "hidden_sizes": tuple(args.hidden_sizes),
            "tau": args.tau,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "max_grad_norm": args.sac_max_grad_norm,
            "auto_alpha": args.auto_alpha,
            "alpha": args.alpha,
            "alpha_lr": args.alpha_lr,
            "target_entropy": args.target_entropy,
            "action_std_init": args.action_std_init,
            "eval_freq_episodes": eval_freq_episodes,
            "eval_episodes": args.sac_eval_episodes,
            "cbf_alpha": config.controller_params["cbf_alpha"],
            "cvar_beta": config.controller_params["cvar_beta"],
        }
    )
    return hyperparameters


def build_ppo_hyperparameters(args, base_hyperparameters, config):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "n_updates_per_iteration": args.ppo_n_updates_per_iteration,
            "lr": args.ppo_lr,
            "clip": args.ppo_clip,
            "lam": args.ppo_lam,
            "num_minibatches": args.ppo_num_minibatches,
            "ent_coef": args.ppo_ent_coef,
            "target_kl": args.ppo_target_kl,
            "max_grad_norm": args.ppo_max_grad_norm,
            "action_std_init": args.ppo_action_std_init,
            "use_ema": args.ppo_use_ema,
            "ema_decay": args.ppo_ema_decay,
            "eval_freq_episodes": args.ppo_eval_freq_episodes,
            "eval_episodes": args.ppo_eval_episodes,
            "alpha": config.controller_params["cbf_alpha"],
            "beta": config.controller_params["cvar_beta"],
        }
    )
    return hyperparameters


def train(env, algo, hyperparameters, actor_model, critic_model, total_timesteps):
    print(f"Training ({algo.upper()})", flush=True)

    model_cls = ALGO_TO_MODEL[algo]
    model = model_cls(policy_class=FCNet, env=env, **hyperparameters)

    loaded_parts = []
    if actor_model != "":
        if not os.path.exists(actor_model):
            print(f"Actor checkpoint not found: {actor_model}", flush=True)
            sys.exit(0)
        model.actor.load_state_dict(torch.load(actor_model, map_location=hyperparameters["device"]))
        if hasattr(model, "_reset_ema_from_actor"):
            model._reset_ema_from_actor()
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


def test(env, actor_model, device, test_episodes=50, base_seed=None):
    print(f"Testing {actor_model}", flush=True)

    if actor_model == "":
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = FCNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(actor_model, map_location=device))

    save_path = os.path.dirname(actor_model)
    policy = RLEvalActorAdapter(policy, env.action_space, device)

    eval_policy(
        policy=policy,
        env=env,
        max_episodes=test_episodes,
        save_path=save_path,
        base_seed=base_seed,
    )


def main(args):
    set_global_seeds(args.seed)

    config = Config()
    env_name = config.env.get("name", "social_nav_var_num")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{config.robot_params['type']}_{args.method}_{args.algo}"
    save_dir = f"./trained_models/{args.model_folder}/{exp_name}"

    if args.mode == "train":
        os.makedirs(save_dir, exist_ok=True)
        print(f"Models will be saved to: {save_dir}")

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

    max_ep_steps = config.env.max_steps if args.max_timesteps_per_episode is None else args.max_timesteps_per_episode
    base_hyperparameters = build_base_hyperparameters(
        args=args,
        config=config,
        env_name=env_name,
        max_ep_steps=max_ep_steps,
        save_dir=save_dir,
        device=device,
    )

    if args.algo == "sac":
        hyperparameters = build_sac_hyperparameters(args, base_hyperparameters, config)
    else:
        hyperparameters = build_ppo_hyperparameters(args, base_hyperparameters, config)

    if args.mode == "train":
        wandb.init(
            project=f"rl_adaptive_cvar_cbf_{args.algo}",
            name=exp_name,
            config=hyperparameters,
        )

    render_mode = "human" if args.mode == "test" else None
    env = build_env(env_name, render_mode=render_mode, config=config)

    if args.mode == "train":
        model_hyperparameters = dict(hyperparameters)
        eval_env = None
        if args.algo == "sac" and args.sac_eval_freq_episodes > 0 and args.sac_eval_episodes > 0:
            eval_env = build_env(env_name, render_mode=None, config=config)
            model_hyperparameters["eval_env"] = eval_env
        if args.algo == "ppo" and args.ppo_eval_freq_episodes > 0 and args.ppo_eval_episodes > 0:
            eval_env = build_env(env_name, render_mode=None, config=config)
            model_hyperparameters["eval_env"] = eval_env

        try:
            train(
                env=env,
                algo=args.algo,
                hyperparameters=model_hyperparameters,
                actor_model=args.actor_model,
                critic_model=args.critic_model,
                total_timesteps=args.total_timesteps,
            )
        finally:
            if eval_env is not None and hasattr(eval_env, "close"):
                eval_env.close()
    else:
        eval_seed = args.seed if args.eval_seed is None else args.eval_seed
        test(
            env=env,
            actor_model=args.actor_model,
            device=device,
            test_episodes=args.test_ep,
            base_seed=eval_seed,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
