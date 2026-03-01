"""
Entry point for running SAC or PPO with vectorized environments.
"""

import multiprocessing
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import wandb
from gymnasium.vector import AsyncVectorEnv
from crowd_nav.policy_utils import get_policy_class

from config.arguments import get_args
from config.config import Config
from crowd_sim.utils import build_env, dump_test_config, dump_train_config, relative_obs_dim_from_env_dim
from eval_policy import RLEvalActorAdapter, eval_policy
from rl.vec_ppo import VecPPO
from rl.vec_sac import VecSAC


ALGO_TO_MODEL = {
    "sac": VecSAC,
    "ppo": VecPPO,
}


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


def prepare_test_save_dir(actor_model):
    base_dir = os.path.dirname(actor_model)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def derive_train_exp_name(timestamp, robot_type, method, algo, actor_model):
    default_name = f"{timestamp}_{robot_type}_{method}_{algo}"
    if not actor_model:
        return default_name

    actor_file = os.path.basename(actor_model)
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(actor_model)))
    if not actor_file.startswith("bc_actor") or not parent_name:
        return default_name

    if parent_name.endswith("_bc"):
        return f"{parent_name[:-3]}_ft"
    return f"{parent_name}_ft"


def ensure_unique_exp_name(base_root, exp_name):
    candidate = exp_name
    idx = 1
    while os.path.exists(os.path.join(base_root, candidate)):
        candidate = f"{exp_name}_{idx}"
        idx += 1
    return candidate


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
            "eval_freq_timesteps": args.ppo_eval_freq_timesteps,
            "eval_episodes": args.ppo_eval_episodes,
            "alpha": config.controller_params["cbf_alpha"],
            "beta": config.controller_params["cvar_beta"],
        }
    )
    return hyperparameters


def train(env, num_envs, algo, hyperparameters, actor_model, critic_model, total_timesteps, method):
    print(f"Training with {num_envs} vectorized environments", flush=True)
    print(f"Algorithm: {algo.upper()}, Policy: FCNet", flush=True)
    PolicyClass = get_policy_class(method)

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


def test(env, actor_model, device, method, hyperparameters, algo):
    print(f"Testing {actor_model}", flush=True)
    if actor_model == "":
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Select Policy Class based on method
    PolicyClass = get_policy_class(method)
    print(f"Algorithm: {algo.upper()}, Policy: {PolicyClass.__name__}", flush=True)
    env_obs_dim = int(env.observation_space.shape[0])
    obs_dim = relative_obs_dim_from_env_dim(env_obs_dim)  
    act_dim = env.action_space.shape[0]

    relevant_keys = ['robot_type', 'safe_dist', 'alpha', 'beta', 'vmax', 'amax', 'omega_max', 'slack_weight']
    policy_kwargs = {k: hyperparameters[k] for k in relevant_keys if k in hyperparameters}		
    print(f"Policy Args: {policy_kwargs}")

    policy = PolicyClass(obs_dim, act_dim, **policy_kwargs).to(device)

    policy.load_state_dict(torch.load(actor_model, map_location=device))
    policy.eval()

    actor = RLEvalActorAdapter(policy, env.action_space, device)
    eval_seed = hyperparameters.get("eval_seed", hyperparameters["seed"])
    save_path = hyperparameters.get("test_save_dir", os.path.dirname(actor_model))

    eval_policy(
        policy=actor,
        env=env,
        max_episodes=hyperparameters["test_ep"],
        save_path=save_path,
        base_seed=eval_seed,
        method=method,
        visualize_episodes=hyperparameters["test_viz_ep"],
    )
    # run_crossing_scenario(actor, env, save_path=save_path)


def main(args):
    set_global_seeds(args.seed)

    if args.method != "rl":
        raise ValueError(
            f"main_vec.py currently supports only method='rl' with FCNet. Got method='{args.method}'."
        )

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

    config = Config()
    env_name = config.env.get("name", "social_nav_var_num")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    robot_type = config.robot_params["type"]
    train_root = os.path.join(".", "trained_models", args.model_folder)
    exp_name = derive_train_exp_name(timestamp, robot_type, args.method, args.algo, args.actor_model)
    if args.mode == "train":
        exp_name = ensure_unique_exp_name(train_root, exp_name)
    save_dir = os.path.join(train_root, exp_name)

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
        os.makedirs(save_dir, exist_ok=True)
        dump_train_config(
            save_dir,
            args,
            config,
            hyperparameters,
            extra={"seed": args.seed, "eval_seed": args.eval_seed, "method": args.method, "algo": args.algo},
        )
        print(f"Models will be saved to: {save_dir}", flush=True)
        wandb.init(project=f"rl_adaptive_cvar_cbf_{args.algo}", name=exp_name, config=hyperparameters)

        num_envs = max(1, multiprocessing.cpu_count())
        env_fns = [make_env_fn(config, env_name) for _ in range(num_envs)]
        vec_env = AsyncVectorEnv(env_fns)
        model_hyperparameters = dict(hyperparameters)
        eval_env = None
        if args.algo == "sac" and args.sac_eval_freq_episodes > 0 and args.sac_eval_episodes > 0:
            eval_env = build_env(env_name, render_mode=None, config=config)
            model_hyperparameters["eval_env"] = eval_env
        if args.algo == "ppo" and args.ppo_eval_freq_timesteps > 0 and args.ppo_eval_episodes > 0:
            eval_env = build_env(env_name, render_mode=None, config=config)
            model_hyperparameters["eval_env"] = eval_env
        try:
            train(
                env=vec_env,
                num_envs=num_envs,
                algo=args.algo,
                hyperparameters=model_hyperparameters,
                actor_model=args.actor_model,
                critic_model=args.critic_model,
                total_timesteps=args.total_timesteps,
                method=args.method,
            )
        finally:
            vec_env.close()
            if eval_env is not None and hasattr(eval_env, "close"):
                eval_env.close()
    else:
        actor_model = args.actor_model
        if not actor_model or not os.path.exists(actor_model):
            print(f"Actor model not found: {actor_model}", flush=True)
            sys.exit(0)

        test_save_dir = prepare_test_save_dir(actor_model)
        hyperparameters["test_save_dir"] = test_save_dir
        dump_test_config(
            test_save_dir,
            config,
            hyperparameters=hyperparameters,
            extra={"eval_seed": args.eval_seed, "method": args.method, "algo": args.algo},
        )

        env = build_env(env_name, render_mode="rgb_array", config=config)
        test(
            env=env,
            actor_model=actor_model,
            device=device,
            method=args.method,
            hyperparameters=hyperparameters,
            algo=args.algo,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
