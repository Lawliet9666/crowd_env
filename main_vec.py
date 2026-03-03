"""
	This file is the executable for running PPO with Vectorized Environments.
"""

import sys
import torch
import wandb
import os
import multiprocessing
import random
import numpy as np
from datetime import datetime
from gymnasium.vector import AsyncVectorEnv
from config.arguments import get_args
from config.config import Config
from rl.vec_ppo import VecPPO
from crowd_nav.rl_policy_factory import get_rl_policy_class
from eval_policy import RLEvalActorAdapter, eval_policy, run_crossing_scenario
from crowd_sim.utils import dump_train_config, build_env, dump_test_config, relative_obs_dim_from_env_dim

def make_env_fn(config, env_name):
	def _init():
		# Create environment with no rendering for training
		env = build_env(env_name, render_mode=None, config=config)
		return env
	return _init

def set_global_seeds(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(	seed)


def get_policy_kwargs(method, config=None):
	kwargs = {}

	cvar_methods = {'rlcvar', 'rlcvarbeta', 'rlcvarbetaradius'}
	if config is not None and method in cvar_methods:
		gmm_cfg = dict(config.human_params.get('gmm', {}))
		kwargs.update({
			'gmm_weights': gmm_cfg.get('weights'),
			'gmm_stds': gmm_cfg.get('stds'),
			'gmm_lateral_ratio': gmm_cfg.get('lateral_ratio', 0.3),
		})

	return kwargs


def build_base_hyperparameters(args, config, env_name, save_dir, device):
	return {
		# Other parameters
		'max_timesteps_per_episode': config.env.max_steps,
		'test_ep': args.test_ep,
		'test_viz_ep': args.test_viz_ep,
		'env_name': env_name,
		'render': args.render,
		'render_every_i': args.render_every_i,
		'save_after_timesteps': args.save_after_timesteps,
		'save_freq': args.save_freq,
		'seed': args.seed,
		'eval_seed': args.seed if args.eval_seed is None else args.eval_seed,
		'save_dir': save_dir,
		'device': device,
		# Environment Parameters
		'safe_dist': config.controller_params['safety_margin'] + config.human_params['radius'] + config.robot_params['radius'], # used in CBF constraints
		'cbf_alpha': config.controller_params['cbf_alpha'],
		'cvar_beta': config.controller_params['cvar_beta'],
		'robot_type': config.robot_params['type'],
		'vmax': config.robot_params['vmax'],
		'amax': config.robot_params['amax'],
		'omega_max': config.robot_params['omega_max'],
	}


def build_ppo_hyperparameters(args, base_hyperparameters):
	hyperparameters = dict(base_hyperparameters)
	hyperparameters.update({
		# PPO Hyperparameters
		'n_updates_per_iteration': args.n_updates_per_iteration,
		'timesteps_per_batch': args.timesteps_per_batch,
		'clip': args.clip,
		'lr': args.lr,
		'gamma': args.gamma,
		'lam': args.lam,
		'ent_coef': args.ent_coef,
		'target_kl': args.target_kl,
		'max_grad_norm': args.max_grad_norm,
		'action_std_init': args.action_std_init,
		'eval_freq_timesteps': args.eval_freq_timesteps,
		'eval_episodes': args.eval_episodes,
	})
	return hyperparameters


def prepare_test_save_dir(actor_model):
	base_dir = os.path.dirname(actor_model)
	run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(base_dir, run_name)
	os.makedirs(run_dir, exist_ok=True)
	return run_dir


def derive_train_exp_name(timestamp, robot_type, method, actor_model):
	default_name = f"{timestamp}_{robot_type}_{method}"
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



def train(env, num_envs, hyperparameters, actor_model, critic_model, method, total_timesteps):
	print(f"Training with {num_envs} vectorized environments", flush=True)	
	# Select Policy Class based on method
	PolicyClass = get_rl_policy_class(method)
	print(f"Algorithm: {method}, Policy: {PolicyClass.__name__}")

	# Set deterministic per-env seeds for reproducibility
	seeds = [hyperparameters['seed'] + i for i in range(num_envs)]
	env.reset(seed=seeds)

	# Create a model for PPO using VecPPO
	model = VecPPO(policy_class=PolicyClass, env=env, num_envs=num_envs, **hyperparameters)

	# Optional warm start: actor-only, critic-only, or both.
	loaded_parts = []
	if actor_model != '':
		if not os.path.exists(actor_model):
			print(f"Actor checkpoint not found: {actor_model}", flush=True)
			sys.exit(0)
		model.actor.load_state_dict(torch.load(actor_model, map_location=hyperparameters['device']))
		loaded_parts.append(f"actor={actor_model}")

	if critic_model != '':
		if not os.path.exists(critic_model):
			print(f"Critic checkpoint not found: {critic_model}", flush=True)
			sys.exit(0)
		model.critic.load_state_dict(torch.load(critic_model, map_location=hyperparameters['device']))
		loaded_parts.append(f"critic={critic_model}")

	if loaded_parts:
		print(f"Warm start loaded: {', '.join(loaded_parts)}", flush=True)
	else:
		print(f"Training from scratch.", flush=True)

	model.learn(total_timesteps=total_timesteps)
	

def test(env, actor_model, device, method, hyperparameters):
	# Test function remains similar, but uses a single env instance usually
	print(f"Testing {actor_model}", flush=True)
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Select Policy Class based on method
	PolicyClass = get_rl_policy_class(method)
	print(f"Algorithm: {method}, Policy: {PolicyClass.__name__}")

	relevant_keys = ['robot_type', 'safe_dist', 'alpha', 'beta', 'vmax', 'amax', 'omega_max', 'slack_weight']
	policy_kwargs = {k: hyperparameters[k] for k in relevant_keys if k in hyperparameters}
	policy_kwargs.update(hyperparameters.get('policy_kwargs', {}))
		
	print(f"Policy Args: {policy_kwargs}")

	obs_dim = relative_obs_dim_from_env_dim(env.observation_space.shape[0])
	act_dim = env.action_space.shape[0]
	policy = PolicyClass(obs_dim, act_dim, **policy_kwargs).to(device)
	policy.load_state_dict(torch.load(actor_model, map_location=device))
	policy.eval()	
	actor = RLEvalActorAdapter(policy, env.action_space, device)
	# actor = policy  # For non-RL methods, use the policy directly without adapter

	# render=True allows eval_policy to dispatch based on render_mode (save gif or show window)
	# Pass base_seed from args to eval_policy to ensure deterministic testing
	eval_seed = hyperparameters.get('eval_seed', hyperparameters['seed'])
	save_path = hyperparameters.get('test_save_dir', os.path.dirname(actor_model))
	# env.render_mode = None  # Override render mode for evaluation to collect frames for GIFs
	eval_policy(
		policy=actor,
		env=env,
		max_episodes=hyperparameters['test_ep'],
		save_path=save_path,
		base_seed=eval_seed,
		method=method,
		visualize_episodes=hyperparameters['test_viz_ep'],
	)
	run_crossing_scenario(actor, env, save_path=save_path)

def main(args):
	# Reproducibility
	set_global_seeds(args.seed)
	
	# Determine device
	if args.device == 'cuda':
		if torch.cuda.is_available():
			device = torch.device('cuda')
			print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
		else:
			device = torch.device('cpu')
			print("GPU requested but not available. Using CPU.", flush=True)
	else:
		device = torch.device('cpu')
		print("Using CPU.", flush=True)

	config = Config() 
	config.env.rl_xy_to_unicycle = bool(args.method == 'rl' and config.robot.type == 'unicycle')
	# Create directory for saving models
	now = datetime.now()
	timestamp = now.strftime("%Y%m%d_%H%M%S")
	robot_type = config.robot_params['type']
	train_root = os.path.join(".", "trained_models", args.model_folder)
	exp_name = derive_train_exp_name(timestamp, robot_type, args.method, args.actor_model)
	if args.mode == 'train':
		exp_name = ensure_unique_exp_name(train_root, exp_name)
	save_dir = os.path.join(train_root, exp_name) # Directory to save models and evaluation results (like GIFs)	
	env_name = config.env.get('name', 'social_nav_var_num')

	base_hyperparameters = build_base_hyperparameters(
		args=args,
		config=config,
		env_name=env_name,
		save_dir=save_dir,
		device=device,
	)
	hyperparameters = build_ppo_hyperparameters(args, base_hyperparameters)
	hyperparameters['policy_kwargs'] = get_policy_kwargs(args.method, config=config)

	if args.mode == 'train':
		os.makedirs(save_dir, exist_ok=True)
		dump_train_config(save_dir, args, config, hyperparameters,extra={"seed": args.seed, "eval_seed": args.eval_seed, "method": args.method})
		print(f"Models will be saved to: {save_dir}")
		wandb.init(
			project="rl_adaptive_cvar_cbf", 
			name=exp_name,
			config=hyperparameters
		)
		# Number of environments for vectorization (can be adjusted)
		# Typically set to number of CPU cores
		num_envs = multiprocessing.cpu_count()
		# num_envs = 4 

		# Create Vectorized Environment
		# AsyncVectorEnv runs environments in separate processes
		env_fns = [make_env_fn(config, env_name) for _ in range(num_envs)]
		vec_env = AsyncVectorEnv(env_fns)
		eval_env = None
		
		# Create a separate environment for periodic training evaluation if enabled.
		if args.eval_freq_timesteps > 0:
			print(
				f"Periodic evaluation enabled every {args.eval_freq_timesteps} timesteps with {args.eval_episodes} episodes; creating eval environment...",
				flush=True,
			)
			eval_env = build_env(env_name, render_mode=None, config=config)
			hyperparameters['eval_env'] = eval_env

		try:
			train(env=vec_env, num_envs=num_envs, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, method=args.method, total_timesteps=args.total_timesteps)
		finally:
			vec_env.close()
			if eval_env is not None:
				eval_env.close()
	else:
		actor_model = args.actor_model
		if not actor_model or not os.path.exists(actor_model):
			print(f"Actor model not found: {actor_model}", flush=True)
			sys.exit(0)

		test_save_dir = prepare_test_save_dir(actor_model)
		hyperparameters['test_save_dir'] = test_save_dir
		dump_test_config(test_save_dir, config, hyperparameters=hyperparameters, extra={"eval_seed": args.eval_seed, "method": args.method})

		# For testing, use --render to switch between on-screen display and GIF capture.
		render_mode = 'human' if args.render else 'rgb_array'
		env = build_env(env_name, render_mode=render_mode, config=config)
		test(env=env, actor_model=actor_model, device=device, method=args.method, hyperparameters=hyperparameters)

if __name__ == '__main__':
	args = get_args() 
	main(args)
