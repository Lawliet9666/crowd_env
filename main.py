"""
	Entry point for training/testing SAC in crowd navigation environments.
"""

import sys
import torch
import wandb
import os
import random
import numpy as np
from datetime import datetime

from config.arguments import get_args
from config.config import Config
from rl.sac import SAC
from rl.network import FCNet 
from eval_policy_v2 import RLEvalActorAdapter, eval_policy
from crowd_sim.utils import build_env

def set_global_seeds(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def train(env, hyperparameters, actor_model, critic_model, total_timesteps):
	print(f"Training", flush=True)

	# Create a model for SAC.
	model = SAC(policy_class=FCNet, env=env, **hyperparameters)
	# model = SAC(policy_class=DiffCBF_NN, env=env, **hyperparameters)

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

	# Train SAC with a specified total timesteps
	model.learn(total_timesteps=total_timesteps)
	

def test(env, actor_model, device, test_episodes=50):
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in training
	policy = FCNet(obs_dim, act_dim).to(device)

	# Load in the actor model saved by SAC
	policy.load_state_dict(torch.load(actor_model, map_location=device))

	save_path = os.path.dirname(actor_model)

	# Wrap raw network with deterministic action adapter expected by eval_policy_v2.
	policy = RLEvalActorAdapter(policy, env.action_space, device)

	# Evaluate policy with a separate module to demonstrate
	# that once we are done training, we no longer need sac.py since it only contains
	# the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.

	eval_policy(policy=policy, env=env, max_episodes=test_episodes, save_path=save_path)
	

def main(args):
	set_global_seeds(args.seed)

	config = Config() # input the config path if needed
	env_name = config.env.get('name', 'social_nav_var_num')

	# Create directory for saving models
	# Structure: trained_models/{model_folder}/{timestamp}_{exp_name}/
	now = datetime.now()
	timestamp = now.strftime("%Y%m%d_%H%M%S")
	exp_name = f"{timestamp}_{config.robot_params['type']}_{args.method}"
	save_dir = f"./trained_models/{args.model_folder}/{exp_name}" # Directory to save models and evaluation results (like GIFs)
	
	if args.mode == 'train':
		os.makedirs(save_dir, exist_ok=True)
		print(f"Models will be saved to: {save_dir}")

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

	max_ep_steps = config.env.max_steps if args.max_timesteps_per_episode is None else args.max_timesteps_per_episode

	hyperparameters = {
		# SAC Hyperparameters
		'timesteps_per_batch': args.timesteps_per_batch,
		'max_timesteps_per_episode': max_ep_steps,
		'buffer_size': args.buffer_size,
		'batch_size': args.batch_size,
		'start_timesteps': args.start_timesteps,
		'updates_per_step': args.updates_per_step,
		'hidden_sizes': tuple(args.hidden_sizes),
		'gamma': args.gamma,
		'tau': args.tau,
		'actor_lr': args.actor_lr,
		'critic_lr': args.critic_lr,
		'max_grad_norm': args.max_grad_norm,
		'auto_alpha': args.auto_alpha,
		'alpha': args.alpha,
		'alpha_lr': args.alpha_lr,
		'target_entropy': args.target_entropy,
		'action_std_init': args.action_std_init,
		# Other parameters
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
		'safe_dist': config.controller_params['safety_margin'] + config.human_params['radius'] + config.robot_params['radius'],
		'cbf_alpha': config.controller_params['cbf_alpha'],
		'cvar_beta': config.controller_params['cvar_beta'],
		'robot_type': config.robot_params['type'],
		'vmax': config.robot_params['vmax'],
		'amax': config.robot_params['amax'],
		'omega_max': config.robot_params['omega_max'],
	}

	# Initialize wandb if training
	if args.mode == 'train':
		wandb.init(
			project="rl_adaptive_cvar_cbf_sac", 
			name=exp_name,
			config=hyperparameters
		)

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
	render_mode = 'human' if args.mode == 'test' else None #  'rgb_array'for save fig or 'human' for visualization
	env = build_env(env_name, render_mode=render_mode, config=config)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(
			env=env,
			hyperparameters=hyperparameters,
			actor_model=args.actor_model,
			critic_model=args.critic_model,
			total_timesteps=args.total_timesteps,
		)
	else:
		test(env=env, actor_model=args.actor_model, device=device, test_episodes=args.test_ep)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
