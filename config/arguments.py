import argparse

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train', choices=['train', 'test'])  # can be 'train' or 'test'
	parser.add_argument('--algo', dest='algo', type=str, default='ppo', choices=['ppo', 'sac'], help='RL algorithm to run')
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # for test mode: trained_models/rl/ppo_actor.pth
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')  	 # for test mode: trained_models/rl/ppo_critic.pth
	parser.add_argument(
		'--method',
		dest='method',
		type=str,
		default='rl',
		choices=[
			'rl',
			'rlcbfgamma',
			'rlcbfgamma_2nets',
			'rlcvarbetaradius',
			'rlcvarbetaradius_2nets',
		],
		help='main_vec RL policy class: rl, rlcbfgamma, rlcbfgamma_2nets, rlcvarbetaradius, rlcvarbetaradius_2nets'
	)
	parser.add_argument(
		'--test_mode',
		type=str,
		default='both',
		choices=['eval', 'crossing', 'both'],
		help='For test mode: run batch eval_policy, run_crossing_scenario, or both'
	)
	parser.add_argument(
		'--obs_topk',
		type=int,
		default=5,
		help='Top-k obstacles used by observation preprocessing'
	)
	parser.add_argument(
		'--obs_farest_dist',
		type=float,
		default=5.0,
		help='Distance cap/padding value for polar observation preprocessing'
	)
	parser.add_argument(
		'--qp_start_timesteps',
		type=int,
		default=0,
		help='Warmup timesteps before enabling QP/optimization layer (0 = enable from start)'
	)

	# -------------------------------------------------------------------------
	# COMMON optimization/logging arguments (shared by SAC and PPO)
	# -------------------------------------------------------------------------
	parser.add_argument('--total_timesteps', type=int, default=20_000_000, help='Total timesteps of the training')
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument(
		'--num_envs',
		type=int,
		default=8,
		help='Number of vectorized environments for main_vec.py (0 = use cpu_count)'
	)

	# -------------------------------------------------------------------------
	# PPO-only arguments
	# -------------------------------------------------------------------------
	parser.add_argument('--n_updates_per_iteration', type=int, default=8, help='Number of epochs to update the policy per iteration')
	parser.add_argument('--timesteps_per_batch', type=int, default=8192)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--clip', type=float, default=0.2)

	parser.add_argument('--lam', type=float, default=0.98, help='Lambda Parameter for GAE')
	parser.add_argument('--num_minibatches', type=int, default=16)
	parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
	parser.add_argument('--target_kl', type=float, default=0.02, help='KL Divergence threshold')
	parser.add_argument('--max_grad_norm', type=float, default=0.5)
	parser.add_argument('--action_std_init', type=float, default=0.5, help='Initial action std')

	parser.add_argument('--eval_freq_timesteps', type=int, default=4_000, help='Frequency of periodic evaluation in timesteps (set <=0 to disable)')
	parser.add_argument('--eval_episodes', type=int, default=50, help='Episodes per periodic training evaluation')
	# -------------------------------------------------------------------------
	# SAC-only arguments
	# -------------------------------------------------------------------------
	parser.add_argument('--sac_timesteps_per_batch', type=int, default=2000, help='Number of gradient updates to perform per iteration (after warmup)')
	parser.add_argument('--sac_buffer_size', type=int, default=500_000)
	parser.add_argument('--sac_batch_size', type=int, default=256)
	parser.add_argument('--sac_start_timesteps', type=int, default=15_000, help='Random action warmup steps')
	parser.add_argument('--sac_updates_per_step', type=int, default=1, help='Gradient steps per env step after warmup')
	parser.add_argument('--sac_hidden_sizes', type=int, nargs='+', default=[256, 256], help='MLP hidden sizes for Q networks')
	parser.add_argument('--sac_tau', type=float, default=0.005, help='Target network soft-update factor')
	parser.add_argument('--sac_actor_lr', type=float, default=3e-4)
	parser.add_argument('--sac_critic_lr', type=float, default=5e-4)
	parser.add_argument('--sac_max_grad_norm', type=float, default=1.0, help='Gradient clipping norm for SAC')

	parser.add_argument('--sac_auto_alpha', action='store_true', default=False, help='Enable automatic entropy temperature tuning')
	parser.add_argument('--sac_alpha', type=float, default=0.0010, help='Initial/fixed entropy temperature')
	parser.add_argument('--sac_alpha_lr', type=float, default=3e-4, help='Learning rate for temperature when auto_alpha is enabled')
	parser.add_argument('--sac_target_entropy', type=float, default=-1.2, help='Target entropy for auto_alpha')
	parser.add_argument('--sac_action_std_init', type=float, default=0.20, help='Initial policy std (before tanh squash)')
	
	parser.add_argument('--sac_eval_freq_episodes', type=int, default=20, help='Run SAC evaluation every N completed training episodes (0 disables)')
	parser.add_argument('--sac_eval_episodes', type=int, default=20, help='Number of episodes per SAC evaluation run')
	# -------------------------------------------------------------------------
	# COMMON evaluation / logging / system arguments (shared by SAC and PPO)
	# -------------------------------------------------------------------------

	parser.add_argument('--test_ep', type=int, default=100, help='Number of episodes for testing')
	parser.add_argument('--test_viz_ep', type=int, default=50, help='Number of episodes to visualize (save gif) during testing')

	parser.add_argument('--render', action='store_true', default=False)
	parser.add_argument('--render_every_i', type=int, default=50, help='Render every i iterations')
	parser.add_argument('--save_after_timesteps', type=int, default=1_000_000, help='Start saving timestep checkpoints after this many timesteps (0 disables)')
	parser.add_argument('--save_freq', type=int, default=1_000_000, help='Save checkpoint every N timesteps (0 disables)')
	parser.add_argument('--model_folder', dest='model_folder', type=str, default='default') # for saving models
	parser.add_argument('--device', dest='device', type=str, default='cuda') # cpu or cuda
	parser.add_argument('--seed', dest='seed', type=int, default=0) # base seed for reproducibility
	parser.add_argument('--eval_seed', dest='eval_seed', type=int, default=100, help='Evaluation seed. Defaults to --seed when not set')
	# parser.add_argument('--eval_seed', dest='eval_seed', type=int, default=None, help='Evaluation seed. Defaults to --seed when not set')


	args = parser.parse_args()

	return args
