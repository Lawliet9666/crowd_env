import argparse


def get_args():
	parser = argparse.ArgumentParser()

	# -------------------------------------------------------------------------
	# COMMON runtime arguments (shared by SAC and PPO)
	# -------------------------------------------------------------------------
	parser.add_argument('--mode', dest='mode', type=str, default='train', choices=['train', 'test'])  # can be 'train' or 'test'
	parser.add_argument('--algo', dest='algo', type=str, default='sac', choices=['sac', 'ppo'], help='RL algorithm to run')
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')   # for test mode: trained actor checkpoint
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')  # optional warm-start critic checkpoint
	parser.add_argument(
		'--method',
		dest='method',
		type=str,
		default='rl',
		choices=[
			'rl', 'rlatt', 'rldeepsets', 
			'rldeepsetscbfgamma', 'rldeepsetscvarbetaradius',
			'rlcbfgamma', 'rlcvarbetaradius',
			'orca', 'cbfqp', 'cvarqp', 'adapcvarqp', 'drcvarqp'
		],
			help='rl, rlatt, rldeepsets, rldeepsetscbfgamma, rldeepsetscvarbetaradius, rlcbfgamma, rlcvarbetaradius, orca, cbfqp, cvarqp, adapcvarqp, drcvarqp'
		)

	# -------------------------------------------------------------------------
	# COMMON optimization/logging arguments (shared by SAC and PPO)
	# -------------------------------------------------------------------------
	parser.add_argument('--total_timesteps', type=int, default=10_000_000, help='Total timesteps of the training')
	parser.add_argument('--timesteps_per_batch', type=int, default=4096, help='Logging interval in env steps')
	parser.add_argument('--max_timesteps_per_episode', type=int, default=None, help='Override env max episode steps')
	parser.add_argument('--gamma', type=float, default=0.99)

	# -------------------------------------------------------------------------
	# SAC-only arguments
	# -------------------------------------------------------------------------
	parser.add_argument('--buffer_size', type=int, default=500_000)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--start_timesteps', type=int, default=15_000, help='Random action warmup steps')
	parser.add_argument('--updates_per_step', type=int, default=1, help='Gradient steps per env step after warmup')
	parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 256], help='MLP hidden sizes for Q networks')
	parser.add_argument('--tau', type=float, default=0.005, help='Target network soft-update factor')
	parser.add_argument('--actor_lr', type=float, default=3e-4)
	parser.add_argument('--critic_lr', type=float, default=5e-4)
	parser.add_argument('--sac_max_grad_norm', type=float, default=1.0, help='Gradient clipping norm for SAC')
	parser.add_argument('--sac_eval_freq_episodes', type=int, default=20, help='Run SAC evaluation every N completed training episodes (0 disables)')
	parser.add_argument('--sac_eval_episodes', type=int, default=20, help='Number of episodes per SAC evaluation run')
	parser.add_argument('--no_auto_alpha', dest='auto_alpha', action='store_false', default=True, help='Disable automatic entropy temperature tuning')
	parser.add_argument('--alpha', type=float, default=0.10, help='Initial/fixed entropy temperature')
	parser.add_argument('--alpha_lr', type=float, default=3e-4, help='Learning rate for temperature when auto_alpha is enabled')
	parser.add_argument('--target_entropy', type=float, default=-1.2, help='Target entropy for auto_alpha')
	parser.add_argument('--action_std_init', type=float, default=0.20, help='Initial policy std (before tanh squash)')

	# -------------------------------------------------------------------------
	# PPO-only arguments
	# -------------------------------------------------------------------------
	parser.add_argument('--ppo_n_updates_per_iteration', type=int, default=8)
	parser.add_argument('--ppo_lr', type=float, default=1e-4)
	parser.add_argument('--ppo_clip', type=float, default=0.2)
	parser.add_argument('--ppo_lam', type=float, default=0.98)
	parser.add_argument('--ppo_num_minibatches', type=int, default=8)
	parser.add_argument('--ppo_ent_coef', type=float, default=0.0)
	parser.add_argument('--ppo_target_kl', type=float, default=0.02)
	parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5, help='Gradient clipping norm for PPO')
	parser.add_argument('--ppo_action_std_init', type=float, default=0.5)
	parser.add_argument('--ppo_eval_freq_episodes', type=int, default=50, help='Run PPO evaluation every N completed training episodes (0 disables)')
	parser.add_argument('--ppo_eval_episodes', type=int, default=20, help='Number of episodes per PPO evaluation run')

	# -------------------------------------------------------------------------
	# COMMON evaluation / logging / system arguments (shared by SAC and PPO)
	# -------------------------------------------------------------------------
	parser.add_argument('--test_ep', type=int, default=100, help='Number of episodes for testing')
	parser.add_argument('--test_viz_ep', type=int, default=50, help='Number of episodes to visualize (save gif) during testing')

	parser.add_argument('--render', action='store_true', default=False)
	parser.add_argument('--render_every_i', type=int, default=50, help='Render every i iterations')
	parser.add_argument('--save_after_timesteps', type=int, default=500_000, help='Start saving timestep checkpoints after this many timesteps (0 disables)')
	parser.add_argument('--save_freq', type=int, default=500_000, help='Save checkpoint every N timesteps (0 disables)')
	parser.add_argument('--model_folder', dest='model_folder', type=str, default='default') # for saving models
	parser.add_argument('--device', dest='device', type=str, default='cuda') # cpu or cuda
	parser.add_argument('--seed', dest='seed', type=int, default=0) # base seed for reproducibility
	parser.add_argument('--eval_seed', dest='eval_seed', type=int, default=100, help='Evaluation seed. Defaults to --seed when not set')

	args = parser.parse_args()

	return args
