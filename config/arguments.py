import argparse

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # for test mode: trained_models/rl/ppo_actor.pth
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')  	 # for test mode: trained_models/rl/ppo_critic.pth
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
	parser.add_argument(
		'--test_mode',
		type=str,
		default='both',
		choices=['eval', 'crossing', 'both'],
		help='For test mode: run batch eval_policy, run_crossing_scenario, or both'
	)

    # PPO Hyperparameters
	parser.add_argument('--total_timesteps', type=int, default=20_000_000, help='Total timesteps of the training')
	parser.add_argument('--n_updates_per_iteration', type=int, default=3)
	parser.add_argument('--timesteps_per_batch', type=int, default=4096)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--clip', type=float, default=0.2)

	parser.add_argument('--lam', type=float, default=0.98, help='Lambda Parameter for GAE')
	parser.add_argument('--num_minibatches', type=int, default=8)
	parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
	parser.add_argument('--target_kl', type=float, default=0.02, help='KL Divergence threshold')
	parser.add_argument('--max_grad_norm', type=float, default=0.5)
	parser.add_argument('--action_std_init', type=float, default=0.5, help='Initial action std')
	parser.add_argument('--disable_ema', action='store_true', default=False, help='Disable EMA actor for eval/checkpoint')
	parser.add_argument('--ema_decay', type=float, default=0.995, help='EMA decay for actor parameters')

	parser.add_argument('--eval_freq_episodes', type=int, default=20, help='Frequency of evaluation in episodes')
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
	parser.add_argument('--debug', action='store_true', default = False, help='If true, evaluate during training')

	args = parser.parse_args()

	return args
