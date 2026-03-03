"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym
import time
import wandb
import os
import imageio

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from rl.network import FCNet
from crowd_sim.utils import absolute_obs_to_relative, relative_obs_dim_from_env_dim

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		
		# Create directory for failure figures
		self.fail_save_dir = os.path.join(self.save_dir, 'fail_figure')
		os.makedirs(self.fail_save_dir, exist_ok=True)

		# Extract environment information
		self.env = env
		if hasattr(env, 'single_observation_space'):
			env_obs_dim = int(env.single_observation_space.shape[0])
			self.act_dim = env.single_action_space.shape[0]
		else:
			env_obs_dim = int(env.observation_space.shape[0])
			self.act_dim = env.action_space.shape[0]
		self.obs_dim = relative_obs_dim_from_env_dim(env_obs_dim)

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, 
							self.act_dim, 
							safe_dist=self.safe_dist,
							alpha=self.alpha,
							beta=self.beta,
							robot_type=self.robot_type
							).to(self.device)                                                  
		self.critic = FCNet(self.obs_dim, 1).to(self.device)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.4).to(self.device)
		self.cov_mat = torch.diag(self.cov_var).to(self.device)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'actor_grads': [],      # gradients of actor network
			'policy_grads': [],     # gradients of policy body/nominal head
			'cbf_grads': [],        # gradients of CBF/alpha head
			'mu_means': [],         # mean of mu (action mean)
			'sigma_means': [],      # mean of sigma (exploration noise)
			'barrier_min_batch': [], # min barrier value in batch
			'barrier_avg_batch': [], # average barrier value in batch
		}
		
		# Track all average episodic rewards for plotting
		self.all_ep_rews = []

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = self.rollout()                     # ALG STEP 3

			# --- Log Min Barrier (Worst Case Safety) ---
			# We compute this outside the main training loop to avoid slowing down backprop.
			# Obs indices: 6 (rel_x), 7 (rel_y)
			with torch.no_grad():
				rel_x = batch_obs[:, 6]
				rel_y = batch_obs[:, 7]
				dist_sq = rel_x**2 + rel_y**2
				barrier = dist_sq - self.safe_dist**2
				min_barrier = torch.min(barrier).item()
				self.logger['barrier_min_batch'] = min_barrier
				self.logger['barrier_avg_batch'] = torch.mean(barrier).item()
			# ---------------------------------------------

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate and store average reward for plotting
			# avg_ep_rew = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
			# self.all_ep_rews.append(avg_ep_rew)

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)

				# --- Log Gradient Norm and Distribution Stats ---
				total_norm_sq = 0.0
				u_norm_sq = 0.0
				gamma_norm_sq = 0.0

				for name, p in self.actor.named_parameters():
					if p.grad is not None:
						param_norm_sq = p.grad.data.norm(2).item() ** 2
						total_norm_sq += param_norm_sq
						
						# Classification: "fc32" and "fc22" are the alpha head path
						if "fc32" in name or "fc22" in name or "alpha" in name:
							# print(f"[GRAD LOG] CBF Param: {name}, Grad Norm: {param_norm_sq ** 0.5:.6f}")
							gamma_norm_sq += param_norm_sq
						elif "fc31" in name or "fc21" in name or "unom" in name or "fc_out" in name or "fc1" in name or "fc2" in name or "layer" in name:
							# "fc31", "fc21", and "fc1" (shared) go here
							# print(f"[GRAD LOG] Policy Param: {name}, Grad Norm: {param_norm_sq ** 0.5:.6f}")
							u_norm_sq += param_norm_sq
						else:
							# print(f"[GRAD LOG] Actor Param: {name}, Grad Norm: {param_norm_sq ** 0.5:.6f}")
							pass

				self.logger['actor_grads'].append(total_norm_sq ** 0.5)
				self.logger['policy_grads'].append(u_norm_sq ** 0.5)
				self.logger['cbf_grads'].append(gamma_norm_sq ** 0.5)

				with torch.no_grad():
					curr_mu = self.actor(batch_obs)
					self.logger['mu_means'].append(curr_mu.mean().item())
					self.logger['sigma_means'].append(self.cov_var.mean().item())
					# # --- NEW CODE ---
					# curr_mu, curr_std = self.actor(batch_obs)
					# self.logger['mu_means'].append(curr_mu.mean().item())
					# self.logger['sigma_means'].append(curr_std.mean().item())	
				# -----------------------------------------------

				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach().cpu())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				actor_path = os.path.join(self.save_dir, 'ppo_actor.pth')
				critic_path = os.path.join(self.save_dir, 'ppo_critic.pth')
				torch.save(self.actor.state_dict(), actor_path)
				torch.save(self.critic.state_dict(), critic_path)

		# return self.all_ep_rews

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []
		n_timeout = 0
		n_success = 0
		n_collision = 0

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. sNote that obs is short for observation. 
			obs, _ = self.env.reset()
			# obs, _ = self.env.reset(options={"scenario": "crossing"})
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in relative format for actor/critic.
				obs_rel = absolute_obs_to_relative(obs)
				batch_obs.append(obs_rel)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs_rel)
				obs, rew, terminated, truncated, infos = self.env.step(action)

				# --- Save Failure Cases ---
				# # if infos.get('is_collision', False) or infos.get('is_timeout', False):
				# if infos.get('is_collision', False):
				# 	print("Saving failure figure...", flush=True)
				# 	try:
				# 		# Handle unwrapped env if necessary to access render_mode
				# 		env_instance = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
						
				# 		# Save original mode
				# 		original_mode = getattr(env_instance, 'render_mode', None)
						
				# 		# Force rgb_array
				# 		env_instance.render_mode = 'rgb_array'
				# 		frame = env_instance.render()
						
				# 		# Restore mode
				# 		env_instance.render_mode = original_mode
				# 		print(f"env render_mode restored to {original_mode}")
				# 		if frame is not None:
				# 			print(f"Saving failure figure for episode {len(batch_lens)} at step {ep_t}...")
				# 			# Filename: fail_iter_X_ep_Y_step_Z.png
				# 			fname = f"fail_iter_{self.logger['i_so_far']}_ep_{len(batch_lens)}_step_{ep_t}.png"
				# 			save_path = os.path.join(self.fail_save_dir, fname)
				# 			imageio.imsave(save_path, frame)
				# 			# print(f"Saved failure figure to {save_path}")
				# 	except Exception as e:
				# 		print(f"Error saving failure figure: {e}")
				# ---------------------------

				# Don't really care about the difference between terminated or truncated in this, so just combine them
				done = terminated | truncated

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					if isinstance(infos, dict):
						n_timeout += int(infos.get('is_timeout', False))
						n_success += int(infos.get('is_success', False))
						n_collision += int(infos.get('is_collision', False))
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(self.device)
		batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(self.device)
		batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(self.device)
		batch_rtgs = self.compute_rtgs(batch_rews).to(self.device)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens
		self.logger['n_timeout'] = n_timeout
		self.logger['n_success'] = n_success
		self.logger['n_collision'] = n_collision

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		obs_rel = absolute_obs_to_relative(obs)
		obs_t = torch.tensor(obs_rel, dtype=torch.float32).to(self.device)

		# Query the actor network for a mean action
		mean = self.actor(obs_t)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.cpu().detach().numpy(), log_prob.detach().cpu().numpy()

	# def get_action(self, obs):
	# 	# --- NEW: Output mean and std for Gaussian policy ---
	# 	# Query the actor network for mean AND std
	# 	mean, std = self.actor(obs)

	# 	# Create the covariance matrix from the learned std
	# 	# std is shape (act_dim,), so we assume diagonal covariance
	# 	cov_mat = torch.diag(std.pow(2))

	# 	# Create distribution
	# 	dist = MultivariateNormal(mean, cov_mat)

	# 	# Sample and log_prob
	# 	action = dist.sample()
	# 	log_prob = dist.log_prob(action)

	# 	return action.cpu().detach().numpy(), log_prob.detach().cpu().numpy()

	# def evaluate(self, batch_obs, batch_acts):
	# 	# Query critic
	# 	V, _ = self.critic(batch_obs)
	# 	V = V.squeeze()
	# 	# Query actor for mean AND std
	# 	mean, std = self.actor(batch_obs)
		
	# 	# Create Covariance Matrix
	# 	cov_mat = torch.diag(std.pow(2))
		
	# 	# Create Distribution
	# 	# mean is (Batch_Size, Act_Dim), cov_mat is (Act_Dim, Act_Dim)
	# 	# PyTorch automatically broadcasts cov_mat to match batch size
	# 	dist = MultivariateNormal(mean, cov_mat)
		
	# 	log_probs = dist.log_prob(batch_acts)

	# 	return V, log_probs

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results
		self.save_dir = './'                            # Directory to save models
		self.device = torch.device('cpu')               # Device to run on

		self.safe_dist = 0.8                            # Default safe distance for CBF
		self.alpha = 2.0                                # Default alpha for CBF
		self.beta = 0.2                                 # Default beta for CVaR
		self.robot_type = 'single_integrator'            # Default robot type

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			if param == 'device':
				self.device = val
				continue

			if isinstance(val, str):
				# Wrap string in quotes so exec handles it correctly
				# e.g. "path/to/dir" -> "'path/to/dir'"
				exec('self.' + param + ' = "' + str(val) + '"')
			else:
				exec('self.' + param + ' = ' + str(val))
        
		# Ensure save_dir is created (though main.py handles it, good for safety)
		if hasattr(self, 'save_dir'):
			os.makedirs(self.save_dir, exist_ok=True)

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		avg_actor_grad = np.mean(self.logger['actor_grads'])
		avg_policy_grad = np.mean(self.logger['policy_grads']) if self.logger['policy_grads'] else 0.0
		avg_cbf_grad = np.mean(self.logger['cbf_grads']) if self.logger['cbf_grads'] else 0.0
		avg_mu = np.mean(self.logger['mu_means'])
		avg_sigma = np.mean(self.logger['sigma_means'])
		min_barrier = self.logger.get('barrier_min_batch', 0.0)
		avg_barrier = self.logger.get('barrier_avg_batch', 0.0)
		n_episodes = max(len(self.logger['batch_lens']), 1)
		timeout_rate = float(self.logger.get('n_timeout', 0)) / n_episodes
		success_rate = float(self.logger.get('n_success', 0)) / n_episodes
		collision_rate = float(self.logger.get('n_collision', 0)) / n_episodes

		if wandb.run is not None:
			wandb.log({
				"iteration": i_so_far,
				"timesteps": t_so_far,
				"ep_len": avg_ep_lens,
				"ep_reward": avg_ep_rews,
				"actor_loss": avg_actor_loss,
				"actor_grad_norm": avg_actor_grad,
				"policy_grad_norm": avg_policy_grad,
				"cbf_grad_norm": avg_cbf_grad,
				"action_mu": avg_mu,
				"action_sigma": avg_sigma,
				"barrier_min_batch": min_barrier,
				"barrier_avg_batch": avg_barrier,
				"timeout_rate": timeout_rate,
				"success_rate": success_rate,
				"collision_rate": collision_rate,
				"iteration_time": float(delta_t)
			}, step=t_so_far)

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))
		avg_actor_grad = str(round(avg_actor_grad, 5))
		avg_policy_grad_str = str(round(avg_policy_grad, 5))
		avg_cbf_grad_str = str(round(avg_cbf_grad, 5))
		avg_mu = str(round(avg_mu, 5))
		avg_sigma = str(round(avg_sigma, 5))
		min_barrier = str(round(min_barrier, 5))
		avg_barrier = str(round(avg_barrier, 5))
		timeout_rate_str = str(round(timeout_rate, 5))
		success_rate_str = str(round(success_rate, 5))
		collision_rate_str = str(round(collision_rate, 5))
		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Average Actor Grad Norm: {avg_actor_grad}", flush=True)
		print(f"  > Policy Grad Norm: {avg_policy_grad_str}", flush=True)
		print(f"  > CBF Grad Norm: {avg_cbf_grad_str}", flush=True)
		print(f"Average Action Mu: {avg_mu}", flush=True)
		print(f"Average Action Sigma: {avg_sigma}", flush=True)
		print(f"Min Barrier Value: {min_barrier}", flush=True)
		print(f"Average Barrier Value: {avg_barrier}", flush=True)
		print(f"Timeout Rate: {timeout_rate_str}", flush=True)
		print(f"Success Rate: {success_rate_str}", flush=True)
		print(f"Collision Rate: {collision_rate_str}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['actor_grads'] = []
		self.logger['mu_means'] = []
		self.logger['sigma_means'] = []
		self.logger['barrier_min_batch'] = []
		self.logger['barrier_avg_batch'] = []
		self.logger['policy_grads'] = []
		self.logger['cbf_grads'] = []
		self.logger['n_timeout'] = 0
		self.logger['n_success'] = 0
		self.logger['n_collision'] = 0
