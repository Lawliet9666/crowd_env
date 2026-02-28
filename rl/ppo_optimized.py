"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import os
import time
import wandb
import gymnasium as gym
import imageio

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Independent
from rl.network import FCNet
from crowd_sim.utils import absolute_obs_batch_to_relative, relative_obs_dim_from_env_dim
# from rl.network_deepsets import DeepSetsPolicy, DeepSetsValueNet

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

        # New: Evaluation Env and Debug Mode
        self.last_eval_timestep = 0
        self.best_success_rate = -1.0

        # Extract environment information
        self.env = env
        if hasattr(env, 'single_observation_space'):
            env_obs_dim = int(env.single_observation_space.shape[0])
            self.act_dim = env.single_action_space.shape[0]
            act_space = env.single_action_space
        else:
            env_obs_dim = int(env.observation_space.shape[0])
            self.act_dim = env.action_space.shape[0]
            act_space = env.action_space
        self.obs_dim = relative_obs_dim_from_env_dim(env_obs_dim)

        # Squashed Gaussian action mapping:
        # z ~ N(mu, sigma), u=tanh(z) in [-1, 1], a=bias+scale*u in [low, high]
        self.act_low = torch.tensor(act_space.low, dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(act_space.high, dtype=torch.float32, device=self.device)
        self.act_scale = 0.5 * (self.act_high - self.act_low)
        self.act_bias = 0.5 * (self.act_high + self.act_low)

        # Initialize actor and critic networks
        actor_kwargs = dict(
            safe_dist=self.safe_dist,
            alpha=self.alpha,
            beta=self.beta,
            robot_type=self.robot_type,
            vmax=self.vmax,
            amax=self.amax,
            omega_max=self.omega_max,
        )
        actor_kwargs.update(getattr(self, "policy_kwargs", {}))

        self.actor = policy_class(self.obs_dim, self.act_dim, **actor_kwargs).to(self.device)
        # if issubclass(policy_class, DeepSetsPolicy):
        #     self.critic = DeepSetsValueNet(self.obs_dim, 1).to(self.device)
        # else:
        self.critic = FCNet(self.obs_dim, 1).to(self.device)

        # Learnable log std for action distribution (keeps QP differentiable)
        init_std = self.action_std_init
        self.log_std = nn.Parameter(torch.log(torch.full((self.act_dim,), init_std, device=self.device)))

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(list(self.actor.parameters()) + [self.log_std], lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,           # timesteps so far
            'i_so_far': 0,           # iterations so far
            'batch_lens': [],        # episodic lengths in batch
            'batch_rews': [],        # episodic returns in batch
            'actor_losses': [],      # losses of actor network in current iteration
            'critic_losses': [],     # losses of critic network in current iteration
            'lr': 0,
            'actor_grads': [],       # gradients of actor network (all params)
            'critic_grads': [],      # gradients of critic network
            'u_norm_grads': [],      # gradients of policy/nominal branch
            'cbf_grads': [],         # gradients of CBF branch
            'unom_head_grads': [],   # gradients of unom head params
            'alpha_head_grads': [],  # gradients of alpha head params
            'beta_head_grads': [],   # gradients of beta head params
            'rsafe_head_grads': [],  # gradients of learned safe-radius head params
            'mu_means': [],          # mean of mu (action mean)
            'sigma_means': [],       # mean of sigma (exploration noise)
            'barrier_min_batch': [], # min barrier value in batch
            'barrier_avg_batch': [], # average barrier value in batch
        }
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
        next_timestep_save = None
        if self.save_freq > 0:
            next_timestep_save = self.save_after_timesteps if self.save_after_timesteps > 0 else self.save_freq
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()                     # ALG STEP 3
            batch_steps = int(np.sum(batch_lens))
            t_so_far += batch_steps

            # --- EVALUATION BLOCK ---
            if self.eval_freq_timesteps > 0 and (t_so_far - self.last_eval_timestep) >= self.eval_freq_timesteps:
                self.evaluate_policy_internal(K= max(1, self.eval_episodes), step=t_so_far)
                self.last_eval_timestep = t_so_far
            # ------------------------

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

            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(self.device)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            
            # Increment the number of iterations
            i_so_far += 1

            # Save checkpoints by timestep schedule (if configured).
            while next_timestep_save is not None and t_so_far >= next_timestep_save:
                self._save_model_files(suffix=f"step_{int(next_timestep_save)}")
                print(f"Saved timestep checkpoint at {int(next_timestep_save)} steps.", flush=True)
                next_timestep_save += self.save_freq

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation: 
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient descent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)
                    self.logger['critic_losses'].append(critic_loss.detach().cpu())

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)

                    # --- Log Gradient Norm and Distribution Stats ---
                    total_norm_sq = 0.0
                    u_norm_sq = 0.0
                    gamma_norm_sq = 0.0
                    unom_head_sq = 0.0
                    alpha_head_sq = 0.0
                    beta_head_sq = 0.0
                    rsafe_head_sq = 0.0

                    for name, p in self.actor.named_parameters():
                        if p.grad is not None:
                            g2 = p.grad.data.norm(2).item() ** 2
                            total_norm_sq += g2

                            # Policy/nominal branch gradients
                            if (
                                "unom" in name
                                or "fc31" in name
                                or "fc21" in name
                                or "fc_out" in name
                                or "fc1" in name
                                or "fc2" in name
                                or "layer" in name
                                or "robot_encoder" in name
                                or "obstacle_encoder" in name
                                or "head" in name
                            ):
                                u_norm_sq += g2

                            # CBF-like branch gradients
                            if "alpha" in name or "fc22" in name or "fc32" in name or "fc33" in name:
                                gamma_norm_sq += g2

                            # Head-specific diagnostics
                            if "fc31" in name or "unom" in name:
                                unom_head_sq += g2
                            if "fc32" in name:
                                if hasattr(self.actor, "last_alpha"):
                                    alpha_head_sq += g2
                                if hasattr(self.actor, "last_beta"):
                                    beta_head_sq += g2
                            if "fc33" in name and hasattr(self.actor, "last_r_safe"):
                                rsafe_head_sq += g2

                    # Fallback for policy families whose names don't match split.
                    if total_norm_sq > 0.0 and (u_norm_sq + gamma_norm_sq) == 0.0:
                        u_norm_sq = total_norm_sq

                    self.logger['actor_grads'].append(total_norm_sq ** 0.5)
                    self.logger['u_norm_grads'].append(u_norm_sq ** 0.5)
                    self.logger['cbf_grads'].append(gamma_norm_sq ** 0.5)
                    self.logger['unom_head_grads'].append(unom_head_sq ** 0.5)
                    self.logger['alpha_head_grads'].append(alpha_head_sq ** 0.5)
                    self.logger['beta_head_grads'].append(beta_head_sq ** 0.5)
                    self.logger['rsafe_head_grads'].append(rsafe_head_sq ** 0.5)

                    with torch.no_grad():
                        curr_mu_latent = self.actor(batch_obs)
                        curr_mu, _ = self._squash_action(curr_mu_latent)
                        self.logger['mu_means'].append(curr_mu.mean().item())
                        self.logger['sigma_means'].append(torch.exp(self.log_std).mean().item())
                    # -----------------------------------------------

                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_norm_sq = 0.0
                    for p in self.critic.parameters():
                        if p.grad is not None:
                            critic_norm_sq += p.grad.data.norm(2).item() ** 2
                    self.logger['critic_grads'].append(critic_norm_sq ** 0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break # if kl aboves threshold
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss.detach().cpu())

            # Print a summary of our training so far
            self._log_summary()


    def evaluate_policy_internal(self, K, step): # TODO: multiple seeds evlauation
        print(f"DEBUG: Evaluating policy at step {step}...", flush=True)
        ret_list = []
        len_list = []
        success_count = 0
        collision_count = 0

        # fail_save_dir = os.path.join(self.save_dir, 'eval_failures')
        # os.makedirs(fail_save_dir, exist_ok=True)
        
        for _ in range(K):
            obs, _ = self.eval_env.reset()
            done = False
            ep_ret = 0
            ep_len = 0
            is_success = False
            is_collision = False
            
            while not done:
                with torch.no_grad():
                    obs_rel = absolute_obs_batch_to_relative(obs)
                    obs_tensor = torch.tensor(obs_rel, dtype=torch.float).to(self.device).unsqueeze(0) # Add batch dim
                    # Use mean action directly (deterministic)
                    if hasattr(self, 'actor'):
                        action_tensor, _ = self._squash_action(self.actor(obs_tensor))
                        action = action_tensor.detach().cpu().numpy()[0] # Remove batch dim
                    else:
                        raise ValueError("Actor model not found during evaluation")

                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated | truncated
                ep_ret += reward
                ep_len += 1
                
                if info.get('is_success', False): is_success = True
                if info.get('is_collision', False): is_collision = True
            
            ret_list.append(ep_ret)
            len_list.append(ep_len)
            if is_success: success_count += 1
            if is_collision: collision_count += 1

            # if not is_success:
            #     try:
            #         env_instance = self.eval_env.unwrapped if hasattr(self.eval_env, 'unwrapped') else self.eval_env
            #         original_mode = getattr(env_instance, 'render_mode', None)
            #         env_instance.render_mode = 'rgb_array'
            #         frame = env_instance.render()
            #         env_instance.render_mode = original_mode
            #         if frame is not None:
            #             fname = f"failure_step_{step}_ep_{len(ret_list)-1}.png"
            #             imageio.imsave(os.path.join(fail_save_dir, fname), frame)
            #     except Exception as e:
            #         print(f"Failed to save eval failure snapshot: {e}")
            
        avg_ret = np.mean(ret_list)
        std_ret = np.std(ret_list)
        avg_len = np.mean(len_list)
        success_rate = success_count / K
        collision_rate = collision_count / K
        
        print(f"DEBUG: Eval Result - Returns: {avg_ret:.2f}+/-{std_ret:.2f}, Success: {success_rate:.2f}")
        
        # Log to wandb if initialized
        if wandb.run is not None:
            wandb.log({
                "eval/return_mean": avg_ret,
                "eval/return_std": std_ret,
                "eval/ep_length_mean": avg_len,
                "eval/success_rate": success_rate,
                "eval/collision_rate": collision_rate
            }, step=step)

        if success_rate > self.best_success_rate:
            self.best_success_rate = float(success_rate)
            best_path = os.path.join(self.save_dir, "ppo_actor_best.pth")
            torch.save(self.actor.state_dict(), best_path)
            print(
                f"Saved best actor at step {step} (success={success_rate:.3f})",
                flush=True,
            )

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            ep_rews = np.asarray(ep_rews, dtype=np.float32).reshape(-1)
            ep_vals = np.asarray(ep_vals, dtype=np.float32).reshape(-1)
            ep_dones = np.asarray(ep_dones, dtype=np.float32).reshape(-1)

            advantages = np.zeros_like(ep_rews, dtype=np.float32)
            gae = 0.0

            # Standard GAE: done[t] indicates whether transition at step t ended the episode.
            for t in reversed(range(len(ep_rews))):
                next_value = ep_vals[t + 1] if (t + 1) < len(ep_rews) else 0.0
                next_non_terminal = 1.0 - ep_dones[t]
                delta = ep_rews[t] + self.gamma * next_value * next_non_terminal - ep_vals[t]
                gae = delta + self.gamma * self.lam * next_non_terminal * gae
                advantages[t] = gae

            batch_advantages.extend(advantages.tolist())

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return torch.tensor(batch_rtgs, dtype=torch.float)


    def rollout(self):
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        n_timeout = 0
        n_success = 0
        n_collision = 0
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            # Reset the environment. Note that obs is short for observation. 
            obs, _ = self.env.reset()
            # Initially, the game is not done
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                obs_rel = absolute_obs_batch_to_relative(obs)
                batch_obs.append(obs_rel)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs_tensor = torch.tensor(obs_rel, dtype=torch.float).to(self.device)
                val = self.critic(obs_tensor)

                obs, rew, terminated, truncated, infos = self.env.step(action)
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(float(val.item()))
                # Track done for the transition at the current step (standard done[t] semantics).
                ep_dones.append(done)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    if isinstance(infos, dict):
                        n_timeout += int(infos.get('is_timeout', False))
                        n_success += int(infos.get('is_success', False))
                        n_collision += int(infos.get('is_collision', False))
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten().to(self.device)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.logger['n_timeout'] = n_timeout
        self.logger['n_success'] = n_success
        self.logger['n_collision'] = n_collision

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs_rel = absolute_obs_batch_to_relative(obs)
        obs = torch.tensor(obs_rel, dtype=torch.float).to(self.device)
        mean = self.actor(obs)  # latent mean (unbounded)
        dist = self._build_action_dist(mean)

        # Sample latent action then squash/mapping to real action bounds
        z = dist.rsample()
        action, u = self._squash_action(z)

        # log pi(a|s) = log p(z|s) - log|det(da/dz)|
        base_log_prob = dist.log_prob(z)
        log_prob = self._squash_log_prob(base_log_prob, u)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if self.deterministic:
            action_det, _ = self._squash_action(mean)
            return action_det.detach().cpu().numpy(), 1

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

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
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # Calculate log probabilities with squashed Gaussian change-of-variables correction.
        mean = self.actor(batch_obs)  # latent mean
        dist = self._build_action_dist(mean)
        z, u = self._unsquash_action(batch_acts)
        base_log_probs = dist.log_prob(z)
        log_probs = self._squash_log_prob(base_log_probs, u)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()

    def _build_action_dist(self, mean):
        std = torch.exp(self.log_std).clamp_min(1e-6)
        return Independent(Normal(mean, std), 1)

    def _squash_action(self, z):
        u = torch.tanh(z)
        action = self.act_bias + self.act_scale * u
        return action, u

    def _unsquash_action(self, action):
        scale = self.act_scale
        bias = self.act_bias
        while scale.dim() < action.dim():
            scale = scale.unsqueeze(0)
            bias = bias.unsqueeze(0)

        u = (action - bias) / scale.clamp_min(1e-6)
        u = u.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        z = 0.5 * (torch.log1p(u) - torch.log1p(-u))  # atanh(u)
        return z, u

    def _squash_log_prob(self, base_log_prob, u):
        # log|det(da/dz)| = sum(log(scale)) + sum(log(1 - tanh(z)^2))
        # Here u = tanh(z), so term becomes log(1 - u^2).
        tanh_logdet = torch.log(1.0 - u.pow(2) + 1e-6).sum(dim=-1)
        scale_logdet = torch.log(self.act_scale.clamp_min(1e-6)).sum()
        return base_log_prob - tanh_logdet - scale_logdet

    def _save_model_files(self, suffix=None):
        name_suffix = "" if suffix is None else f"_{suffix}"
        actor_path = os.path.join(self.save_dir, f"ppo_actor{name_suffix}.pth")
        critic_path = os.path.join(self.save_dir, f"ppo_critic{name_suffix}.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 400            # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 3e-4                                  # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.num_minibatches = 6                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0                               # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold
        self.action_std_init = 0.5                       # Initial action std (learnable)
        self.save_after_timesteps = 0                    # First timestep to save checkpoint (0 means disabled)
        self.save_freq = 0                  # Checkpoint interval in timesteps (0 means disabled)
        self.eval_freq_timesteps = 200000                # Eval cadence in timesteps
        self.eval_episodes = 50                          # Episodes per periodic evaluation

        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.deterministic = False                      # If we're testing, don't sample actions
        self.seed = None								# Sets the seed of our program, used for reproducibility of results
        self.save_dir = './'                            # Directory to save models
        self.device = torch.device('cpu')               # Device to run on

        self.safe_dist = 0.8                            # Default safe distance for CBF
        self.alpha = 2.0                                # Default alpha for CBF
        self.beta = 0.2                                 # Default beta for CVaR
        self.robot_type = 'single_integrator'           # Default robot type
        self.vmax = 2.0                                   # Default max control input
        self.omega_max = 3.0                             # Default max angular velocity for unicycle
        self.amax = 3.0                                   # Default max acceleration for double integrator

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            if param == 'eval_env':
                self.eval_env = val
                continue
            if param == 'device':
                self.device = val
                continue

            if isinstance(val, str):
                exec('self.' + param + ' = "' + str(val) + '"')
            else:
                exec('self.' + param + ' = ' + str(val))

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
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean().item() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean().item() for losses in self.logger['critic_losses']]) if self.logger['critic_losses'] else 0.0
        avg_actor_grad = np.mean(self.logger['actor_grads']) if self.logger['actor_grads'] else 0.0
        avg_critic_grad = np.mean(self.logger['critic_grads']) if self.logger['critic_grads'] else 0.0
        avg_policy_grad = np.mean(self.logger['u_norm_grads']) if self.logger['u_norm_grads'] else 0.0
        avg_cbf_grad = np.mean(self.logger['cbf_grads']) if self.logger['cbf_grads'] else 0.0
        avg_unom_head_grad = np.mean(self.logger['unom_head_grads']) if self.logger['unom_head_grads'] else 0.0
        avg_alpha_head_grad = np.mean(self.logger['alpha_head_grads']) if self.logger['alpha_head_grads'] else 0.0
        avg_beta_head_grad = np.mean(self.logger['beta_head_grads']) if self.logger['beta_head_grads'] else 0.0
        avg_rsafe_head_grad = np.mean(self.logger['rsafe_head_grads']) if self.logger['rsafe_head_grads'] else 0.0
        avg_mu = np.mean(self.logger['mu_means']) if self.logger['mu_means'] else 0.0
        avg_sigma = np.mean(self.logger['sigma_means']) if self.logger['sigma_means'] else 0.0
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
                "critic_loss": avg_critic_loss,
                "actor_grad_norm": avg_actor_grad,
                "critic_grad_norm": avg_critic_grad,
                "policy_grad_norm": avg_policy_grad,
                "cbf_grad_norm": avg_cbf_grad,
                "unom_head_grad_norm": avg_unom_head_grad,
                "alpha_head_grad_norm": avg_alpha_head_grad,
                "beta_head_grad_norm": avg_beta_head_grad,
                "rsafe_head_grad_norm": avg_rsafe_head_grad,
                "action_mu": avg_mu,
                "action_sigma": avg_sigma,
                "barrier_min_batch": min_barrier,
                "barrier_avg_batch": avg_barrier,
                "timeout_rate": float(self.logger.get('n_timeout', 0)) / max(len(self.logger['batch_lens']), 1),
                "success_rate": float(self.logger.get('n_success', 0)) / max(len(self.logger['batch_lens']), 1),
                "collision_rate": float(self.logger.get('n_collision', 0)) / max(len(self.logger['batch_lens']), 1),
                "iteration_time": float(delta_t),
                "lr": lr,
            }, step=t_so_far)

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))
        avg_actor_grad = str(round(avg_actor_grad, 5))
        avg_critic_grad = str(round(avg_critic_grad, 5))
        avg_policy_grad = str(round(avg_policy_grad, 5))
        avg_cbf_grad = str(round(avg_cbf_grad, 5))
        avg_unom_head_grad = str(round(avg_unom_head_grad, 5))
        avg_alpha_head_grad = str(round(avg_alpha_head_grad, 5))
        avg_beta_head_grad = str(round(avg_beta_head_grad, 5))
        avg_rsafe_head_grad = str(round(avg_rsafe_head_grad, 5))
        avg_mu = str(round(avg_mu, 5))
        avg_sigma = str(round(avg_sigma, 5))
        min_barrier = str(round(min_barrier, 5))
        avg_barrier = str(round(avg_barrier, 5))
        timeout_rate = str(round(timeout_rate, 5))
        success_rate = str(round(success_rate, 5))
        collision_rate = str(round(collision_rate, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Average Actor Grad Norm: {avg_actor_grad}", flush=True)
        print(f"Average Critic Grad Norm: {avg_critic_grad}", flush=True)
        print(f"  > Policy Grad Norm: {avg_policy_grad}", flush=True)
        print(f"  > CBF Grad Norm: {avg_cbf_grad}", flush=True)
        print(f"  > unom Head Grad Norm: {avg_unom_head_grad}", flush=True)
        print(f"  > alpha Head Grad Norm: {avg_alpha_head_grad}", flush=True)
        print(f"  > beta Head Grad Norm: {avg_beta_head_grad}", flush=True)
        print(f"  > r_safe Head Grad Norm: {avg_rsafe_head_grad}", flush=True)
        print(f"Average Action Mu: {avg_mu}", flush=True)
        print(f"Average Action Sigma: {avg_sigma}", flush=True)
        print(f"Min Barrier Value: {min_barrier}", flush=True)
        print(f"Average Barrier Value: {avg_barrier}", flush=True)
        print(f"Timeout Rate: {timeout_rate}", flush=True)
        print(f"Success Rate: {success_rate}", flush=True)
        print(f"Collision Rate: {collision_rate}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['actor_grads'] = []
        self.logger['critic_grads'] = []
        self.logger['u_norm_grads'] = []
        self.logger['cbf_grads'] = []
        self.logger['unom_head_grads'] = []
        self.logger['alpha_head_grads'] = []
        self.logger['beta_head_grads'] = []
        self.logger['rsafe_head_grads'] = []
        self.logger['mu_means'] = []
        self.logger['sigma_means'] = []
        self.logger['barrier_min_batch'] = []
        self.logger['barrier_avg_batch'] = []
        self.logger['n_timeout'] = 0
        self.logger['n_success'] = 0
        self.logger['n_collision'] = 0
