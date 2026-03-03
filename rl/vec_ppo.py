import torch
import numpy as np
# from rl.ppo import PPO
from rl.ppo_optimized import PPO

class VecPPO(PPO):
    def __init__(self, policy_class, env, num_envs, **hyperparameters):
        super().__init__(policy_class, env, **hyperparameters)
        self.num_envs = num_envs

    def rollout(self):
        """
            Rollout logic for Vectorized Environments.
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        batch_obs_qp = [] if self.use_dual_actor_input else None

        # Buffers for each environment
        env_obs = [[] for _ in range(self.num_envs)]
        env_obs_qp = [[] for _ in range(self.num_envs)] if self.use_dual_actor_input else None
        env_acts = [[] for _ in range(self.num_envs)]
        env_log_probs = [[] for _ in range(self.num_envs)]
        env_rews = [[] for _ in range(self.num_envs)]
        env_dones = [[] for _ in range(self.num_envs)]

        n_timeout = 0
        n_success = 0
        n_collision = 0

        # Reset all environments
        obs_raw, _ = self.env.reset()
        
        t_so_far = 0
        
        # We continue until we have collected enough timesteps in COMPLETED episodes
        while t_so_far < self.timesteps_per_batch:
            obs_actor, obs_qp = self._preprocess_obs_pair(obs_raw)
            # Get actions for all envs
            actions, log_probs = self.get_action(obs_actor, obs_qp=obs_qp, preprocessed=True)
            
            # Step the vectorized environment
            next_obs_raw, rews, terminations, truncations, infos = self.env.step(actions)
            
            dones = terminations | truncations

            for i in range(self.num_envs):
                # Store step data
                env_obs[i].append(obs_actor[i])
                if env_obs_qp is not None:
                    env_obs_qp[i].append(obs_qp[i])
                env_acts[i].append(actions[i])
                env_log_probs[i].append(log_probs[i])
                env_rews[i].append(rews[i])
                env_dones[i].append(bool(dones[i]))
                
                if dones[i]:
                    # Standardize info access (Gymnasium tuple-of-dicts vs Legacy dict-of-arrays)
                    info = infos[i] if isinstance(infos, (tuple, list)) else {k: v[i] for k, v in infos.items() if v is not None}
                    
                    # Check 'final_info' for meaningful terminal state (Gymnasium auto-reset)
                    info = info.get('final_info') or info

                    n_timeout += int(info.get('is_timeout', False))
                    n_success += int(info.get('is_success', False))
                    n_collision += int(info.get('is_collision', False))

                    # Episode finished for env i
                    ep_len = len(env_rews[i])
                    ep_rews = env_rews[i]
                    
                    # Store episode data to batch
                    batch_obs.extend(env_obs[i])
                    if batch_obs_qp is not None:
                        batch_obs_qp.extend(env_obs_qp[i])
                    batch_acts.extend(env_acts[i])
                    batch_log_probs.extend(env_log_probs[i])
                    batch_rews.append(ep_rews)
                    batch_lens.append(ep_len)

                    # Compute value estimates for this episode
                    with torch.no_grad():
                        ep_obs_tensor = torch.tensor(np.array(env_obs[i]), dtype=torch.float).to(self.device)
                        ep_vals = self.critic(self._to_critic_obs(ep_obs_tensor)).squeeze().detach().cpu().numpy().tolist()
                    if not isinstance(ep_vals, list):
                        ep_vals = [ep_vals]
                    batch_vals.append(ep_vals)
                    batch_dones.append(env_dones[i])
                    
                    # Calculate RTGs for this episode and extend
                    # We can use the existing compute_rtgs but meant for batch, 
                    # so let's do it locally or aggregate and call later.
                    # Original PPO calls compute_rtgs(batch_rews) at the end.
                    
                    t_so_far += ep_len
                    
                    # Reset buffers for env i
                    env_obs[i] = []
                    if env_obs_qp is not None:
                        env_obs_qp[i] = []
                    env_acts[i] = []
                    env_log_probs[i] = []
                    env_rews[i] = []
                    env_dones[i] = []
            
            # Update obs
            obs_raw = next_obs_raw

        # Convert to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(self.device)
        if batch_obs_qp is not None:
            self._last_batch_obs_qp = torch.tensor(np.array(batch_obs_qp), dtype=torch.float).to(self.device)
        else:
            self._last_batch_obs_qp = None
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(self.device)
        
        # Log
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.logger['n_timeout'] = n_timeout
        self.logger['n_success'] = n_success
        self.logger['n_collision'] = n_collision

        # Debug print to show actual batch size vs target
        actual_steps = np.sum(batch_lens)
        print(f"  [VecEnv] Collected {actual_steps} steps (Target: {self.timesteps_per_batch}). This creates larger batches and fewer iterations.", flush=True)

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
