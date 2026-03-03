"""
Vectorized-environment SAC wrapper.
Encapsulates vector rollout + replay updates behind the same learn() API as SAC.
"""

import numpy as np
import torch

from rl.sac import SAC


class VecSAC(SAC):
    def __init__(self, policy_class, env, num_envs, **hyperparameters):
        self.num_envs = int(num_envs)
        super().__init__(policy_class=policy_class, env=env, **hyperparameters)

    @staticmethod
    def _extract_final_observation(infos, idx):
        if not isinstance(infos, dict):
            return None
        final_obs = infos.get("final_observation", None)
        if final_obs is None:
            return None
        try:
            obs_i = final_obs[idx]
        except Exception:
            return None
        if obs_i is None:
            return None
        return np.asarray(obs_i, dtype=np.float32)

    def _next_save_step(self):
        if self.save_freq <= 0:
            return None
        if self.save_after_timesteps <= 0:
            return int(self.save_freq)
        k = max(1, (int(self.save_after_timesteps) + int(self.save_freq) - 1) // int(self.save_freq))
        return k * int(self.save_freq)

    def _extract_info_dict(self, infos, idx):
        if isinstance(infos, (tuple, list)):
            info = infos[idx] if idx < len(infos) else {}
            if not isinstance(info, dict):
                return {}
            final_info = info.get("final_info")
            if isinstance(final_info, dict):
                return final_info
            return info

        if isinstance(infos, dict):
            info_i = {}
            for key, val in infos.items():
                if val is None:
                    continue
                try:
                    info_i[key] = val[idx]
                except Exception:
                    continue
            final_info = info_i.get("final_info")
            if isinstance(final_info, dict):
                return final_info
            return info_i

        return {}

    def rollout(self):
        """
        Vector rollout in VecPPO style:
        - collect until timesteps_per_batch steps from completed episodes
        - keep episode statistics per env
        - add every transition to replay buffer (off-policy)
        """
        n_timeout = 0
        n_success = 0
        n_collision = 0
        batch_lens = []
        batch_rews = []

        obs_raw, _ = self.env.reset()
        obs_actor, obs_qp = self._preprocess_obs_pair(obs_raw)
        ep_lens = np.zeros(self.num_envs, dtype=np.int32)
        ep_rets = np.zeros(self.num_envs, dtype=np.float32)

        completed_steps = 0
        env_steps_collected = 0

        while completed_steps < self.timesteps_per_batch:
            self._set_actor_timestep(self._env_steps_total)
            if self._env_steps_total < self.start_timesteps:
                actions = np.stack(
                    [self.env.single_action_space.sample() for _ in range(self.num_envs)], axis=0
                ).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs_actor, dtype=torch.float32, device=self.device)
                    obs_qp_t = None
                    if obs_qp is not None:
                        obs_qp_t = torch.as_tensor(obs_qp, dtype=torch.float32, device=self.device)
                    action_t, _, _ = self._sample_action(obs_t, obs_qp=obs_qp_t, deterministic=False)
                    actions = action_t.detach().cpu().numpy().astype(np.float32)

            next_obs_raw, rews, terminations, truncations, infos = self.env.step(actions)
            dones = np.logical_or(terminations, truncations)
            next_obs_actor, next_obs_qp = self._preprocess_obs_pair(next_obs_raw)

            for i in range(self.num_envs):
                transition_next_obs_actor = next_obs_actor[i]
                transition_next_obs_qp = None if next_obs_qp is None else next_obs_qp[i]
                if bool(dones[i]):
                    final_obs = self._extract_final_observation(infos, i)
                    if final_obs is not None:
                        final_obs_actor, final_obs_qp = self._preprocess_obs_pair(final_obs)
                        transition_next_obs_actor = final_obs_actor
                        if next_obs_qp is not None:
                            transition_next_obs_qp = final_obs_qp

                transition_obs_qp = None if obs_qp is None else obs_qp[i]
                self.replay_buffer.add(
                    obs_actor[i],
                    actions[i],
                    rews[i],
                    transition_next_obs_actor,
                    float(dones[i]),
                    obs_qp=transition_obs_qp,
                    next_obs_qp=transition_next_obs_qp,
                )

                barrier_source = transition_next_obs_qp if transition_next_obs_qp is not None else transition_next_obs_actor
                barrier_val = self._barrier_from_obs(barrier_source)
                if not np.isnan(barrier_val):
                    self.logger["barrier_vals"].append(float(barrier_val))

                ep_lens[i] += 1
                ep_rets[i] += float(rews[i])

                if bool(dones[i]):
                    info_i = self._extract_info_dict(infos, i)
                    n_timeout += int(info_i.get("is_timeout", False))
                    n_success += int(info_i.get("is_success", False))
                    n_collision += int(info_i.get("is_collision", False))

                    batch_lens.append(int(ep_lens[i]))
                    batch_rews.append(float(ep_rets[i]))
                    completed_steps += int(ep_lens[i])
                    ep_lens[i] = 0
                    ep_rets[i] = 0.0

            obs_actor = next_obs_actor
            obs_qp = next_obs_qp
            step_incr = self.num_envs
            env_steps_collected += step_incr
            self._env_steps_total += step_incr

        self.logger["batch_lens"] = batch_lens
        self.logger["batch_rews"] = batch_rews
        self.logger["n_timeout"] = n_timeout
        self.logger["n_success"] = n_success
        self.logger["n_collision"] = n_collision

        print(
            f"  [VecEnv] Collected {completed_steps} steps (Target: {self.timesteps_per_batch}). "
            "This creates larger batches and fewer iterations.",
            flush=True,
        )

        return completed_steps, env_steps_collected

    def learn(self, total_timesteps):
        print(
            f"Learning VecSAC... num_envs={self.num_envs}, "
            f"max_ep_steps={self.max_timesteps_per_episode}, total={total_timesteps}",
            flush=True,
        )

        if not self._same_state_dict(self.q1, self.q1_target):
            self.q2.load_state_dict(self.q1.state_dict())
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q1.state_dict())

        seeds = None if self.seed is None else [int(self.seed) + i for i in range(self.num_envs)]
        self.env.reset(seed=seeds)
        self._env_steps_total = 0
        t_so_far = 0
        i_so_far = 0
        episodes_so_far = 0
        last_eval_episode_count = 0

        next_save_step = self._next_save_step()

        while t_so_far < total_timesteps:
            collected_steps, env_steps_collected = self.rollout()
            t_so_far += int(collected_steps)
            i_so_far += 1
            episodes_so_far += int(len(self.logger.get("batch_lens", [])))

            if self.replay_buffer.len >= self.batch_size:
                # SAC updates are scaled by actual transitions collected in this rollout.
                n_updates = max(1, int(self.updates_per_step) * int(env_steps_collected))
                for _ in range(n_updates):
                    stats = self._update_step()
                    self._accumulate_update_stats(stats)

            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far
            self._log_summary()

            if (
                self.eval_env is not None
                and self.eval_freq_episodes is not None
                and int(self.eval_freq_episodes) > 0
                and self.eval_episodes is not None
                and int(self.eval_episodes) > 0
                and (episodes_so_far - last_eval_episode_count) >= int(self.eval_freq_episodes)
            ):
                self._evaluate_policy_internal(step=t_so_far)
                last_eval_episode_count = episodes_so_far

            if next_save_step is not None:
                while t_so_far >= next_save_step:
                    self._save_models(step=int(next_save_step))
                    next_save_step += int(self.save_freq)

        self._save_models(step=t_so_far)
