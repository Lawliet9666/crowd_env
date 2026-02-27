"""
Soft Actor-Critic training class adapted to this project's PPO-style interface.
"""

import inspect
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.size = int(size)
        self.ptr = 0
        self.len = 0

        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.size, 1), dtype=np.float32)
        self.done = np.zeros((self.size, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.act[self.ptr] = np.asarray(act, dtype=np.float32).reshape(-1)
        self.rew[self.ptr] = float(rew)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.act[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.rew[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.done[idx], dtype=torch.float32, device=self.device),
        )


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        dims = [obs_dim + act_dim, *hidden_sizes, 1]
        self.net = _mlp(dims)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class SAC:
    def __init__(self, policy_class, env, **hyperparameters):
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.deterministic = False

        if hasattr(env, "single_observation_space"):
            obs_space = env.single_observation_space
            act_space = env.single_action_space
        else:
            obs_space = env.observation_space
            act_space = env.action_space

        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(act_space.shape))
        self.action_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)

        finite_mask = np.isfinite(self.action_low) & np.isfinite(self.action_high)
        if not np.all(finite_mask):
            self.action_low = np.where(finite_mask, self.action_low, -1.0)
            self.action_high = np.where(finite_mask, self.action_high, 1.0)

        self.action_scale = torch.as_tensor(
            0.5 * (self.action_high - self.action_low), dtype=torch.float32, device=self.device
        )
        self.action_bias = torch.as_tensor(
            0.5 * (self.action_high + self.action_low), dtype=torch.float32, device=self.device
        )
        self._log_action_scale = torch.sum(torch.log(self.action_scale.abs() + 1e-6)).detach()

        self.actor = self._build_actor(policy_class).to(self.device)
        log_std_init = float(np.log(max(float(self.action_std_init), 1e-3)))
        self.actor_log_std = nn.Parameter(
            torch.full((self.act_dim,), log_std_init, dtype=torch.float32, device=self.device)
        )

        self.q1 = QNetwork(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.q1_target = QNetwork(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Compatibility with existing warm-start path in main.py
        self.critic = self.q1

        self.actor_optim = torch.optim.Adam(
            [{"params": self.actor.parameters()}, {"params": [self.actor_log_std]}],
            lr=self.actor_lr,
        )
        self.critic_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.critic_lr,
        )

        self.alpha_value = float(self.alpha)
        self.log_alpha = None
        self.alpha_optim = None
        if self.auto_alpha:
            init_log_alpha = np.log(max(self.alpha_value, 1e-6))
            self.log_alpha = torch.tensor(init_log_alpha, dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            if self.target_entropy is None:
                self.target_entropy = -float(self.act_dim)

        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size, self.device)

        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,
            "i_so_far": 0,
            "batch_lens": [],
            "batch_rews": [],
            "actor_losses": [],
            "critic_losses": [],
            "actor_grads": [],
            "critic_grads": [],
            "q1_vals": [],
            "q2_vals": [],
            "logp_vals": [],
            "alpha_vals": [],
            "mu_means": [],
            "sigma_means": [],
            "barrier_vals": [],
            "n_timeout": 0,
            "n_success": 0,
            "n_collision": 0,
        }

    def learn(self, total_timesteps):
        print(
            f"Learning SAC... {self.max_timesteps_per_episode} max episode steps, "
            f"log interval {self.timesteps_per_batch}, total {total_timesteps} timesteps",
            flush=True,
        )

        if not self._same_state_dict(self.q1, self.q1_target):
            # q1 was likely warm-started externally through model.critic in main.py.
            self.q2.load_state_dict(self.q1.state_dict())
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q1.state_dict())

        episode_idx = 0
        last_eval_episode_count = 0
        ep_len = 0
        ep_ret = 0.0
        t_so_far = 0
        next_save_step = self._next_save_step()

        obs, _ = self.env.reset(seed=self.seed)

        while t_so_far < total_timesteps:
            t_so_far += 1
            if self.render and (self.logger["i_so_far"] % max(1, self.render_every_i) == 0):
                self.env.render()

            if t_so_far < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action, _ = self.get_action(obs, deterministic=False)

            next_obs, rew, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            self.replay_buffer.add(obs, action, rew, next_obs, float(done))
            obs = next_obs
            ep_ret += float(rew)
            ep_len += 1

            barrier_val = self._barrier_from_obs(obs)
            if not np.isnan(barrier_val):
                self.logger["barrier_vals"].append(float(barrier_val))

            if self.replay_buffer.len >= self.batch_size:
                for _ in range(self.updates_per_step):
                    stats = self._update_step()
                    self._accumulate_update_stats(stats)

            if done or ep_len >= self.max_timesteps_per_episode:
                self.logger["batch_lens"].append(ep_len)
                self.logger["batch_rews"].append(ep_ret)
                if isinstance(info, dict):
                    self.logger["n_timeout"] += int(info.get("is_timeout", False))
                    self.logger["n_success"] += int(info.get("is_success", False))
                    self.logger["n_collision"] += int(info.get("is_collision", False))

                episode_idx += 1
                if (
                    self.eval_env is not None
                    and self.eval_freq_episodes is not None
                    and int(self.eval_freq_episodes) > 0
                    and self.eval_episodes is not None
                    and int(self.eval_episodes) > 0
                    and (episode_idx - last_eval_episode_count) >= int(self.eval_freq_episodes)
                ):
                    self._evaluate_policy_internal(step=t_so_far)
                    last_eval_episode_count = episode_idx

                reset_seed = None if self.seed is None else int(self.seed) + episode_idx
                obs, _ = self.env.reset(seed=reset_seed)
                ep_len = 0
                ep_ret = 0.0

            if self.timesteps_per_batch > 0 and (t_so_far % self.timesteps_per_batch == 0):
                self.logger["t_so_far"] = t_so_far
                self.logger["i_so_far"] += 1
                self._log_summary()

            if next_save_step is not None:
                while t_so_far >= next_save_step:
                    self._save_models(step=int(next_save_step))
                    next_save_step += int(self.save_freq)

        self._save_models(step=t_so_far)

    def get_action(self, obs, deterministic=None):
        if deterministic is None:
            deterministic = bool(self.deterministic)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action_t, logp_t, mean_t = self._sample_action(obs_t, deterministic=deterministic)

        if deterministic:
            action_t = mean_t

        action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if logp_t is None:
            logp = 1.0
        else:
            logp = float(logp_t.squeeze(0).detach().cpu().item())
        return action, logp

    def _sample_action(self, obs, deterministic=False):
        mu = self.actor(obs)
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        log_std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).unsqueeze(0).expand_as(mu)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        pre_tanh = mu if deterministic else dist.rsample()
        tanh_out = torch.tanh(pre_tanh)
        action = tanh_out * self.action_scale + self.action_bias
        mean_action = torch.tanh(mu) * self.action_scale + self.action_bias

        if deterministic:
            return action, None, mean_action

        logp = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        logp -= torch.log(1 - tanh_out.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        logp -= self._log_action_scale
        return action, logp, mean_action

    def _update_step(self):
        obs, act, rew, next_obs, done = self.replay_buffer.sample(self.batch_size)
        alpha = self._alpha_tensor()

        with torch.no_grad():
            next_action, next_logp, _ = self._sample_action(next_obs, deterministic=False)
            q_next = torch.min(self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action))
            target = rew + self.gamma * (1.0 - done) * (q_next - alpha * next_logp)

        q1_val = self.q1(obs, act)
        q2_val = self.q2(obs, act)
        critic_loss = F.mse_loss(q1_val, target) + F.mse_loss(q2_val, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad = self._grad_norm(list(self.q1.parameters()) + list(self.q2.parameters()))
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.max_grad_norm)
        self.critic_optim.step()

        action_pi, logp_pi, _ = self._sample_action(obs, deterministic=False)
        q_pi = torch.min(self.q1(obs, action_pi), self.q2(obs, action_pi))
        actor_loss = (alpha * logp_pi - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad = self._grad_norm(list(self.actor.parameters()) + [self.actor_log_std])
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.actor_log_std], self.max_grad_norm)
        self.actor_optim.step()

        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_value = float(self.log_alpha.exp().detach().cpu().item())

        with torch.no_grad():
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

        with torch.no_grad():
            mu = self.actor(obs)
            log_std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            sigma = torch.exp(log_std)

        stats = {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "q1": float(q1_val.mean().detach().cpu().item()),
            "q2": float(q2_val.mean().detach().cpu().item()),
            "logp": float(logp_pi.mean().detach().cpu().item()),
            "alpha": float(self.alpha_value),
            "actor_grad": float(actor_grad),
            "critic_grad": float(critic_grad),
            "mu_mean": float(mu.mean().detach().cpu().item()),
            "sigma_mean": float(sigma.mean().detach().cpu().item()),
        }
        if alpha_loss is not None:
            stats["alpha_loss"] = float(alpha_loss.detach().cpu().item())
        return stats

    def _accumulate_update_stats(self, stats):
        self.logger["actor_losses"].append(stats["actor_loss"])
        self.logger["critic_losses"].append(stats["critic_loss"])
        self.logger["actor_grads"].append(stats["actor_grad"])
        self.logger["critic_grads"].append(stats["critic_grad"])
        self.logger["q1_vals"].append(stats["q1"])
        self.logger["q2_vals"].append(stats["q2"])
        self.logger["logp_vals"].append(stats["logp"])
        self.logger["alpha_vals"].append(stats["alpha"])
        self.logger["mu_means"].append(stats["mu_mean"])
        self.logger["sigma_means"].append(stats["sigma_mean"])

    def _log_summary(self):
        prev_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - prev_t) / 1e9

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = float(np.mean(self.logger["batch_lens"])) if self.logger["batch_lens"] else 0.0
        avg_ep_rews = float(np.mean(self.logger["batch_rews"])) if self.logger["batch_rews"] else 0.0
        avg_actor_loss = float(np.mean(self.logger["actor_losses"])) if self.logger["actor_losses"] else 0.0
        avg_critic_loss = float(np.mean(self.logger["critic_losses"])) if self.logger["critic_losses"] else 0.0
        avg_actor_grad = float(np.mean(self.logger["actor_grads"])) if self.logger["actor_grads"] else 0.0
        avg_critic_grad = float(np.mean(self.logger["critic_grads"])) if self.logger["critic_grads"] else 0.0
        avg_q1 = float(np.mean(self.logger["q1_vals"])) if self.logger["q1_vals"] else 0.0
        avg_q2 = float(np.mean(self.logger["q2_vals"])) if self.logger["q2_vals"] else 0.0
        avg_logp = float(np.mean(self.logger["logp_vals"])) if self.logger["logp_vals"] else 0.0
        avg_alpha = float(np.mean(self.logger["alpha_vals"])) if self.logger["alpha_vals"] else self.alpha_value
        avg_mu = float(np.mean(self.logger["mu_means"])) if self.logger["mu_means"] else 0.0
        avg_sigma = float(np.mean(self.logger["sigma_means"])) if self.logger["sigma_means"] else 0.0
        min_barrier = float(np.min(self.logger["barrier_vals"])) if self.logger["barrier_vals"] else np.nan
        avg_barrier = float(np.mean(self.logger["barrier_vals"])) if self.logger["barrier_vals"] else np.nan

        n_episodes = max(len(self.logger["batch_lens"]), 1)
        timeout_rate = float(self.logger["n_timeout"]) / n_episodes
        success_rate = float(self.logger["n_success"]) / n_episodes
        collision_rate = float(self.logger["n_collision"]) / n_episodes

        if wandb.run is not None:
            payload = {
                "iteration": i_so_far,
                "timesteps": t_so_far,
                "ep_len": avg_ep_lens,
                "ep_reward": avg_ep_rews,
                "loss/actor": avg_actor_loss,
                "loss/critic": avg_critic_loss,
                "stats/q1": avg_q1,
                "stats/q2": avg_q2,
                "stats/logp": avg_logp,
                "stats/alpha": avg_alpha,
                "actor_grad_norm": avg_actor_grad,
                "critic_grad_norm": avg_critic_grad,
                "action_mu": avg_mu,
                "action_sigma": avg_sigma,
                "timeout_rate": timeout_rate,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "iteration_time": float(delta_t),
            }
            if not np.isnan(min_barrier):
                payload["barrier_min_batch"] = min_barrier
            if not np.isnan(avg_barrier):
                payload["barrier_avg_batch"] = avg_barrier
            wandb.log(payload, step=t_so_far)

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens:.2f}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews:.2f}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss:.5f}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss:.5f}", flush=True)
        print(f"Average Actor Grad Norm: {avg_actor_grad:.5f}", flush=True)
        print(f"Average Critic Grad Norm: {avg_critic_grad:.5f}", flush=True)
        print(f"Average Q1: {avg_q1:.5f}", flush=True)
        print(f"Average Q2: {avg_q2:.5f}", flush=True)
        print(f"Average LogPi: {avg_logp:.5f}", flush=True)
        print(f"Average Alpha: {avg_alpha:.5f}", flush=True)
        print(f"Average Action Mu: {avg_mu:.5f}", flush=True)
        print(f"Average Action Sigma: {avg_sigma:.5f}", flush=True)
        if not np.isnan(min_barrier):
            print(f"Min Barrier Value: {min_barrier:.5f}", flush=True)
            print(f"Average Barrier Value: {avg_barrier:.5f}", flush=True)
        print(f"Timeout Rate: {timeout_rate:.5f}", flush=True)
        print(f"Success Rate: {success_rate:.5f}", flush=True)
        print(f"Collision Rate: {collision_rate:.5f}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t:.2f} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
        self.logger["critic_losses"] = []
        self.logger["actor_grads"] = []
        self.logger["critic_grads"] = []
        self.logger["q1_vals"] = []
        self.logger["q2_vals"] = []
        self.logger["logp_vals"] = []
        self.logger["alpha_vals"] = []
        self.logger["mu_means"] = []
        self.logger["sigma_means"] = []
        self.logger["barrier_vals"] = []
        self.logger["n_timeout"] = 0
        self.logger["n_success"] = 0
        self.logger["n_collision"] = 0

    @torch.no_grad()
    def _evaluate_policy_internal(self, step):
        if self.eval_env is None:
            return
        episodes = int(self.eval_episodes)
        if episodes <= 0:
            return

        returns = []
        lengths = []
        success_count = 0
        collision_count = 0

        for ep in range(episodes):
            eval_seed = None if self.eval_seed is None else int(self.eval_seed) + ep
            obs, _ = self.eval_env.reset(seed=eval_seed)

            done = False
            ep_ret = 0.0
            ep_len = 0
            ep_success = False
            ep_collision = False

            while not done and ep_len < int(self.max_timesteps_per_episode):
                action, _ = self.get_action(obs, deterministic=True)
                obs, rew, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(rew)
                ep_len += 1

                if isinstance(info, dict):
                    ep_success = ep_success or bool(info.get("is_success", False))
                    ep_collision = ep_collision or bool(info.get("is_collision", False))

            returns.append(ep_ret)
            lengths.append(ep_len)
            if ep_success and not ep_collision:
                success_count += 1
            if ep_collision:
                collision_count += 1

        eval_ret_mean = float(np.mean(returns)) if returns else 0.0
        eval_ret_std = float(np.std(returns)) if returns else 0.0
        eval_len_mean = float(np.mean(lengths)) if lengths else 0.0
        success_rate = float(success_count) / float(max(episodes, 1))
        collision_rate = float(collision_count) / float(max(episodes, 1))

        print(
            f"[SAC Eval] step={step} return={eval_ret_mean:.2f}±{eval_ret_std:.2f} "
            f"len={eval_len_mean:.1f} success={success_rate:.3f} collision={collision_rate:.3f}",
            flush=True,
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/return_mean": eval_ret_mean,
                    "eval/return_std": eval_ret_std,
                    "eval/ep_length_mean": eval_len_mean,
                    "eval/success_rate": success_rate,
                    "eval/collision_rate": collision_rate,
                },
                step=step,
            )

    def _save_models(self, step):
        os.makedirs(self.save_dir, exist_ok=True)

        actor_path = os.path.join(self.save_dir, "sac_actor.pth")
        critic_path = os.path.join(self.save_dir, "sac_critic.pth")
        checkpoint_path = os.path.join(self.save_dir, "sac_checkpoint.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.q1.state_dict(), critic_path)
        torch.save(
            {
                "step": step,
                "actor": self.actor.state_dict(),
                "actor_log_std": self.actor_log_std.detach().cpu(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "alpha_value": self.alpha_value,
                "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
                "alpha_optim": None if self.alpha_optim is None else self.alpha_optim.state_dict(),
            },
            checkpoint_path,
        )
        print(f"[SAC] Saved checkpoints at step {step}: {actor_path}, {critic_path}", flush=True)

    def _next_save_step(self):
        if int(self.save_freq) <= 0:
            return None
        if int(self.save_after_timesteps) <= 0:
            return int(self.save_freq)
        k = max(1, (int(self.save_after_timesteps) + int(self.save_freq) - 1) // int(self.save_freq))
        return k * int(self.save_freq)

    def _build_actor(self, policy_class):
        actor_kwargs = {
            "safe_dist": self.safe_dist,
            "alpha": self.cbf_alpha,
            "beta": self.cvar_beta,
            "robot_type": self.robot_type,
            "vmax": self.vmax,
            "amax": self.amax,
            "omega_max": self.omega_max,
        }

        try:
            sig = inspect.signature(policy_class.__init__)
            accepted = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in actor_kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            filtered_kwargs = {}

        return policy_class(self.obs_dim, self.act_dim, **filtered_kwargs)

    @staticmethod
    def _same_state_dict(module_a, module_b):
        for p_a, p_b in zip(module_a.parameters(), module_b.parameters()):
            if not torch.allclose(p_a.detach(), p_b.detach()):
                return False
        return True

    @staticmethod
    def _grad_norm(params):
        total_sq = 0.0
        for p in params:
            if p.grad is None:
                continue
            grad_norm = p.grad.detach().data.norm(2).item()
            total_sq += grad_norm * grad_norm
        return total_sq ** 0.5

    def _soft_update(self, source, target):
        with torch.no_grad():
            for p_src, p_tgt in zip(source.parameters(), target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p_src.data)

    def _alpha_tensor(self):
        if self.auto_alpha and self.log_alpha is not None:
            return self.log_alpha.exp().detach()
        return torch.tensor(self.alpha_value, dtype=torch.float32, device=self.device)

    def _barrier_from_obs(self, obs):
        # Observation layout (first obstacle block): idx 6 rel_x, idx 7 rel_y, idx 11 mask.
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_arr.shape[0] >= 12 and obs_arr[11] > 0.5:
            rel_x = float(obs_arr[6])
            rel_y = float(obs_arr[7])
            return rel_x * rel_x + rel_y * rel_y - float(self.safe_dist) ** 2
        return np.nan

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 2_000
        self.max_timesteps_per_episode = 200
        self.n_updates_per_iteration = 8
        self.batch_size = 256
        self.buffer_size = 500_000
        self.start_timesteps = 15_000
        self.updates_per_step = 1

        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.actor_lr = 3e-4
        self.critic_lr = 5e-4
        self.hidden_sizes = (256, 256)
        self.max_grad_norm = 1.0

        self.alpha = 0.10
        self.auto_alpha = True
        self.alpha_lr = 3e-4
        self.target_entropy = -1.2
        self.action_std_init = 0.20

        self.render = False
        self.render_every_i = 50
        self.save_after_timesteps = 0
        self.save_freq = 1_000_000
        self.save_dir = "./"
        self.seed = None
        self.eval_seed = None
        self.eval_env = None
        self.eval_freq_episodes = 0
        self.eval_episodes = 20
        self.device = torch.device("cpu")

        self.safe_dist = 0.8
        self.cbf_alpha = 2.0
        self.cvar_beta = 0.2
        self.robot_type = "single_integrator"
        self.vmax = 1.0
        self.amax = 1.0
        self.omega_max = 1.0
        self.env_name = ""

        for param, val in hyperparameters.items():
            if param == "device":
                self.device = val if isinstance(val, torch.device) else torch.device(val)
            else:
                setattr(self, param, val)

        if self.actor_lr is None:
            self.actor_lr = float(self.lr)
        if self.critic_lr is None:
            self.critic_lr = float(self.lr)
        if isinstance(self.hidden_sizes, list):
            self.hidden_sizes = tuple(self.hidden_sizes)
        if self.updates_per_step is None:
            ratio = max(1.0, float(self.timesteps_per_batch) / float(max(1, self.batch_size)))
            self.updates_per_step = max(1, int(round(float(self.n_updates_per_iteration) / ratio)))
        if self.eval_seed is None:
            self.eval_seed = self.seed

        os.makedirs(self.save_dir, exist_ok=True)

        if self.seed is not None:
            assert isinstance(self.seed, int)
            _set_seed(self.seed)
            print(f"Successfully set seed to {self.seed}", flush=True)
