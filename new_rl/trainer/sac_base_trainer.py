import os
import shutil
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import wandb

from new_rl.trainer.trainer import Trainer
from new_rl.utils import to_tensor, soft_update, map_action_to_env
from omegaconf import DictConfig, OmegaConf

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.device = device
        self.size = int(size)
        self.ptr = 0
        self.len = 0

        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.size, 1), dtype=np.float32)
        self.done = np.zeros((self.size, 1), dtype=np.float32)

    def add_batch(
        self,
        o: np.ndarray,     # (N, obs_dim)
        a: np.ndarray,     # (N, act_dim)
        r: np.ndarray,     # (N,) or (N,1)
        o2: np.ndarray,    # (N, obs_dim)
        d: np.ndarray,     # (N,) or (N,1)  (terminated|truncated)
    ):
        o = np.asarray(o, dtype=np.float32).reshape(-1, self.obs.shape[1])
        o2 = np.asarray(o2, dtype=np.float32).reshape(-1, self.next_obs.shape[1])
        a = np.asarray(a, dtype=np.float32).reshape(-1, self.act.shape[1])

        r = np.asarray(r, dtype=np.float32).reshape(-1, 1)
        d = np.asarray(d, dtype=np.float32).reshape(-1, 1)

        n = o.shape[0]
        assert o2.shape[0] == n and a.shape[0] == n and r.shape[0] == n and d.shape[0] == n

        # write with wraparound
        end = self.ptr + n
        if end <= self.size:
            sl = slice(self.ptr, end)
            self.obs[sl] = o
            self.act[sl] = a
            self.rew[sl] = r
            self.next_obs[sl] = o2
            self.done[sl] = d
        else:
            first = self.size - self.ptr
            remainder = n - first
            sl1 = slice(self.ptr, self.size)
            sl2 = slice(0, remainder)

            self.obs[sl1] = o[:first]
            self.act[sl1] = a[:first]
            self.rew[sl1] = r[:first]
            self.next_obs[sl1] = o2[:first]
            self.done[sl1] = d[:first]

            self.obs[sl2] = o[first:]
            self.act[sl2] = a[first:]
            self.rew[sl2] = r[first:]
            self.next_obs[sl2] = o2[first:]
            self.done[sl2] = d[first:]

        self.ptr = end % self.size
        self.len = min(self.len + n, self.size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (
            (
                torch.as_tensor(self.obs[idx], device=self.device),
                torch.as_tensor(self.act[idx], device=self.device),
                torch.as_tensor(self.rew[idx], device=self.device),
                torch.as_tensor(self.next_obs[idx], device=self.device),
                torch.as_tensor(self.done[idx], device=self.device),
            ),
            idx,
            None,
        )


class SACBaseTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.config.env.obs_dim,
            act_dim=self.config.env.act_dim,
            size=self.config.trainer.replay_buffer_size,
            device=self.device,
        )

    def setup_wandb(self):
        """SAC-specific run name (avoids PPO keys)."""
        t = self.config.trainer
        run_name = (
            self.config.run_name
            + "-"
            + self.config.model.type
            + f"-bs{t.batch_size}-a{t.alpha}-alr{t.actor_lr:.1e}-clr{t.critic_lr:.1e}"
        )
        if t.auto_alpha:
            run_name += "-auto"
        run_name += f"-{t.action_bound_method}"
        
        if t.use_init_weights:
            run_name += "-iw"
        else:
            run_name += "-noiw"

        self.human_num = t.max_human_num
        self.run_name = run_name
        self.save_dir = os.path.join(self.config.save_dir, self.run_name)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self.config.overwrite:
                raise ValueError(
                    f"Save directory {self.save_dir} already exists, "
                    "please use a different run_name or delete the existing directory"
                )
            else:
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)

        with open(os.path.join(self.save_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.config, f)

        print("config: ", self.config)

        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.run_name,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.use_curriculum = False

    def train(self):
        cfg = self.config.trainer
        actor = self.model.actor
        q1, q2 = self.model.q1, self.model.q2
        q1_target, q2_target = self.model.q1_target, self.model.q2_target
        rb = self.replay_buffer

        num_envs = int(cfg.num_envs)
        print("num_envs: ", num_envs)
        act_dim = self.act_dim

        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr)

        # alpha
        if cfg.auto_alpha:
            init_log_alpha = float(np.log(cfg.alpha))
            log_alpha = torch.tensor(init_log_alpha, requires_grad=True, device=self.device)
            alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
            target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
        else:
            log_alpha = None
            alpha_opt = None
            target_entropy = None

        def current_alpha() -> torch.Tensor:
            if cfg.auto_alpha:
                return log_alpha.exp()
            return torch.tensor(cfg.alpha, device=self.device)

        # reset: obs (num_envs, obs_dim)
        o, info = self.train_envs.reset(seed=self.config.seed)
        o = np.asarray(o, dtype=np.float32)
        assert o.shape[0] == num_envs, f"obs shape mismatch: {o.shape} vs num_envs={num_envs}"

        start_time = time.time()
        # total_steps = total transitions (same meaning as PPO)
        num_updates = int(cfg.total_steps) // num_envs

        success_count = 0
        collision_count = 0
        timeout_count = 0
        ep_return_sum = 0.0
        ep_length_sum = 0.0
        ep_count = 0

        pbar = tqdm(total=num_updates, desc="SAC")

        for update in range(1, num_updates + 1):
            global_step = update * num_envs  # total transitions collected
            pbar.update(1)

            # -------- act (vectorized) --------
            if global_step < cfg.start_timesteps:
                # sample random action for each env in env-space
                # (gymnasium vector env action_space.sample() sometimes returns (act_dim,), sometimes (num_envs, act_dim) depending on wrapper)
                a_env = np.stack([self.action_space.sample() for _ in range(num_envs)], axis=0).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = to_tensor(o, self.device)                 # (N, obs_dim)
                    a_t, _, _ = actor.sample(obs_t)                   # (N, act_dim) in *env space* per your Actor
                    a = a_t.cpu().numpy()

                # IMPORTANT: since your Actor returns env-space actions already,
                # use "model_clip"/"model_tanh" mode (pass-through + clip)
                a_env = map_action_to_env(a, self.action_low, self.action_high, cfg.action_bound_method)

            # env step expects (N, act_dim)
            o2, r, term, trunc, info = self.train_envs.step(a_env)

            o2 = np.asarray(o2, dtype=np.float32)
            r = np.asarray(r, dtype=np.float32).reshape(num_envs)
            term = np.asarray(term, dtype=np.float32).reshape(num_envs)
            trunc = np.asarray(trunc, dtype=np.float32).reshape(num_envs)

            # -------- truncation fix here --------
            if self.config.env.type == "crowdsim":
                done = np.logical_or(term > 0.5, trunc > 0.5).astype(np.float32)  # (N,)
            else:
                done = (term > 0.5).astype(np.float32)

            # store transition (done is correct terminal flag)
            rb.add_batch(o, a_env, r, o2, done)
            
            if "episode" in info:
                returns = info.get("episode", {}).get("r", [])
                lengths = info.get("episode", {}).get("l", [])
                success_count += np.sum(info.get("is_success", 0))
                collision_count += np.sum(info.get("is_collision", 0))
                timeout_count += np.sum(info.get("is_timeout", 0))
                finished = [(r, lengths[i]) for i, r in enumerate(returns) if r is not None]
                if finished:
                    rets, lens = zip(*finished)
                    ep_return_sum += float(np.sum(rets))
                    ep_length_sum += float(np.sum(lens))
                    ep_count += len(rets)

            if global_step % self.config.wandb_interval == 0:
                wandb.log(
                    {
                        "train/episodic_return": ep_return_sum / ep_count, 
                        "train/episodic_length": ep_length_sum / ep_count,
                        "train/success_rate": success_count / (success_count + collision_count + timeout_count),
                        "train/collision_rate": collision_count / (success_count + collision_count + timeout_count),
                        "train/timeout_rate": timeout_count / (success_count + collision_count + timeout_count),
                    },
                    step=global_step,
                )

            # advance obs
            o = o2
            # -------- updates --------
            if global_step >= cfg.start_timesteps and rb.len >= cfg.batch_size:
                for _ in range(cfg.update_per_step):
                    (obs, act, rew, next_obs, done_b), _, _ = rb.sample(cfg.batch_size)
                    alpha_t = current_alpha()

                    # critic target
                    with torch.no_grad():
                        next_a, next_logp, _ = actor.sample(next_obs)
                        q_next = torch.min(q1_target(next_obs, next_a), q2_target(next_obs, next_a))
                        target_q = rew + cfg.gamma * (1.0 - done_b) * (q_next - alpha_t * next_logp)

                    q1_val = q1(obs, act)
                    q2_val = q2(obs, act)
                    critic_loss = F.mse_loss(q1_val, target_q) + F.mse_loss(q2_val, target_q)

                    critic_opt.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_opt.step()

                    # actor
                    a_pi, logp_pi, _ = actor.sample(obs)
                    q_pi = torch.min(q1(obs, a_pi), q2(obs, a_pi))
                    actor_loss = (alpha_t * logp_pi - q_pi).mean()

                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()

                    # alpha
                    alpha_loss_val = None
                    if cfg.auto_alpha:
                        alpha_loss = -(log_alpha * (logp_pi.detach() + target_entropy)).mean()
                        alpha_opt.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_opt.step()
                        alpha_loss_val = float(alpha_loss.item())

                    soft_update(q1, q1_target, cfg.tau)
                    soft_update(q2, q2_target, cfg.tau)

                    if global_step % self.config.wandb_interval == 0:
                        sps = int(global_step / max(1e-6, (time.time() - start_time)))
                        log_dict = {
                            "charts/sps": sps,
                            "loss/critic": float(critic_loss.item()),
                            "loss/actor": float(actor_loss.item()),
                            "stats/q1": float(q1_val.mean().item()),
                            "stats/q2": float(q2_val.mean().item()),
                            "stats/q_pi": float(q_pi.mean().item()),
                            "stats/logp": float(logp_pi.mean().item()),
                            "stats/alpha": float(current_alpha().item()),
                        }
                        with torch.no_grad():
                            log_dict["diagnostics/entropy_approx"] = float(actor.entropy_approx(obs).mean().item())
                        if cfg.auto_alpha:
                            log_dict["loss/alpha"] = alpha_loss_val
                            log_dict["diagnostics/target_entropy"] = float(target_entropy)
                        wandb.log(log_dict, step=global_step)

            # -------- eval --------
            if global_step >= cfg.start_timesteps and global_step % cfg.eval_interval == 0:
                sps = int(global_step / max(1e-6, (time.time() - start_time)))
                eval_mean, eval_std, success, collision, timeout = self.eval()
                print(
                    f"steps {global_step:7d} | eval_return {eval_mean:7.2f}±{eval_std:5.2f} | "
                    f"success {success:.2%} | collision {collision:.2%} | timeout {timeout:.2%} | sps {sps}"
                )
                wandb.log(
                    {
                        "charts/eval_return_mean": eval_mean,
                        "charts/eval_return_std": eval_std,
                        "charts/eval_success_rate": success,
                        "charts/eval_collision_rate": collision,
                        "charts/eval_timeout_rate": timeout,
                        "charts/sps": sps,
                    },
                    step=global_step,
                )
                self.save_ckpt(self.save_dir, global_step, eval_mean)
                pbar.set_postfix(eval_return=f"{eval_mean:.1f}", refresh=False)

        pbar.close()
        self.train_envs.close()
        wandb.finish()