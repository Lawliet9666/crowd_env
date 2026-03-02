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

    def add(self, o: np.ndarray, a: np.ndarray, r: float, o2: np.ndarray, d: float):
        self.obs[self.ptr] = np.asarray(o, dtype=np.float32).flatten()
        self.act[self.ptr] = np.asarray(a, dtype=np.float32).flatten()
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = np.asarray(o2, dtype=np.float32).flatten()
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

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
            None,  # importance_weights (PER not used)
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
        act_dim = self.act_dim

        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr
        )

        if cfg.auto_alpha:
            init_log_alpha = float(np.log(cfg.alpha))
            log_alpha = torch.tensor(init_log_alpha, requires_grad=True, device=self.device)
            alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
            target_entropy = (
                cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
            )
        else:
            log_alpha = None
            alpha_opt = None
            target_entropy = None

        def current_alpha() -> torch.Tensor:
            if cfg.auto_alpha:
                return log_alpha.exp()
            return torch.tensor(cfg.alpha, device=self.device)

        # Single env (num_envs=1): obs/action shapes (1, dim)
        o, _ = self.train_envs.reset(seed=self.config.seed)
        if o.ndim > 1:
            o = o[0]
        global_step = 0
        start_time = time.time()
        total_steps = int(cfg.total_steps)
        
        success_count = 0
        collision_count = 0
        timeout_count = 0
        r_float = 0.0  # episodic return for logging (updated when episode ends)
        l_float = 0.0  # episodic length for logging
        
        import pdb; pdb.set_trace()

        # Training loop
        pbar = tqdm(total=total_steps, desc="SAC")
        while global_step < total_steps:
            global_step += 1
            pbar.update(1)

            # Act
            if global_step < cfg.start_timesteps:
                a = self.action_space.sample()
                if a.ndim == 1 and self.config.trainer.num_envs == 1:
                    pass
                else:
                    a = a[0] if a.ndim > 1 else a
            else:
                with torch.no_grad():
                    obs_t = to_tensor(o, self.device).unsqueeze(0)
                    a_t, _, _ = actor.sample(obs_t)
                    a = a_t.squeeze(0).cpu().numpy()
            
            a_env = map_action_to_env(
                a, self.action_low, self.action_high, cfg.action_bound_method
            )
            step_in = np.expand_dims(a_env, axis=0) if a_env.ndim == 1 else a_env
            o2, r, term, trunc, info = self.train_envs.step(step_in)
            done = term | trunc

            # Flatten for single-env
            r_val = float(r[0]) if np.isscalar(r) or r.size == 1 else float(r.ravel()[0])
            d_val = float(term[0]) if term.size else float(term)
            o_flat = o[0] if o.ndim > 1 else o
            o2_flat = o2[0] if o2.ndim > 1 else o2
            a_flat = a_env.flatten() if a_env.ndim > 1 else a_env

            rb.add(o_flat, a_flat, r_val, o2_flat, d_val)
            
            if "episode" in info:
                ep = info["episode"]
                er = np.asarray(ep["r"])
                el = np.asarray(ep["l"])
                r_float = float(er.flat[0]) if er.size else 0.0
                l_float = float(el.flat[0]) if el.size else 0.0
                success_count += ep.get("is_success", [0])[0]
                collision_count += ep.get("is_collision", [0])[0]
                timeout_count += ep.get("is_timeout", [0])[0]
            

            if global_step % self.config.wandb_interval == 0:
                wandb.log(
                    {
                        "train/episodic_return": r_float, 
                        "train/episodic_length": l_float,
                    },
                    step=global_step,
                )
                if success_count + collision_count + timeout_count > 0:
                    wandb.log(
                        {
                            "train/success_rate": success_count / (success_count + collision_count + timeout_count),
                            "train/collision_rate": collision_count / (success_count + collision_count + timeout_count),
                            "train/timeout_rate": timeout_count / (success_count + collision_count + timeout_count),
                        },
                        step=global_step,
                    )
                    success_count = 0
                    collision_count = 0
                    timeout_count = 0

            if np.any(done):
                o, _ = self.train_envs.reset(seed=self.config.seed + global_step)
            else:
                o = o2
            if o.ndim > 1:
                o = o[0]

            # Update
            if global_step >= cfg.start_timesteps and rb.len >= cfg.batch_size:
                for _ in range(cfg.update_per_step):
                    sampled = rb.sample(cfg.batch_size)
                    obs, act, rew, next_obs, done_b = sampled[0]
                    indices, iw = sampled[1], sampled[2]

                    alpha_t = current_alpha()

                    # Critic update
                    with torch.no_grad():
                        next_a, next_logp, _ = actor.sample(next_obs)
                        q_next = torch.min(
                            q1_target(next_obs, next_a), q2_target(next_obs, next_a)
                        )
                        target_q = (
                            rew
                            + cfg.gamma
                            * (1.0 - done_b)
                            * (q_next - alpha_t * next_logp)
                        )

                    q1_val = q1(obs, act)
                    q2_val = q2(obs, act)
                    td1 = (q1_val - target_q).flatten()
                    td2 = (q2_val - target_q).flatten()

                    if iw is not None:
                        iw = iw.to(self.device)
                        critic_loss = (td1.pow(2) * iw).mean() + (td2.pow(2) * iw).mean()
                    else:
                        critic_loss = F.mse_loss(q1_val, target_q) + F.mse_loss(
                            q2_val, target_q
                        )

                    critic_opt.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_opt.step()

                    # Actor update
                    a_pi, logp_pi, _ = actor.sample(obs)
                    q_pi = torch.min(q1(obs, a_pi), q2(obs, a_pi))
                    actor_loss = (alpha_t * logp_pi - q_pi).mean()

                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()

                    # Alpha update
                    alpha_loss_val = None
                    if cfg.auto_alpha:
                        alpha_loss = -(
                            log_alpha * (logp_pi.detach() + target_entropy)
                        ).mean()
                        alpha_opt.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_opt.step()
                        alpha_loss_val = float(alpha_loss.item())

                    soft_update(q1, q1_target, cfg.tau)
                    soft_update(q2, q2_target, cfg.tau)

                    if global_step % self.config.wandb_interval == 0:
                        sps = int(global_step / max(1e-6, (time.time() - start_time)))
                        log_dict = {
                            "global_step": global_step,
                            "charts/sps": sps,
                            "charts/actor_lr": actor_opt.param_groups[0]["lr"],
                            "charts/critic_lr": critic_opt.param_groups[0]["lr"],
                            "loss/critic": float(critic_loss.item()),
                            "loss/actor": float(actor_loss.item()),
                            "stats/q1": float(q1_val.mean().item()),
                            "stats/q2": float(q2_val.mean().item()),
                            "stats/q_pi": float(q_pi.mean().item()),
                            "stats/logp": float(logp_pi.mean().item()),
                            "stats/alpha": float(current_alpha().item()),
                        }
                        with torch.no_grad():
                            ent_approx = actor.entropy_approx(obs).mean().item()
                        log_dict["diagnostics/entropy_approx"] = float(ent_approx)
                        if cfg.auto_alpha:
                            log_dict["loss/alpha"] = alpha_loss_val
                            log_dict["diagnostics/target_entropy"] = float(
                                target_entropy
                            )
                        wandb.log(log_dict, step=global_step)

            # online eval
            if global_step > cfg.start_timesteps and global_step % cfg.eval_interval == 0:
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
                        "global_step": global_step,
                    },
                    step=global_step,
                )
                self.save_ckpt(self.save_dir, global_step, eval_mean)
                pbar.set_postfix(eval_return=f"{eval_mean:.1f}", refresh=False)

        pbar.close()
        self.train_envs.close()
        wandb.finish()
