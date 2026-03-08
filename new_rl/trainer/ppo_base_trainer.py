import math
import time
import numpy as np
import torch
import torch.nn as nn
import wandb

from new_rl.trainer.trainer import Trainer
from new_rl.utils import map_action_to_env


def _flat(t: torch.Tensor) -> torch.Tensor:
    """Flatten first two dims: (S, N, ...) -> (S*N, ...)."""
    return t.reshape(-1, *t.shape[2:]) if t.dim() > 2 else t.reshape(-1)


class PPOBaseTrainer(Trainer):
    def _update_lr(self, update: int) -> None:
        cfg = self.config.trainer
        num_updates = self.num_updates
        if cfg.lr_schedule == "linear":
            progress = update / num_updates
            new_lr = cfg.lr * max(0.0, 1.0 - progress)
        elif cfg.lr_schedule == "cosine":
            progress = update / num_updates
            new_lr = cfg.lr_min + (cfg.lr - cfg.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            new_lr = cfg.lr
        self.optimizer.param_groups[0]["lr"] = new_lr

    def _update_ent_coef(self, update: int) -> None:
        ent_coef_decay = self.config.trainer.ent_coef_decay
        ent_max = self.config.trainer.ent_coef
        ent_min = self.config.trainer.ent_coef_min
        max_coef_decay_steps = min (3e7 // self.config.trainer.batch_size, self.num_updates) # otherwise entropy explode
        if ent_coef_decay:
            progress = update / max_coef_decay_steps
            self.ent_coef = ent_max + (ent_min - ent_max) * min(1.0, progress)
        else:
            self.ent_coef = ent_max

    def _collect_rollout(
        self,
        obs_np: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        buffers: dict,
    ) -> tuple[torch.Tensor, np.ndarray, int]:
        """Collect one rollout. Returns (next_obs_t, next_obs_np, global_step)."""
        cfg = self.config.trainer
        model = self.model
        device = self.device
        steps_per_env = self.steps_per_env
        obs_normalizer = self.model.obs_normalizer
        
        success_count = 0
        collision_count = 0
        timeout_count = 0

        global_step = buffers["global_step"]
        for step in range(steps_per_env):
            global_step += cfg.num_envs

            if hasattr(model, "update_obs_normalizer"):
                model.update_obs_normalizer(obs_np)
            elif obs_normalizer is not None:
                obs_normalizer.update(obs_np)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                if hasattr(model, "set_timestep"):
                    model.set_timestep(global_step)
                action, logp, _, value = model.get_action_and_value(obs_t)

            action_np = map_action_to_env(action.cpu().numpy(), action_low, action_high, cfg.action_bound_method)
            next_obs_np, rewards, terminated, truncated, infos = self.train_envs.step(action_np)
            dones = np.logical_or(terminated, truncated)

            buffers["obs_buf"][step] = obs_t
            buffers["act_buf"][step] = action
            buffers["logp_buf"][step] = logp
            buffers["rew_buf"][step] = torch.tensor(rewards, dtype=torch.float32, device=device)
            buffers["done_buf"][step] = torch.tensor(dones, dtype=torch.float32, device=device)
            buffers["val_buf"][step] = value
            obs_np = next_obs_np

            if "episode" in infos:
                returns = infos.get("episode", {}).get("r", [])
                lengths = infos.get("episode", {}).get("l", [])
                success_count += np.sum(infos.get("is_success", 0))
                collision_count += np.sum(infos.get("is_collision", 0))
                timeout_count += np.sum(infos.get("is_timeout", 0))
                finished = [(r, lengths[i]) for i, r in enumerate(returns) if r is not None]
                if finished:
                    rets, lens = zip(*finished)
            
        
        if global_step % (self.config.wandb_interval*cfg.num_envs) == 0:
            wandb.log(
                {
                    "train/episodic_return": np.mean(rets), 
                    "train/episodic_length": np.mean(lens),
                    "train/success_rate": success_count / (success_count + collision_count + timeout_count),
                    "train/collision_rate": collision_count / (success_count + collision_count + timeout_count),
                    "train/timeout_rate": timeout_count / (success_count + collision_count + timeout_count),
                },
                step=global_step,
            )

        buffers["global_step"] = global_step

        next_obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)

        return next_obs_t, obs_np, global_step

    def _compute_gae(
        self,
        values: torch.Tensor,
        next_val: torch.Tensor,
        rew_buf: torch.Tensor,
        done_buf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (advantages, returns) both flattened."""
        cfg = self.config.trainer
        steps_per_env = self.steps_per_env
        device = self.device
        return_normalizer = self.model.return_normalizer

        val_2d = values.reshape(steps_per_env, cfg.num_envs)
        if return_normalizer is not None and cfg.return_scale_only:
            std = self.model.return_normalizer.get_std()
            val_2d_gae = val_2d * std
            next_val_gae = next_val * std
        else:
            val_2d_gae = val_2d
            next_val_gae = next_val

        adv_2d = torch.zeros_like(val_2d_gae)
        lastgae = torch.zeros(cfg.num_envs, device=device)
        for t in reversed(range(steps_per_env)):
            nextval = next_val_gae if t == steps_per_env - 1 else val_2d_gae[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nextval * nonterminal - val_2d_gae[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * lastgae
            adv_2d[t] = lastgae

        ret_unnorm = (adv_2d + val_2d_gae).reshape(-1)
        adv_flat = adv_2d.reshape(-1)
        return adv_flat, ret_unnorm

    def _normalize_returns_and_adv(
        self,
        ret_unnorm: torch.Tensor,
        adv_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update return RMS and return (b_ret, b_adv)."""
        
        if self.model.return_normalizer is not None:
            ret_np = ret_unnorm.detach().cpu().numpy()
            self.model.return_normalizer.update(ret_np)
            ret_scaled = self.model.return_normalizer.scale_only(ret_np) if self.config.trainer.return_scale_only else self.model.return_normalizer.normalize(ret_np)
            b_ret = torch.tensor(ret_scaled, dtype=torch.float32, device=self.device) if isinstance(ret_scaled, np.ndarray) else ret_scaled.float().to(self.device)
        else:
            b_ret = ret_unnorm

        if self.config.trainer.use_adv_norm:
            b_adv = (adv_flat - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)
        else:
            b_adv = adv_flat

        return b_ret, b_adv

    def _compute_explained_variance(
        self,
        b_val: torch.Tensor,
        ret_unnorm: torch.Tensor,
    ) -> float:

        with torch.no_grad():
            if self.model.return_normalizer is not None and self.config.trainer.return_scale_only:
                std = self.model.return_normalizer.get_std()
                v_pred_reward = b_val * std
            else:
                v_pred_reward = b_val
            var_y = ret_unnorm.var()
            return float("nan") if var_y == 0 else float(1.0 - (ret_unnorm - v_pred_reward).var() / var_y)

    def _ppo_epoch(
        self,
        b_obs: torch.Tensor,
        b_act: torch.Tensor,
        b_logp: torch.Tensor,
        b_adv: torch.Tensor,
        b_ret: torch.Tensor,
        b_val: torch.Tensor,
        next_obs_t: torch.Tensor,
    ) -> tuple[list, list, list, list, int]:
        """Run PPO update epochs. Returns (mb_pg_losses, mb_vf_losses, approx_kls, clipfracs, epochs_ran)."""

        idxs = torch.arange(self.batch_size, device=self.device)
        clipfracs, approx_kls = [], []
        mb_pg_losses, mb_vf_losses, mb_ent_losses, mb_total_losses = [], [], [], []

        for epoch in range(self.config.trainer.update_epochs):
            if self.config.trainer.recompute_adv and epoch > 0:
                with torch.no_grad():
                    chunks = []
                    for s in range(0, self.batch_size, 512):
                        _, _, _, v = self.model.get_action_and_value(b_obs[s : s + 512])
                        chunks.append(v)
                    new_val = torch.cat(chunks)
                    _, _, _, next_value = self.model.get_action_and_value(next_obs_t)

                adv_flat, ret_unnorm = self._compute_gae(new_val, next_value, self._rew_buf, self._done_buf)
                b_val = new_val
                b_ret, b_adv = self._normalize_returns_and_adv(ret_unnorm, adv_flat)

            perm = idxs[torch.randperm(self.batch_size, device=self.device)]
            for start in range(0, self.batch_size, self.minibatch_size):
                mb_idx = perm[start : start + self.minibatch_size]
                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]
                mb_val = b_val[mb_idx]

                _, newlogp, entropy, newv = self.model.get_action_and_value(mb_obs, mb_act)
                logratio = newlogp - mb_logp
                ratio = logratio.exp()

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self.config.trainer.clip_coef, 1 + self.config.trainer.clip_coef),
                ).mean()

                if self.config.trainer.clip_vloss:
                    v_clipped = mb_val + torch.clamp(newv - mb_val, -self.config.trainer.clip_coef, self.config.trainer.clip_coef)
                    vf_loss = 0.5 * torch.max((newv - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()
                else:
                    vf_loss = 0.5 * (newv - mb_ret).pow(2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + self.config.trainer.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_grad_norm)
                self.optimizer.step()

                mb_pg_losses.append(float(pg_loss.item()))
                mb_vf_losses.append(float(vf_loss.item()))
                mb_ent_losses.append(float(ent_loss.item()))
                mb_total_losses.append(float(loss.item()))

                with torch.no_grad(): # logging
                    approx_kl = float((mb_logp - newlogp).mean())
                    clipfrac = float(((ratio - 1.0).abs() > self.config.trainer.clip_coef).float().mean())
                    approx_kls.append(approx_kl)
                    clipfracs.append(clipfrac)

        return mb_pg_losses, mb_vf_losses, mb_ent_losses, mb_total_losses, approx_kls, clipfracs

    def _log_update(
        self,
        global_step: int,
        approx_kls: list,
        clipfracs: list,
        mb_pg_losses: list,
        mb_vf_losses: list,
        mb_ent_losses: list,
        mb_total_losses: list,
        explained_var: float,
        eval_mean: float,
        eval_std: float,
        success_rate: float,
        collision_rate: float,
        timeout_rate: float,
        sps: int,
    ) -> None:
        log_dict = {
            "global_step": global_step,
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "charts/ent_coef": self.ent_coef,
            "charts/sps": sps,
            "loss/policy": np.mean(mb_pg_losses) if mb_pg_losses else np.nan,
            "loss/value": np.mean(mb_vf_losses) if mb_vf_losses else np.nan,
            "loss/entropy": np.mean(mb_ent_losses) if mb_ent_losses else np.nan,
            "loss/total": np.mean(mb_total_losses) if mb_total_losses else np.nan,
            "diagnostics/approx_kl": np.mean(approx_kls) if approx_kls else np.nan,
            "diagnostics/clipfrac": np.mean(clipfracs) if clipfracs else np.nan,
            "diagnostics/explained_variance": explained_var,
        }
        if not np.isnan(eval_mean):
            log_dict["charts/eval_return_mean"] = eval_mean
            log_dict["charts/eval_return_std"] = eval_std
            if success_rate + collision_rate + timeout_rate > 0:
                log_dict["charts/eval_success_rate"] = success_rate
                log_dict["charts/eval_collision_rate"] = collision_rate
                log_dict["charts/eval_timeout_rate"] = timeout_rate
        if self.model.obs_normalizer is not None:
            log_dict["diagnostics/obs_mean_norm"] = self.model.obs_normalizer.get_mean()
            log_dict["diagnostics/obs_std_norm"] = self.model.obs_normalizer.get_std()
        if self.model.return_normalizer is not None:
            log_dict["diagnostics/ret_mean_norm"] = self.model.return_normalizer.get_mean()
            log_dict["diagnostics/ret_std_norm"] = self.model.return_normalizer.get_std()
        
        wandb.log(log_dict, step=global_step)

    def train(self):        
        action_space = self.train_envs.single_action_space
        action_low = np.asarray(action_space.low, dtype=np.float32)
        action_high = np.asarray(action_space.high, dtype=np.float32)
        
        obs_buf = torch.zeros((self.steps_per_env, self.config.trainer.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        act_buf = torch.zeros((self.steps_per_env, self.config.trainer.num_envs, self.config.env.act_dim), dtype=torch.float32, device=self.device)
        logp_buf = torch.zeros((self.steps_per_env, self.config.trainer.num_envs), dtype=torch.float32, device=self.device)
        rew_buf = torch.zeros((self.steps_per_env, self.config.trainer.num_envs), dtype=torch.float32, device=self.device)
        done_buf = torch.zeros((self.steps_per_env, self.config.trainer.num_envs), dtype=torch.float32, device=self.device)
        val_buf = torch.zeros((self.steps_per_env, self.config.trainer.num_envs), dtype=torch.float32, device=self.device)

        buffers = {
            "obs_buf": obs_buf,
            "act_buf": act_buf,
            "logp_buf": logp_buf,
            "rew_buf": rew_buf,
            "done_buf": done_buf,
            "val_buf": val_buf,
            "global_step": 0,
        }
        self._rew_buf = rew_buf
        self._done_buf = done_buf

        obs_np, _ = self.train_envs.reset(seed=self.config.seed)
        start_time = time.time()

        for update in range(1, self.num_updates + 1):
            self._update_lr(update)
            self._update_ent_coef(update)
            
            if self.use_curriculum:
                num_curriculum_stages = (
                    (self.max_human_num - self.initial_human_num) / self.increase_human_num + 1
                )
                cirriculum_updates = max(1, int(self.num_updates // num_curriculum_stages))
                if update % cirriculum_updates == 0:
                    self.reset_env()
            
            self.model.eval()
            next_obs_t, obs_np, global_step = self._collect_rollout(obs_np, action_low, action_high, buffers)

            b_obs = _flat(obs_buf)
            b_act = _flat(act_buf)
            b_logp = _flat(logp_buf)
            b_val = _flat(val_buf)

            self.model.eval()
            with torch.no_grad():
                _, _, _, next_value = self.model.get_action_and_value(next_obs_t)

            adv_flat, ret_unnorm = self._compute_gae(b_val, next_value, rew_buf, done_buf)
            b_ret, b_adv = self._normalize_returns_and_adv(ret_unnorm, adv_flat)
            explained_var = self._compute_explained_variance(b_val, ret_unnorm)

            self.model.train()
            mb_pg, mb_vf, mb_ent, mb_total, approx_kls, clipfracs = self._ppo_epoch(
                b_obs, b_act, b_logp, b_adv, b_ret, b_val, next_obs_t
            )

            sps = int(global_step / (time.time() - start_time))

            do_eval = update % self.config.trainer.eval_interval == 0 or update == 1 or update == self.num_updates
            if do_eval:
                eval_mean, eval_std, success_rate, collision_rate, timeout_rate = self.eval()
                remain_time = (self.num_updates - update) * ((time.time() - start_time) / 3600) / update if update > 0 else None
                print(
                    f"update {update:4d}/{self.num_updates} | steps {global_step:8d} | "
                    f"eval {eval_mean:8.2f}±{eval_std:5.2f} | "
                    f"success {success_rate:.3f} | collision {collision_rate:.3f} | timeout {timeout_rate:.3f} | "
                    f"kl {np.mean(approx_kls):.4f} | clip {np.mean(clipfracs):.3f} | sps {sps} | time {(time.time() - start_time) / 3600:.2f}hrs | remain {remain_time:.2f}hrs"
                )
                self.save_ckpt(self.save_dir, step=global_step, performance=eval_mean, max_keep=10)
            else:
                eval_mean, eval_std, success_rate, collision_rate, timeout_rate = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

            self._log_update(
                global_step,
                approx_kls,
                clipfracs,
                mb_pg,
                mb_vf,
                mb_ent,
                mb_total,
                explained_var,
                eval_mean,
                eval_std,
                success_rate,
                collision_rate,
                timeout_rate,
                sps,
            )

        self.train_envs.close()
        wandb.finish()
