# sac_minimal_wandb.py

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import wandb
from dataclasses import dataclass


# =========================
# Config (Tianshou-style)
# =========================
@dataclass
class Config:
    task: str = "HalfCheetah-v4"
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    buffer_size: int = 1_000_000
    hidden_sizes: tuple = (256, 256)

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    gamma: float = 0.99
    tau: float = 0.005

    alpha: float = 0.2
    auto_alpha: bool = False
    alpha_lr: float = 3e-4

    start_timesteps: int = 10_000
    batch_size: int = 256

    epoch: int = 50
    epoch_num_steps: int = 5000
    update_per_step: int = 1

    eval_interval: int = 5000   # eval every N env steps (like ppo_mininal)
    eval_episodes: int = 10


# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.size = size
        self.ptr = 0
        self.len = 0

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (
            torch.tensor(self.obs[idx]).to(self.device),
            torch.tensor(self.act[idx]).to(self.device),
            torch.tensor(self.rew[idx]).to(self.device),
            torch.tensor(self.next_obs[idx]).to(self.device),
            torch.tensor(self.done[idx]).to(self.device),
        )


# =========================
# Networks
# =========================
LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, act_low, act_high):
        super().__init__()
        self.net = mlp(obs_dim, hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

        self.register_buffer("scale", torch.tensor((act_high - act_low) / 2.0))
        self.register_buffer("bias", torch.tensor((act_high + act_low) / 2.0))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        tanh = torch.tanh(z)

        action = tanh * self.scale + self.bias

        logp = dist.log_prob(z).sum(-1, keepdim=True)
        logp -= torch.log(1 - tanh.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action, logp, torch.tanh(mu) * self.scale + self.bias


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, o, a):
        return self.q(torch.cat([o, a], dim=-1))


# =========================
# Evaluation (deterministic policy, like ppo_mininal)
# =========================
@torch.no_grad()
def evaluate(env_id, actor, device, episodes=10, seed=1000):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            o_t = torch.tensor(o, dtype=torch.float32).to(device).unsqueeze(0)
            _, _, a = actor.sample(o_t)  # 3rd return = deterministic (mean) action for eval
            o, r, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            total += float(r)
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# =========================
# Main
# =========================
def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    wandb.init(project="sac-minimal", config=vars(cfg))

    env = gym.make(cfg.task)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    actor = Actor(obs_dim, act_dim, cfg.hidden_sizes[0], act_low, act_high).to(device)
    q1 = Critic(obs_dim, act_dim, cfg.hidden_sizes[0]).to(device)
    q2 = Critic(obs_dim, act_dim, cfg.hidden_sizes[0]).to(device)

    q1_target = Critic(obs_dim, act_dim, cfg.hidden_sizes[0]).to(device)
    q2_target = Critic(obs_dim, act_dim, cfg.hidden_sizes[0]).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr)

    # alpha
    if cfg.auto_alpha:
        log_alpha = torch.tensor(np.log(cfg.alpha), requires_grad=True, device=device)
        alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
        target_entropy = -act_dim
    else:
        alpha = cfg.alpha

    rb = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, device)

    o, _ = env.reset(seed=cfg.seed)
    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.epoch):
        for step in range(cfg.epoch_num_steps):
            global_step += 1

            # action
            if global_step < cfg.start_timesteps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a, _, _ = actor.sample(torch.tensor(o).float().to(device).unsqueeze(0))
                    a = a.squeeze(0).cpu().numpy()

            o2, r, term, trunc, _ = env.step(a)
            done = term or trunc

            rb.add(o, a, r, o2, float(done))
            o = o2 if not done else env.reset()[0]

            # update
            if rb.len > cfg.batch_size:
                for _ in range(cfg.update_per_step):
                    obs, act, rew, next_obs, done = rb.sample(cfg.batch_size)

                    # ===== critic =====
                    with torch.no_grad():
                        next_a, next_logp, _ = actor.sample(next_obs)
                        q_next = torch.min(
                            q1_target(next_obs, next_a),
                            q2_target(next_obs, next_a)
                        )
                        target = rew + cfg.gamma * (1 - done) * (q_next - cfg.alpha * next_logp)

                    q1_val = q1(obs, act)
                    q2_val = q2(obs, act)

                    critic_loss = F.mse_loss(q1_val, target) + F.mse_loss(q2_val, target)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # ===== actor =====
                    a_pi, logp_pi, _ = actor.sample(obs)
                    q_pi = torch.min(q1(obs, a_pi), q2(obs, a_pi))

                    actor_loss = (cfg.alpha * logp_pi - q_pi).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # ===== target update =====
                    with torch.no_grad():
                        for p, p_t in zip(q1.parameters(), q1_target.parameters()):
                            p_t.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
                        for p, p_t in zip(q2.parameters(), q2_target.parameters()):
                            p_t.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

                    # ===== logging =====
                    wandb.log({
                        "loss/critic": critic_loss.item(),
                        "loss/actor": actor_loss.item(),
                        "stats/q1": q1_val.mean().item(),
                        "stats/q2": q2_val.mean().item(),
                        "stats/logp": logp_pi.mean().item(),
                        "global_step": global_step
                    })

            # ===== eval during training (like ppo_mininal) =====
            do_eval = cfg.eval_interval > 0 and (global_step % cfg.eval_interval == 0 or global_step == 1)
            if do_eval:
                eval_mean, eval_std = evaluate(cfg.task, actor, device, cfg.eval_episodes, seed=1000)
                sps = int(global_step / (time.time() - start_time))
                print(
                    f"epoch {epoch:3d} | steps {global_step:7d} | "
                    f"eval_return {eval_mean:7.2f}±{eval_std:5.2f} | sps {sps}"
                )
                wandb.log({
                    "charts/eval_return_mean": eval_mean,
                    "charts/eval_return_std": eval_std,
                    "charts/sps": sps,
                    "epoch": epoch,
                    "global_step": global_step
                })

    # Final eval
    eval_mean, eval_std = evaluate(cfg.task, actor, device, episodes=20, seed=2000)
    print(f"\nFinal test over 20 episodes: {eval_mean:.2f} ± {eval_std:.2f}")
    wandb.log({"final/test_return_mean": eval_mean, "final/test_return_std": eval_std}, step=global_step)
    wandb.finish()

    torch.save(actor.state_dict(), "sac_actor.pt")


if __name__ == "__main__":
    main()