import torch
import torch.nn as nn
import numpy as np
from new_rl.utils import MLP
from omegaconf import DictConfig

class Actor(nn.Module):
    """Tanh-squashed Gaussian policy with correct log-prob correction (SAC)."""

    def __init__(self, obs_dim: int, act_dim: int, 
                 act_low: np.ndarray, act_high: np.ndarray, 
                 mlp_config: DictConfig):
        super().__init__()
        
        self.backbone = MLP(in_dim=obs_dim, out_dim=mlp_config.hidden_sizes[-1], hidden_sizes=mlp_config.hidden_sizes[:-1], act=mlp_config.act, add_last_act=True)
        self.mu = nn.Linear(mlp_config.hidden_sizes[-1], act_dim)
        self.log_std = nn.Linear(mlp_config.hidden_sizes[-1], act_dim)

        # scale/bias for mapping [-1,1] -> [low,high]
        act_low_t = torch.as_tensor(act_low, dtype=torch.float32)
        act_high_t = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (act_high_t - act_low_t) / 2.0)
        self.register_buffer("bias", (act_high_t + act_low_t) / 2.0)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs: torch.Tensor):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)

        z = dist.rsample()                 # reparameterized
        tanh_z = torch.tanh(z)             # in [-1,1]
        action = tanh_z * self.scale + self.bias

        # log pi(a) with tanh correction
        logp = dist.log_prob(z).sum(-1, keepdim=True)
        logp -= torch.log(1.0 - tanh_z.pow(2) + 1e-6).sum(-1, keepdim=True)

        # deterministic action (for eval): tanh(mean)
        action_mean = torch.tanh(mu) * self.scale + self.bias
        return action, logp, action_mean

    def entropy_approx(self, obs: torch.Tensor) -> torch.Tensor:
        # Approx entropy in pre-tanh space: sum log std + const (not exact after tanh)
        mu, std = self(obs)
        # entropy of Normal: 0.5*log(2*pi*e*std^2)
        ent = 0.5 * torch.log(2 * torch.pi * torch.e * (std ** 2)).sum(-1, keepdim=True)
        return ent


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, 
                 mlp_config: DictConfig):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden_sizes=mlp_config.hidden_sizes, act=mlp_config.act)

    def forward(self, o: torch.Tensor, a: torch.Tensor):
        return self.q(torch.cat([o, a], dim=-1))

class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        actor_mlp_config: DictConfig,
        critic_mlp_config: DictConfig,
        **kwargs
    ):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, act_low, act_high, actor_mlp_config)
        self.q1 = Critic(obs_dim, act_dim, critic_mlp_config)
        self.q2 = Critic(obs_dim, act_dim, critic_mlp_config)
        self.q1_target = Critic(obs_dim, act_dim, critic_mlp_config)
        self.q2_target = Critic(obs_dim, act_dim, critic_mlp_config)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

    def get_action_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic (mean) action for evaluation. Output already in env space [act_low, act_high]."""
        _, _, action_mean = self.actor.sample(obs)
        return action_mean
