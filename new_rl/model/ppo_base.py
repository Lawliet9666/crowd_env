import torch
import torch.nn as nn
from torch.distributions import Normal
from new_rl.utils import MLP
from omegaconf import DictConfig

from new_rl.utils import RunningMeanStd

class ActorContinuous(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, mlp_config: DictConfig):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_sizes=mlp_config.hidden_sizes, act=mlp_config.act)
        # log_std as learnable parameter, initialized to -0.5 (tianshou default)
        self.logstd = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = self.logstd.exp().clamp(1e-6, 10.0)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor):
        dist = self._dist(obs)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self._dist(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)


class CriticContinuous(nn.Module):
    def __init__(self, obs_dim: int, mlp_config: DictConfig):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden_sizes=mlp_config.hidden_sizes, act=mlp_config.act)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_mlp_config: DictConfig,
        critic_mlp_config: DictConfig,
        use_obs_norm: bool = True,
        use_return_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.actor = ActorContinuous(obs_dim, act_dim, actor_mlp_config)
        self.critic = CriticContinuous(obs_dim, critic_mlp_config)
        self.obs_normalizer = RunningMeanStd(shape=(obs_dim,)) if use_obs_norm else None
        self.return_normalizer = RunningMeanStd(shape=()) if use_return_norm else None

    def _maybe_normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer is not None:
            return self.obs_normalizer.normalize(obs)
        return obs

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        obs = self._maybe_normalize_obs(obs)
        if action is None:
            action, logprob, entropy = self.actor(obs)
        else:
            logprob, entropy = self.actor.evaluate(obs, action)
        value = self.critic(obs)
        return action, logprob, entropy, value

    def get_action_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """For eval: normalize obs then return actor mean (no sampling)."""
        obs = self._maybe_normalize_obs(obs)
        return self.actor.net(obs)