import torch
import torch.nn as nn
from torch.distributions import Normal
from omegaconf import DictConfig, OmegaConf

from crowd_nav.rl_policy_factory import get_rl_policy_class
from new_rl.utils import MLP, RunningMeanStd


SUPPORTED_SAFETY_METHODS = {
    "rlcbfgamma",
    "rlcvarbetaradius",
    "rlcbfgamma_2nets",
    "rlcvarbetaradius_2nets",
}


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, mlp_config: DictConfig):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_sizes=mlp_config.hidden_sizes, act=mlp_config.act)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


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
        method: str = "rl",
        act_low=None,
        act_high=None,
        env_obs_dim: int | None = None,
        qp_obs_dim: int | None = None,
        qp_start_timesteps: int = 0,
        safe_dist: float = 0.8,
        alpha: float = 2.0,
        beta: float = 0.2,
        robot_type: str = "single_integrator",
        vmax: float = 3.0,
        amax: float = 3.0,
        omega_max: float = 3.0,
        policy_kwargs=None,
        action_std_init: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.method = str(method).strip()
        self.actor_obs_dim = int(obs_dim)
        self.env_obs_dim = int(env_obs_dim) if env_obs_dim is not None else int(obs_dim)
        self.use_dual_actor_input = self.method in SUPPORTED_SAFETY_METHODS
        self.qp_obs_dim = int(qp_obs_dim) if self.use_dual_actor_input else 0

        if self.use_dual_actor_input:
            expected_env_obs_dim = self.actor_obs_dim + self.qp_obs_dim
            if self.env_obs_dim != expected_env_obs_dim:
                raise ValueError(
                    f"Dual-input PPO expects env_obs_dim={expected_env_obs_dim}, got {self.env_obs_dim} "
                    f"(actor={self.actor_obs_dim}, qp={self.qp_obs_dim})."
                )
        elif self.env_obs_dim != self.actor_obs_dim:
            raise ValueError(
                f"RL PPO expects env_obs_dim == actor_obs_dim == {self.actor_obs_dim}, got {self.env_obs_dim}."
            )

        act_low_t = torch.as_tensor(act_low, dtype=torch.float32)
        act_high_t = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("act_low", act_low_t)
        self.register_buffer("act_high", act_high_t)
        self.register_buffer("act_scale", 0.5 * (act_high_t - act_low_t))
        self.register_buffer("act_bias", 0.5 * (act_high_t + act_low_t))

        if self.method == "rl":
            self.actor = ActorMLP(self.actor_obs_dim, act_dim, actor_mlp_config)
            self.actor_outputs_real_action = False
        elif self.method in SUPPORTED_SAFETY_METHODS:
            actor_kwargs = {
                "safe_dist": float(safe_dist),
                "alpha": float(alpha),
                "beta": float(beta),
                "robot_type": str(robot_type),
                "vmax": float(vmax),
                "amax": float(amax),
                "omega_max": float(omega_max),
                "qp_obs_dim": int(self.qp_obs_dim),
                "qp_start_timesteps": int(qp_start_timesteps),
            }
            if policy_kwargs is not None:
                actor_kwargs.update(self._to_dict(policy_kwargs))
            PolicyClass = get_rl_policy_class(self.method)
            self.actor = PolicyClass(self.actor_obs_dim, act_dim, **actor_kwargs)
            self.actor_outputs_real_action = bool(getattr(self.actor, "outputs_real_action", False))
        else:
            raise ValueError(
                f"Unsupported PPO method '{self.method}'. "
                f"Expected one of: rl, rlcbfgamma, rlcvarbetaradius, rlcbfgamma_2nets, rlcvarbetaradius_2nets."
            )

        self.critic = CriticContinuous(self.actor_obs_dim, critic_mlp_config)
        self.logstd = nn.Parameter(torch.log(torch.full((act_dim,), float(action_std_init))))
        self.obs_normalizer = RunningMeanStd(shape=(self.actor_obs_dim,)) if use_obs_norm else None
        self.return_normalizer = RunningMeanStd(shape=()) if use_return_norm else None

    @staticmethod
    def _to_dict(value):
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return dict(value)
        return value

    def set_timestep(self, timestep: int) -> None:
        if hasattr(self.actor, "set_timestep"):
            self.actor.set_timestep(int(timestep))

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if obs.shape[-1] != self.env_obs_dim:
            raise ValueError(f"Expected obs last dim {self.env_obs_dim}, got {obs.shape[-1]}.")
        if not self.use_dual_actor_input:
            return obs, None
        obs_actor = obs[..., : self.actor_obs_dim]
        obs_qp = obs[..., self.actor_obs_dim : self.actor_obs_dim + self.qp_obs_dim]
        return obs_actor, obs_qp

    def _maybe_normalize_obs(self, obs_actor: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer is not None:
            return self.obs_normalizer.normalize(obs_actor)
        return obs_actor

    def update_obs_normalizer(self, obs) -> None:
        if self.obs_normalizer is None:
            return
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.act_low.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_actor, _ = self._split_obs(obs_t)
        self.obs_normalizer.update(obs_actor)

    def _actor_forward(self, obs_actor: torch.Tensor, obs_qp: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_dual_actor_input:
            return self.actor(obs_actor)
        if obs_qp is None:
            raise ValueError("Dual-input safety actor requires relative obs_qp.")
        return self.actor(obs_actor, obs_qp)

    def _build_dist(self, mean_z: torch.Tensor) -> Normal:
        std = self.logstd.exp().clamp(1e-6, 10.0)
        return Normal(mean_z, std)

    def _clip_real_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.act_low
        high = self.act_high
        while low.dim() < action.dim():
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        return torch.max(torch.min(action, high), low)

    def _env_action_to_unit(self, action: torch.Tensor) -> torch.Tensor:
        scale = self.act_scale
        bias = self.act_bias
        while scale.dim() < action.dim():
            scale = scale.unsqueeze(0)
            bias = bias.unsqueeze(0)
        return ((action - bias) / scale.clamp_min(1e-6)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    def _action_mean_from_actor_output(self, actor_output: torch.Tensor) -> torch.Tensor:
        if self.actor_outputs_real_action:
            action_mean = self._clip_real_action(actor_output)
            return self._env_action_to_unit(action_mean)
        return torch.tanh(actor_output)

    def _policy_latent_mean(self, actor_output: torch.Tensor) -> torch.Tensor:
        if self.actor_outputs_real_action:
            unit_mean = self._action_mean_from_actor_output(actor_output)
            return self._atanh(unit_mean)
        return actor_output

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _sample_action(self, mean_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._build_dist(mean_z)
        z = dist.rsample()
        action_unit = torch.tanh(z)
        logprob = dist.log_prob(z).sum(-1) - torch.log(1.0 - action_unit.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action_unit, logprob, entropy

    def _evaluate_action(self, mean_z: torch.Tensor, action_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._build_dist(mean_z)
        z = self._atanh(action_unit)
        logprob = dist.log_prob(z).sum(-1) - torch.log(1.0 - action_unit.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        obs_actor_raw, obs_qp = self._split_obs(obs)
        obs_actor = self._maybe_normalize_obs(obs_actor_raw)
        actor_output = self._actor_forward(obs_actor, obs_qp)
        mean_z = self._policy_latent_mean(actor_output)

        if action is None:
            action, logprob, entropy = self._sample_action(mean_z)
        else:
            action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            logprob, entropy = self._evaluate_action(mean_z, action)

        value = self.critic(obs_actor)
        return action, logprob, entropy, value

    def get_action_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        obs_actor_raw, obs_qp = self._split_obs(obs)
        obs_actor = self._maybe_normalize_obs(obs_actor_raw)
        actor_output = self._actor_forward(obs_actor, obs_qp)
        return self._action_mean_from_actor_output(actor_output)
