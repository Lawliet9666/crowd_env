import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def MLP(
    in_dim: int, out_dim: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    act: str = "relu",
    add_last_act: bool = False
) -> nn.Sequential:
    if len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must contain at least one layer size")

    if act == "relu":
        act_cls = nn.ReLU
    elif act == "tanh":
        act_cls = nn.Tanh
    elif act == "silu":
        act_cls = nn.SiLU
    else:
        raise ValueError(f"Invalid activation function: {act}")

    modules = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            modules.append(nn.Linear(in_dim, hidden_sizes[i]))
        else:
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        # Create a fresh module instance per layer.
        modules.append(act_cls())
    modules.append(nn.Linear(hidden_sizes[-1], out_dim))
    if add_last_act:
        modules.append(act_cls())
    return nn.Sequential(*modules)



class RunningMeanStd(nn.Module):
    """Welford online algorithm for running mean and variance.
    Uses register_buffer for mean/var/count — non-trainable, updated on the fly, moves with model.
    """

    def __init__(self, shape: tuple[int, ...] = (), eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(eps, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor | np.ndarray) -> None:
        """Update running stats from batch. x: (batch, *shape). Accepts numpy or tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=self.mean.device, dtype=torch.float64)
        else:
            x = x.double().to(device=self.mean.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.double().var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean.copy_(new_mean)
        self.var.copy_(M2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor | np.ndarray, clip: float = 10.0) -> torch.Tensor:
        """Full normalize: (x - mean) / std, clipped."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=self.mean.device)
        else:
            x = x.to(device=self.mean.device)
        mean = self.mean.to(x.dtype)
        std = (self.var.sqrt() + 1e-8).to(x.dtype)
        return torch.clamp((x - mean) / std, -clip, clip)

    def scale_only(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Tianshou-style: divide by std only (no mean subtraction)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=self.mean.device)
        else:
            x = x.to(device=self.mean.device)
        std = (self.var.sqrt() + 1e-8).to(x.dtype)
        return x / std

    def get_std(self) -> float:
        """Return scalar std for logging."""
        std = self.var.sqrt() + 1e-8
        return float(std.mean() if std.numel() > 1 else std.item())

    def get_mean(self) -> float:
        """Return scalar mean for logging."""
        m = self.mean
        return float(m.mean() if m.numel() > 1 else m.item())


def map_action_to_env(
    action: np.ndarray | torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_bound_method: str,
) -> np.ndarray:
    """Bound and scale policy output from [-1, 1] to environment action range."""
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    if action_bound_method == "env_clip":
        action = np.clip(action, -1.0, 1.0)
    elif action_bound_method == "env_tanh":
        action = np.tanh(action)
    else:
        raise ValueError(
            f"Unsupported action_bound_method '{action_bound_method}'. "
            "Expected one of {'env_clip', 'env_tanh'}."
        )

    action = action_low + (action_high - action_low) * (action + 1.0) / 2.0
    return action



# ---------------------------------------------------------------------------
# Init (tianshou-style orthogonal)
# ---------------------------------------------------------------------------
@torch.no_grad()
def init_weights(model: nn.Module) -> None:
    def _last_linear(module: nn.Module) -> nn.Linear | None:
        linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
        return linears[-1] if linears else None

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    # Small last layer for near-zero initial actions
    if hasattr(model, "actor"):
        last_actor_layer = _last_linear(model.actor.net)
        if last_actor_layer is not None:
            last_actor_layer.weight.data *= 0.01

    # Critic last layer: gain=1
    if hasattr(model, "critic"):
        last_critic_layer = _last_linear(model.critic.net)
        if last_critic_layer is not None:
            nn.init.orthogonal_(last_critic_layer.weight, gain=1.0)
