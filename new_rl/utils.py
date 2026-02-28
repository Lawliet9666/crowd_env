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
    if act == "relu":
        act = nn.ReLU()
    elif act == "tanh":
        act = nn.Tanh()
    elif act == "silu":
        act = nn.SiLU()
    else:
        raise ValueError(f"Invalid activation function: {act}")
    modules = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            modules.append(nn.Linear(in_dim, hidden_sizes[i]))
        else:
            modules.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        modules.append(act)
    modules.append(nn.Linear(hidden_sizes[-1], out_dim)) 
    if add_last_act:
        modules.append(act)
    return nn.Sequential(*modules)




class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""

    def __init__(self, shape: tuple[int, ...] = (), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Full normalize: (x - mean) / std, clipped."""
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)

    def scale_only(self, x: np.ndarray) -> np.ndarray:
        """Tianshou-style: divide by std only (no mean subtraction)."""
        return x / (np.sqrt(self.var) + 1e-8)

    # Torch variants (for tensors on device)
    def normalize_th(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(np.sqrt(self.var) + 1e-8, dtype=x.dtype, device=x.device)
        return torch.clamp((x - mean) / std, -clip, clip)

    def scale_only_th(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.tensor(np.sqrt(self.var) + 1e-8, dtype=x.dtype, device=x.device)
        return x / std

    def get_std(self) -> float:
        """Return scalar std for logging (handles numpy or torch)."""
        var = self.var
        if hasattr(var, "cpu"):
            std = var.sqrt() + 1e-8
            return float(std.mean() if std.numel() > 1 else std.item())
        return float(np.mean(np.sqrt(np.asarray(var)) + 1e-8))

    def get_mean(self) -> float:
        """Return scalar mean for logging (handles numpy or torch)."""
        m = self.mean
        if hasattr(m, "cpu"):
            return float(m.mean() if m.numel() > 1 else m.item())
        return float(np.mean(m))


def map_action_to_env(
    action: np.ndarray | torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_bound_method: str,
) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    """Bound and scale policy output to env action range."""
    if action_bound_method == "env_clip":
        action = np.clip(action, -1.0, 1.0)
    if action_bound_method == "env_tanh":
        action = np.tanh(action)
    action = action_low + (action_high - action_low) * (action + 1.0) / 2.0
    return action



# ---------------------------------------------------------------------------
# Init (tianshou-style orthogonal)
# ---------------------------------------------------------------------------
@torch.no_grad()
def init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    # Small last layer for near-zero initial actions
    if hasattr(model, "actor"):
        last_actor_layer = list(model.actor.net.children())[-1]
        last_actor_layer.weight.data *= 0.01
    # Critic last layer: gain=1
    if hasattr(model, "critic"):
        last_critic_layer = list(model.critic.net.children())[-1]
        nn.init.orthogonal_(last_critic_layer.weight, gain=1.0)