import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.network_qpth import BarrierNet as BarrierNetV1


class BarrierNet(BarrierNetV1):
    """
    BarrierNet v2 (rlcbfgamma_2nets):
    - u_nom branch uses actor input (polar): fc1 -> fc21 -> fc31
    - alpha branch uses QP input (relative): alpha_fc1 -> alpha_fc2 -> alpha_out
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        qp_obs_dim,
        alpha_hidden1=128,
        alpha_hidden2=64,
        alpha_max=4.0,
        **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, qp_obs_dim=qp_obs_dim, **kwargs)
        self.safety_obs_dim = int(4 * self.obs_topk)  # [clearance, closing, ttc, mask] * K
        self.ttc_cap = 8.0
        self.clearance_scale = max(float(self.safe_dist), 1e-3)
        self.closing_scale = 2.0
        self.alpha_fc1 = nn.Linear(self.safety_obs_dim, int(alpha_hidden1))
        self.alpha_fc2 = nn.Linear(int(alpha_hidden1), int(alpha_hidden2))
        self.alpha_out = nn.Linear(int(alpha_hidden2), 1)
        self.alpha_max = float(alpha_max)
        print(
            f"[CBFNet 2nets] alpha_from=risk_features(qp), alpha_max={self.alpha_max:.3f}",
            flush=True,
        )

    def _build_safety_features(self, obs_qp):
        """
        Build compact risk features for alpha head (keeps QP constraints unchanged):
          per obstacle: [clearance, closing_speed, ttc, mask]
        """
        n_batch = obs_qp.size(0)
        blocks = obs_qp[:, 6:].reshape(n_batch, self.obs_topk, 6)
        rel = blocks[:, :, 0:2]
        human_vel = blocks[:, :, 2:4]
        human_r = blocks[:, :, 4]
        mask = blocks[:, :, 5].clamp(0.0, 1.0)

        robot_vel = obs_qp[:, 2:4].unsqueeze(1)
        robot_r = obs_qp[:, 5:6]

        dist = torch.linalg.norm(rel, dim=2)
        clearance = dist - (robot_r + human_r)

        rel_vel = human_vel - robot_vel
        closing = -torch.sum(rel * rel_vel, dim=2) / (dist + 1e-6)
        closing_pos = torch.clamp(closing, min=0.0)
        ttc = torch.where(
            closing_pos > 1e-4,
            dist / (closing_pos + 1e-6),
            torch.full_like(dist, self.ttc_cap),
        )

        clearance_n = torch.clamp(clearance / self.clearance_scale, -2.0, 4.0)
        closing_n = torch.clamp(closing / self.closing_scale, -4.0, 4.0)
        ttc_n = torch.clamp(ttc / self.ttc_cap, 0.0, 1.0)

        # Padded slots carry neutral risk with mask=0.
        clearance_n = torch.where(mask > 0.5, clearance_n, torch.ones_like(clearance_n))
        closing_n = torch.where(mask > 0.5, closing_n, torch.zeros_like(closing_n))
        ttc_n = torch.where(mask > 0.5, ttc_n, torch.ones_like(ttc_n))

        return torch.stack([clearance_n, closing_n, ttc_n, mask], dim=2).reshape(n_batch, -1)

    def forward(self, obs_actor, obs_qp=None):
        if isinstance(obs_actor, np.ndarray):
            obs_actor = torch.tensor(obs_actor, dtype=torch.float)
        if obs_qp is None:
            raise ValueError("BarrierNet(2nets) requires dual input: obs_actor (polar) and obs_qp (relative).")
        if isinstance(obs_qp, np.ndarray):
            obs_qp = torch.tensor(obs_qp, dtype=torch.float)

        obs_actor = obs_actor.to(self.fc1.weight.device)
        obs_qp = obs_qp.to(self.fc1.weight.device)

        if obs_actor.dim() == 1:
            obs_actor = obs_actor.unsqueeze(0)
        if obs_qp.dim() == 1:
            obs_qp = obs_qp.unsqueeze(0)

        obs_actor = obs_actor.reshape(obs_actor.size(0), -1)
        obs_qp = obs_qp.reshape(obs_qp.size(0), -1)
        if obs_actor.size(1) != self.actor_obs_dim:
            raise ValueError(
                f"BarrierNet(2nets) expected obs_actor dim={self.actor_obs_dim}, got {obs_actor.size(1)}."
            )
        if obs_qp.size(1) != self.qp_obs_dim:
            raise ValueError(
                f"BarrierNet(2nets) expected obs_qp dim={self.qp_obs_dim}, got {obs_qp.size(1)}."
            )

        # u_nom from actor (polar) branch
        x = F.silu(self.fc1(obs_actor))
        x21 = F.silu(self.fc21(x))
        unom = self.fc31(x21)

        # alpha from risk features derived from relative QP observation
        safety_feat = self._build_safety_features(obs_qp)
        xa = F.silu(self.alpha_fc1(safety_feat))
        xa = F.silu(self.alpha_fc2(xa))
        alpha = self.alpha_max * torch.sigmoid(self.alpha_out(xa)).squeeze(-1)
        self.last_alpha = alpha

        # Warmup phase: bypass QP during training.
        if self.training and int(self.current_timestep) < int(self.qp_start_timesteps):
            return unom

        if self.robot_type == "single_integrator":
            return self.dCBF_SI(obs_qp, unom, alpha)
        if self.robot_type == "unicycle":
            return self.dCBF_Unicycle(obs_qp, unom, alpha)
        if self.robot_type == "unicycle_dynamic":
            return torch.zeros_like(unom)
        raise NotImplementedError(f"Robot type {self.robot_type} not supported in BarrierNet(2nets)")
