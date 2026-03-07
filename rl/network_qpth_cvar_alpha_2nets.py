import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction

from rl.my_classes import test_solver as solver
from rl.network_qpth_cvar import (
    BarrierNet as BarrierNetV1,
    cvar_coeff_from_beta,
)


class BarrierNet(BarrierNetV1):
    """
    BarrierNet 2nets (CVaR):
    - u_nom from actor input (polar)
    - alpha / beta / r_safe from QP input (relative), via a separate MLP
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
        self.alpha_fc1 = nn.Linear(self.qp_obs_dim, int(alpha_hidden1))
        self.alpha_fc2 = nn.Linear(int(alpha_hidden1), int(alpha_hidden2))
        self.alpha_out = nn.Linear(int(alpha_hidden2), 1)
        self.beta_out = nn.Linear(int(alpha_hidden2), 1)
        self.rsafe_out = nn.Linear(int(alpha_hidden2), 1)
        self.alpha_max = float(alpha_max)
        print(
            f"[CVaRNet 2nets] alpha/beta/r_safe_from=relative(qp), alpha_max={self.alpha_max:.3f}",
            flush=True,
        )

    def _qp_safety_params_from_obs(self, obs_qp):
        xa = F.silu(self.alpha_fc1(obs_qp))
        xa = F.silu(self.alpha_fc2(xa))

        alpha = self._map_alpha_from_sigmoid(self.alpha_out(xa), default_max=self.alpha_max)
        beta = self._map_beta_from_sigmoid(self.beta_out(xa), default_max=self.beta)
        r_safe_learned = self._map_radius_from_sigmoid(
            self.rsafe_out(xa),
            default_min=self.safe_dist,
            default_max=2.0 * self.safe_dist,
        )
        return alpha, beta, r_safe_learned

    def forward(self, obs_actor, obs_qp=None):
        if isinstance(obs_actor, np.ndarray):
            obs_actor = torch.tensor(obs_actor, dtype=torch.float)
        if obs_qp is None:
            raise ValueError("CVaR-BarrierNet(2nets) requires dual input: obs_actor (polar) and obs_qp (relative).")
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
                f"CVaR-BarrierNet(2nets) expected obs_actor dim={self.actor_obs_dim}, got {obs_actor.size(1)}."
            )
        if obs_qp.size(1) != self.qp_obs_dim:
            raise ValueError(
                f"CVaR-BarrierNet(2nets) expected obs_qp dim={self.qp_obs_dim}, got {obs_qp.size(1)}."
            )

        # u_nom from actor (polar) branch.
        x = F.silu(self.fc1(obs_actor))
        x21 = F.silu(self.fc21(x))
        u_nom = self.fc31(x21)

        # alpha / beta / r_safe from relative QP observation branch.
        alpha, beta, r_safe_learned = self._qp_safety_params_from_obs(obs_qp)
        self.last_alpha = alpha
        self.last_beta = beta
        self.last_r_safe = r_safe_learned

        # Warmup phase: bypass QP during training.
        if self.training and int(self.current_timestep) < int(self.qp_start_timesteps):
            return u_nom

        if self.robot_type == "single_integrator":
            return self.dCVaR_CBF_SI(obs_qp, u_nom, beta, r_safe_learned, alpha=alpha)
        if self.robot_type == "unicycle":
            return self.dCVaR_CBF_Unicycle(obs_qp, u_nom, beta, r_safe_learned, alpha=alpha)
        if self.robot_type == "unicycle_dynamic":
            raise NotImplementedError("dCVaR_CBF_UnicycleDynamic not implemented")
        raise NotImplementedError(f"Robot type {self.robot_type} not supported in CVaR BarrierNet 2nets")

    def dCVaR_CBF_SI(self, obs, u_nom, beta, r_safe_learned, alpha=None):
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        Q = torch.eye(self.nCls, device=device).unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        p = -2 * u_nom
        Q = 2 * Q

        rel, human_vel, mask = self._extract_obstacle_blocks(obs)
        rel_x = rel[:, :, 0]
        rel_y = rel[:, :, 1]
        dist_sq = rel_x**2 + rel_y**2
        h = dist_sq - r_safe_learned.unsqueeze(1) ** 2

        cvar_coeff = cvar_coeff_from_beta(beta).unsqueeze(1).unsqueeze(2)  # (B,1,1)
        means, variances = self._predict_gmm_multi(human_vel)  # (B,K,M,2), (B,K,M)

        rel_norm_sq = (rel**2).sum(dim=2, keepdim=True)  # (B,K,1)
        sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq + 1e-8)
        rel_dot_mu = 2.0 * (means * rel.unsqueeze(2)).sum(dim=3)  # (B,K,M)

        if alpha is None:
            alpha_term = torch.full((nBatch,), float(self.alpha), device=device, dtype=obs.dtype)
        else:
            alpha_term = alpha.to(device=device, dtype=obs.dtype).reshape(-1)
        rhs = (-alpha_term.unsqueeze(1).unsqueeze(2) * h.unsqueeze(2)) + rel_dot_mu + (sigma_f * cvar_coeff)

        tau = 0.1
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=2)  # (B,K)

        G = (-2.0 * rel * mask.unsqueeze(-1)).contiguous()
        h_qp = (-(rhs_wc * mask)).contiguous()

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=self.fc1.weight.device, dtype=torch.float64)

        if self.training:
            x = QPFunction(verbose=0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            if nBatch == 1:
                x = solver(
                    Q_qp[0],
                    p_qp[0],
                    G_qp[0],
                    h_qp_qp[0],
                    device=device,
                    dtype=torch.float32,
                    warm_start_x=self._qp_warm_start,
                )
                self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                x = QPFunction(verbose=0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        return x.to(dtype=obs.dtype)

    def dCVaR_CBF_Unicycle(self, obs, u_nom_xy, beta, r_safe_learned, alpha=None):
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        theta = obs[:, 4]
        rel, human_vel, mask = self._extract_obstacle_blocks(obs)

        epsilon = 0.2
        R_safe = r_safe_learned.unsqueeze(1) + epsilon

        c = torch.cos(theta)
        s = torch.sin(theta)

        J = torch.zeros(nBatch, 2, 2, device=device, dtype=obs.dtype)
        J[:, 0, 0] = c
        J[:, 0, 1] = -epsilon * s
        J[:, 1, 0] = s
        J[:, 1, 1] = epsilon * c

        # Original lookahead-space cost (kept for reference):
        # min ||J u - u_nom_xy||^2
        # JT = J.transpose(1, 2)
        # Q = 2.0 * torch.bmm(JT, J)
        # Q = Q + 1e-6 * torch.eye(self.nCls, device=device, dtype=obs.dtype).unsqueeze(0)
        # p = -2.0 * torch.bmm(JT, u_nom_xy.unsqueeze(-1)).squeeze(-1)

        # New control-space cost (same style as CBF):
        # min ||u - u_nom_vw||^2
        # Here we interpret actor output as nominal unicycle control [v, w].
        u_nom_vw = u_nom_xy
        Q = 2.0 * torch.eye(self.nCls, device=device, dtype=obs.dtype).unsqueeze(0).expand(
            nBatch, self.nCls, self.nCls
        )
        p = -2.0 * u_nom_vw

        heading = torch.stack([c, s], dim=1).unsqueeze(1)
        p_L = rel + epsilon * heading
        h = (p_L[:, :, 0] ** 2 + p_L[:, :, 1] ** 2) - R_safe**2

        lg_v = 2 * (p_L[:, :, 0] * c.unsqueeze(1) + p_L[:, :, 1] * s.unsqueeze(1))
        lg_w = 2 * epsilon * (p_L[:, :, 1] * c.unsqueeze(1) - p_L[:, :, 0] * s.unsqueeze(1))

        cvar_coeff = cvar_coeff_from_beta(beta).unsqueeze(1).unsqueeze(2)
        means, variances = self._predict_gmm_multi(human_vel)
        pL_norm_sq = (p_L**2).sum(dim=2, keepdim=True)
        sigma_f = torch.sqrt(4.0 * variances * pL_norm_sq + 1e-8)
        rel_dot_mu = 2.0 * (means * p_L.unsqueeze(2)).sum(dim=3)

        if alpha is None:
            alpha_term = torch.full((nBatch,), float(self.alpha), device=device, dtype=obs.dtype)
        else:
            alpha_term = alpha.to(device=device, dtype=obs.dtype).reshape(-1)
        rhs = (-alpha_term.unsqueeze(1).unsqueeze(2) * h.unsqueeze(2)) + rel_dot_mu + (sigma_f * cvar_coeff)

        tau = 0.1
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=2)

        G = torch.stack([-(lg_v * mask), -(lg_w * mask)], dim=2).contiguous()
        h_qp = (-(rhs_wc * mask)).contiguous()

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=self.fc1.weight.device, dtype=torch.float64)

        if self.training:
            x = QPFunction(verbose=0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            if nBatch == 1:
                x = solver(
                    Q_qp[0],
                    p_qp[0],
                    G_qp[0],
                    h_qp_qp[0],
                    device=device,
                    dtype=torch.float32,
                    warm_start_x=self._qp_warm_start,
                )
                self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                x = QPFunction(verbose=0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        return x.to(dtype=obs.dtype)
