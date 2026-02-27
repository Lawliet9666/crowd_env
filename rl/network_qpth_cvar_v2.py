import torch.nn as nn
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction
import numpy as np
from rl.my_classes import test_solver as solver
from controller.traj_prediction_batch import TrajPredictorTorch as TrajPredictor
from scipy.stats import norm

import math

def normal_ppf(p: torch.Tensor) -> torch.Tensor:
    # Φ^{-1}(p) = sqrt(2) * erfinv(2p-1)
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)

def normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * z * z)

def cvar_coeff_from_beta(beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    beta = beta.clamp(min=eps, max=1.0 - eps)
    z = normal_ppf(1.0 - beta)
    pdf = normal_pdf(z)
    return pdf / beta

class BarrierNet(nn.Module):
    def __init__(self, nFeatures, 
                 nCls, 
                 nHidden1 = 256, 
                 nHidden21 = 256, 
                 nHidden22 = 256, 
                 safe_dist=0.8, 
                 alpha=2.0, 
                 beta=0.1,
                 robot_type='single_integrator', vmax=3.0, amax=3.0, omega_max=3.0, **kwargs):
        super().__init__()
        self.nFeatures = nFeatures
        self.nCls = nCls

        self.safe_dist = safe_dist
        self.alpha = alpha   
        self.beta = beta
        self.robot_type = robot_type

        self.last_alpha = alpha
        self.last_beta = beta
        self._qp_warm_start = None

        if self.robot_type == 'single_integrator':
            self.u_min = [-vmax, -vmax]
            self.u_max = [vmax, vmax]
            self._cbf_name = "dCVaR_CBF_SI"
        elif self.robot_type == 'unicycle':
            self.u_min = [-vmax, -omega_max]
            self.u_max = [vmax, omega_max]
            self._cbf_name = "dCVaR_CBF_Unicycle"
        elif self.robot_type == 'unicycle_dynamic':
            self.u_min = [-omega_max, -amax]
            self.u_max = [omega_max, amax]
            self._cbf_name = "dCVaR_CBF_UnicycleDynamic"
        else:
            self.u_min = None
            self.u_max = None
            self._cbf_name = "unknown"

        print(
            f"[CVaRNet v2] robot_type={self.robot_type}, safe_dist={self.safe_dist:.3f}, umax={self.u_max}",
            flush=True,
        )
        self.predictor = TrajPredictor()

        self.fc1 = nn.Linear(nFeatures, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21) 
        self.fc22 = nn.Linear(nHidden1, nHidden22)
        self.fc23 = nn.Linear(nHidden1, nHidden22)  # for radius 
        self.fc31 = nn.Linear(nHidden21, nCls) 
        self.fc32 = nn.Linear(nHidden22, 1) 
        self.fc33 = nn.Linear(nHidden22, 1)  # for radius

    def _extract_obstacle_blocks(self, obs):
        """
        Parse obstacle observations to a fixed tensor form:
          rel:  (B, K, 2)   [p_r - p_h]
          vel:  (B, K, 2)   [v_hx, v_hy]
          mask: (B, K)      1 real / 0 dummy

        Supports:
        1) New local-sensing format: [robot(6), K * (rel_x, rel_y, vx, vy, radius, mask)]
        2) Legacy single-obstacle format: [robot(6), rel_x, rel_y, vx, vy, radius]
        """
        nBatch = obs.size(0)
        device = obs.device
        dtype = obs.dtype

        if obs.size(1) >= 12 and ((obs.size(1) - 6) % 6 == 0):
            blocks = obs[:, 6:].reshape(nBatch, -1, 6)
            rel = blocks[:, :, 0:2]
            vel = blocks[:, :, 2:4]
            mask = blocks[:, :, 5].clamp(0.0, 1.0)
        elif obs.size(1) >= 11:
            rel = obs[:, 6:8].unsqueeze(1)
            vel = obs[:, 8:10].unsqueeze(1)
            mask = torch.ones((nBatch, 1), device=device, dtype=dtype)
        else:
            rel = torch.zeros((nBatch, 1, 2), device=device, dtype=dtype)
            vel = torch.zeros((nBatch, 1, 2), device=device, dtype=dtype)
            mask = torch.zeros((nBatch, 1), device=device, dtype=dtype)

        return rel, vel, mask

    def _predict_gmm_multi(self, human_vel):
        """
        human_vel: (B, K, 2)
        Returns:
          means: (B, K, M, 2)
          variances: (B, K, M)
        """
        bsz, k, _ = human_vel.shape
        flat_vel = human_vel.reshape(-1, 2)  # (B*K,2)
        _, means_flat, variances_flat = self.predictor.predict_gmm(flat_vel)
        m = means_flat.shape[1]
        means = means_flat.reshape(bsz, k, m, 2)
        variances = variances_flat.reshape(bsz, k, m)
        return means, variances

    def forward(self, obs):
        # if torch.rand(1).item() < 0.001:
        #      print("DEBUG: BarrierNet v2 Forward")
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(self.fc1.weight.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(obs.size(0), -1)

        x = F.relu(self.fc1(obs))
        x21 = F.relu(self.fc21(x))
        x22 = F.relu(self.fc22(x))
        x23 = F.relu(self.fc23(x))  # reuse fc22 for radius

        u_nom = self.fc31(x21)
        # beta = self.beta * torch.sigmoid(self.fc32(x22)).squeeze(-1)  # (B,) in [0, self.beta]
        beta = 1 * torch.sigmoid(self.fc32(x22)).squeeze(-1)  # (B,) in [0, 1]
        self.last_beta = beta

        # Predict a multiplier between 1.0 and 2.0 (or 0 and 2, but 1.0 is safer baseline)
        # r_scale = 2.0*torch.sigmoid(self.fc33(x23)).squeeze(-1) # (B,) in [0, 2]
        r_scale = 1.0 + 1.5*torch.sigmoid(self.fc33(x23)).squeeze(-1) # (B,) in [0, 2]
        r_safe_learned = self.safe_dist * r_scale
        self.last_r_safe = r_safe_learned

        if self.robot_type == 'single_integrator':
            u_safe = self.dCVaR_CBF_SI(obs, u_nom, beta, r_safe_learned)
        elif self.robot_type == 'unicycle':
            u_safe = self.dCVaR_CBF_Unicycle(obs, u_nom, beta, r_safe_learned)
        elif self.robot_type == 'unicycle_dynamic':
            raise NotImplementedError("dCVaR_CBF_UnicycleDynamic not implemented")
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not supported in CVaR BarrierNet v2")
        return u_safe

    def dCVaR_CBF_SI(self, obs, u_nom, beta, r_safe_learned):
        """
        Solve: min ||u - u_nom||^2
        s.t. for each GMM mode i:
            2*rel^T u >= -alpha*h - cvar

        Using qpth form: G u <= h_qp
        where:
            G = -2*rel^T
            h_qp = -rhs_i
        """
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        # QP: min ||u - u_nom||^2
        Q = torch.eye(self.nCls, device=device).unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        p = -2 *u_nom  
        Q = 2 * Q

        rel, human_vel, mask = self._extract_obstacle_blocks(obs)  # (B,K,2), (B,K,2), (B,K)
        rel_x = rel[:, :, 0]
        rel_y = rel[:, :, 1]

        # R_safe = self.safe_dist
        # Use learned r_safe
        dist_sq = rel_x**2 + rel_y**2
        
        # Expand r_safe_learned for broadcasting against (B,)
        # Note: h(x) = dist^2 - (r_safe_learned)^2
        h = dist_sq - r_safe_learned.unsqueeze(1)**2
        # barrier h(x) = ||rel||^2 - R^2

        cvar_coeff = cvar_coeff_from_beta(beta).unsqueeze(1).unsqueeze(2)  # (B,1,1)
        
        means, variances = self._predict_gmm_multi(human_vel)  # (B,K,M,2), (B,K,M)


        # sigma_f = sqrt(4*sigma^2*||rel||^2)  (isotropic Sigma_v = sigma^2 I)
        rel_norm_sq = (rel ** 2).sum(dim=2, keepdim=True)            # (B,K,1)
        sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq + 1e-8)

        # rhs_i = -alpha*h + 2*rel^T mu_v_i + sigma_f * cvar_coeff
        # 2*rel^T mu: (B,K,M)
        rel_dot_mu = 2.0 * (means * rel.unsqueeze(2)).sum(dim=3)

        rhs = (-self.alpha * h.unsqueeze(2)) + rel_dot_mu + (sigma_f * cvar_coeff)  # (B,K,M)

        # Collapse GMM modes to one robust rhs per obstacle slot.
        tau = 0.1   # small temperature for smooth max
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=2)  # (B,K)

        G = (-2.0 * rel * mask.unsqueeze(-1)).contiguous()     # (B,K,2)
        h_qp = (-(rhs_wc * mask)).contiguous()                 # (B,K)

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=self.fc1.weight.device, dtype=torch.float64)

        if self.training:    
            x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            if nBatch == 1:
                 x = solver(
                     Q_qp[0], p_qp[0], G_qp[0], h_qp_qp[0],
                     device=device,
                     dtype=torch.float32,
                     warm_start_x=self._qp_warm_start,
                 )
                 self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                 x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        return x.to(dtype=obs.dtype)

    def dCVaR_CBF_Unicycle(self, obs, u_nom_xy, beta, r_safe_learned):
        """
        Lookahead CVaR-CBF for Unicycle with learned safe radius.
        """
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        # Obs parsing
        theta = obs[:, 4]
        rel, human_vel, mask = self._extract_obstacle_blocks(obs)  # (B,K,2), (B,K,2), (B,K)

        epsilon = 0.2
        R_safe = r_safe_learned.unsqueeze(1) + epsilon

        c = torch.cos(theta)
        s = torch.sin(theta)

        # J maps unicycle control u=[v, w] to lookahead Cartesian velocity v_L=[vx, vy].
        # J = [[cos(theta), -eps*sin(theta)],
        #      [sin(theta),  eps*cos(theta)]]
        J = torch.zeros(nBatch, 2, 2, device=device, dtype=obs.dtype)
        J[:, 0, 0] = c
        J[:, 0, 1] = -epsilon * s
        J[:, 1, 0] = s
        J[:, 1, 1] = epsilon * c

        # Cost (match controller/cbf_qp.py):
        # min ||J u - u_nom_xy||^2
        JT = J.transpose(1, 2)
        Q = 2.0 * torch.bmm(JT, J)
        Q = Q + 1e-6 * torch.eye(self.nCls, device=device, dtype=obs.dtype).unsqueeze(0)
        p = -2.0 * torch.bmm(JT, u_nom_xy.unsqueeze(-1)).squeeze(-1)


        # # QP: min ||u - u_nom||^2
        # Q = torch.eye(self.nCls, device=device).unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        # p = -2 * u_nom
        # Q = 2 * Q + 1e-6 * torch.eye(self.nCls, device=device)


        # Lookahead relative position
        heading = torch.stack([c, s], dim=1).unsqueeze(1)  # (B,1,2)
        p_L = rel + epsilon * heading  # (B,K,2)

        h = (p_L[:, :, 0] ** 2 + p_L[:, :, 1] ** 2) - R_safe ** 2

        # Lg terms for u = [v, w]
        lg_v = 2 * (p_L[:, :, 0] * c.unsqueeze(1) + p_L[:, :, 1] * s.unsqueeze(1))
        lg_w = 2 * epsilon * (p_L[:, :, 1] * c.unsqueeze(1) - p_L[:, :, 0] * s.unsqueeze(1))

        # CVaR term from GMM prediction of human velocity
        cvar_coeff = cvar_coeff_from_beta(beta).unsqueeze(1).unsqueeze(2)  # (B,1,1)
        means, variances = self._predict_gmm_multi(human_vel)  # (B,K,M,2), (B,K,M)

        pL_norm_sq = (p_L ** 2).sum(dim=2, keepdim=True)  # (B,K,1)
        sigma_f = torch.sqrt(4.0 * variances * pL_norm_sq + 1e-8)  # (B,K,M)

        rel_dot_mu = 2.0 * (means * p_L.unsqueeze(2)).sum(dim=3)  # (B,K,M)

        rhs = (-self.alpha * h.unsqueeze(2)) + rel_dot_mu + (sigma_f * cvar_coeff)  # (B,K,M)

        tau = 0.1
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=2)  # (B,K)

        G = torch.stack([-(lg_v * mask), -(lg_w * mask)], dim=2).contiguous()  # (B,K,2)
        h_qp = (-(rhs_wc * mask)).contiguous()  # (B,K)

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=self.fc1.weight.device, dtype=torch.float64)

        if self.training:    
            x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            if nBatch == 1:
                 x = solver(
                     Q_qp[0], p_qp[0], G_qp[0], h_qp_qp[0],
                     device=device,
                     dtype=torch.float32,
                     warm_start_x=self._qp_warm_start,
                 )
                 self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                 x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        return x.to(dtype=obs.dtype)
        
