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
    def __init__(self, 
                 nFeatures, 
                 nCls,
                 nHidden1 = 256, 
                 nHidden21 = 256, 
                 nHidden22 = 256, 
                 safe_dist = 0.8,
                 alpha = 2.0,
                 beta = 0.2,
                 robot_type='single_integrator',
                 vmax = 3.0, amax = 3.0, omega_max = 3.0,
                 gmm_weights=None, gmm_stds=None, gmm_lateral_ratio=0.3
            ):
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
            f"[CVaRNet] robot_type={self.robot_type}, safe_dist={self.safe_dist:.3f}, umax={self.u_max}",
            flush=True,
        )
        
        self.predictor = TrajPredictor(
            lateral_ratio=gmm_lateral_ratio,
            weights=gmm_weights,
            stds=gmm_stds,
        )

        self.fc1 = nn.Linear(nFeatures, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21) 
        self.fc22 = nn.Linear(nHidden1, nHidden22) 
        self.fc31 = nn.Linear(nHidden21, nCls) 
        self.fc32 = nn.Linear(nHidden22, 1) 

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(self.fc1.weight.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(obs.size(0), -1)

        x = F.relu(self.fc1(obs))
        x21 = F.relu(self.fc21(x))
        x22 = F.relu(self.fc22(x))

        u_nom = self.fc31(x21)
        beta = self.beta *torch.sigmoid(self.fc32(x22)).squeeze(-1)  # (B,)
        self.last_beta = beta

        if self.robot_type == 'single_integrator':
            u_safe = self.dCVaR_CBF_SI(obs, u_nom, beta)
        elif self.robot_type == 'unicycle':
            u_safe = self.dCVaR_CBF_Unicycle(obs, u_nom, beta)
        elif self.robot_type == 'unicycle_dynamic':
            raise NotImplementedError("dCVaR_CBF_UnicycleDynamic not implemented")
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not supported in CVaR BarrierNet")
        return u_safe

    def dCVaR_CBF_SI(self, obs, u_nom, beta):
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

        rel_x = obs[:, 6]
        rel_y = obs[:, 7]
        human_vel = obs[:, 8:10]  # (B,2)
        rel = obs[:, 6:8]  # (B,2)

        R_safe = self.safe_dist
        dist_sq = rel_x**2 + rel_y**2
        h = dist_sq - R_safe**2
        # barrier h(x) = ||rel||^2 - R^2

        cvar_coeff = cvar_coeff_from_beta(beta)  # (B,)
        
        _ , means, variances = self.predictor.predict_gmm(human_vel)
        M = means.size(1)


        # sigma_f = sqrt(4*sigma^2*||rel||^2)  (isotropic Sigma_v = sigma^2 I)
        rel_norm_sq = (rel ** 2).sum(dim=1, keepdim=True)            # (B,1)
        # sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq)          # (B,M)
        sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq + 1e-8)

        # rhs_i = -alpha*h + 2*rel^T mu_v_i + sigma_f * cvar_coeff
        # 2*rel^T mu: (B,M)
        rel_dot_mu = 2.0 * (means * rel.unsqueeze(1)).sum(dim=2)     # (B,M)

        # Fix broadcasting: cvar_coeff is (B,), sigma_f is (B,M)
        # We need cvar_coeff to broadcast over M
        cvar_coeff = cvar_coeff.unsqueeze(1) # (B, 1)

        rhs = (-self.alpha * h.unsqueeze(1)) + rel_dot_mu + (sigma_f * cvar_coeff)  # (B,M)
        # # qpth expects G u <= h_qp
        # h_qp = -rhs  # (B,M)
        # # G: (B,M,2) each row is [-2*rel_x, -2*rel_y]
        # G = (-2.0 * rel).unsqueeze(1).expand(-1, M, -1).contiguous()  # (B,M,2)

        # collapse to worst-case single constraint (since G rows are identical anyway)
        # rhs_wc = rhs.max(dim=1).values                 # (B,)
        tau = 0.1   # small temperature for smooth max
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=1)

        G = (-2.0 * rel).unsqueeze(1).contiguous()     # (B,1,2)
        h_qp = (-rhs_wc).unsqueeze(1).contiguous()     # (B,1)

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

    def dCVaR_CBF_Unicycle(self, obs, u_nom_xy, beta):
        """
        Lookahead CVaR-CBF for Unicycle.
        Constraint: Lg(u) >= -Lf - alpha*h + CVaR term
        """
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        # Obs parsing
        theta = obs[:, 4]
        rel = obs[:, 6:8]
        human_vel = obs[:, 8:10]  # (B,2)

        epsilon = 0.2
        R_safe = self.safe_dist + epsilon

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

        # Lookahead relative position
        p_L = rel + epsilon * torch.stack([c, s], dim=1)  # (B,2)

        h = (p_L[:, 0] ** 2 + p_L[:, 1] ** 2) - R_safe ** 2

        # Lg terms for u = [v, w]
        lg_v = 2 * (p_L[:, 0] * c + p_L[:, 1] * s)
        lg_w = 2 * epsilon * (p_L[:, 1] * c - p_L[:, 0] * s)

        # CVaR term from GMM prediction of human velocity
        cvar_coeff = cvar_coeff_from_beta(beta)  # (B,)
        _, means, variances = self.predictor.predict_gmm(human_vel)

        pL_norm_sq = (p_L ** 2).sum(dim=1, keepdim=True)  # (B,1)
        sigma_f = torch.sqrt(4.0 * variances * pL_norm_sq + 1e-8)  # (B,M)

        rel_dot_mu = 2.0 * (means * p_L.unsqueeze(1)).sum(dim=2)  # (B,M)
        cvar_coeff = cvar_coeff.unsqueeze(1)  # (B,1)

        rhs = (-self.alpha * h.unsqueeze(1)) + rel_dot_mu + (sigma_f * cvar_coeff)  # (B,M)

        tau = 0.1
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=1)  # (B,)

        G = torch.stack([-lg_v, -lg_w], dim=1).unsqueeze(1).contiguous()  # (B,1,2)
        h_qp = (-rhs_wc).unsqueeze(1).contiguous()  # (B,1)

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=device, dtype=torch.float64)

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



        
