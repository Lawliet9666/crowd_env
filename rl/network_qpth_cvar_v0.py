import torch.nn as nn
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction
import numpy as np
from rl.my_classes import test_solver as solver
from controller.traj_prediction_batch import TrajPredictorTorch as TrajPredictor
from scipy.stats import norm

class BarrierNet(nn.Module):
    def __init__(self, nFeatures, nCls, nHidden1=128, nHidden2=32,
                 safe_dist=0.8, alpha=2.0, beta=0.1,
                 gmm_weights=None, gmm_stds=None, gmm_lateral_ratio=0.3):
        super().__init__()
        self.nFeatures = nFeatures
        self.nCls = nCls
        self.safe_dist = safe_dist
        self.alpha = alpha   
        self.last_alpha = alpha
        self._qp_warm_start = None

        self.predictor = TrajPredictor(
            lateral_ratio=gmm_lateral_ratio,
            weights=gmm_weights,
            stds=gmm_stds,
        )

        self.fc1 = nn.Linear(nFeatures, nHidden1)
        self.fc2 = nn.Linear(nHidden1, nHidden2)
        self.fc_out = nn.Linear(nHidden2, nCls)   

        inv_cdf = norm.ppf(1 - beta)
        pdf_val = norm.pdf(inv_cdf)
        self.cvar_coeff = pdf_val / beta

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(self.fc1.weight.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(obs.size(0), -1)

        x = F.silu(self.fc1(obs))
        x = F.silu(self.fc2(x))
        u_nom = self.fc_out(x)

        u_safe = self.dCVaR_CBF_SI(obs, u_nom)
        return u_safe

    def dCVaR_CBF_SI(self, obs, u_nom):
        """
        Solve: min ||u - u_nom||^2
        s.t. for each GMM mode i:
            2*rel^T u >= -alpha*h + 2*rel^T mu_v_i + sigma_f_i * cvar_coeff

        Using qpth form: G u <= h_qp
        where:
            G = -2*rel^T
            h_qp = -rhs_i
        """
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        # QP: min ||u - u_nom||^2
        Q = torch.eye(self.nCls, device=device).unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        p = -2 *u_nom  # 关键：让解接近 u_nom
        Q = 2 * Q

        rel_x = obs[:, 6]
        rel_y = obs[:, 7]
        human_vel = obs[:, 8:10]  # (B,2)
        rel = obs[:, 6:8]  # (B,2)

        R_safe = self.safe_dist
        dist_sq = rel_x**2 + rel_y**2
        h = dist_sq - R_safe**2
        # barrier h(x) = ||rel||^2 - R^2

        
        _ , means, variances = self.predictor.predict_gmm(human_vel)
        M = means.size(1)


        # sigma_f = sqrt(4*sigma^2*||rel||^2)  (isotropic Sigma_v = sigma^2 I)
        rel_norm_sq = (rel ** 2).sum(dim=1, keepdim=True)            # (B,1)
        # sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq)          # (B,M)
        sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq + 1e-8)

        # rhs_i = -alpha*h + 2*rel^T mu_v_i + sigma_f * cvar_coeff
        # 2*rel^T mu: (B,M)
        rel_dot_mu = 2.0 * (means * rel.unsqueeze(1)).sum(dim=2)     # (B,M)

        rhs = (-self.alpha * h.unsqueeze(1)) + rel_dot_mu + (sigma_f * self.cvar_coeff)  # (B,M)
        # # qpth expects G u <= h_qp
        # h_qp = -rhs  # (B,M)
        # # G: (B,M,2) each row is [-2*rel_x, -2*rel_y]
        # G = (-2.0 * rel).unsqueeze(1).expand(-1, M, -1).contiguous()  # (B,M,2)

        # ✅ collapse to worst-case single constraint (since G rows are identical anyway)
        rhs_wc = rhs.max(dim=1).values                 # (B,)
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

        
