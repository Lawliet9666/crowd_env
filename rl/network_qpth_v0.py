import torch.nn as nn
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction
import numpy as np
from rl.my_classes import test_solver as solver
from rl.network import ResidualMLPBlock 



class BarrierNet(nn.Module):
    def __init__(self, nFeatures, nCls, nHidden1=128, nHidden2=32,
                 safe_dist=0.8, alpha=2.0, beta=0.2, 
                 robot_type='single_integrator',
                 vmax = 3.0, amax = 3.0, omega_max = 3.0,
                 slack_weight=10.0):
        super().__init__()
        self.nFeatures = nFeatures
        self.nCls = nCls
        self.robot_type = robot_type
        self.safe_dist = safe_dist
        self.alpha = alpha   
        self.last_alpha = alpha
        self.slack_weight = slack_weight

        if self.robot_type == 'single_integrator':
            self.u_min = [-vmax, -vmax]
            self.u_max = [vmax, vmax]
        elif self.robot_type == 'unicycle':
            self.u_min = [-vmax, -omega_max]
            self.u_max = [vmax, omega_max]
        elif self.robot_type == 'unicycle_dynamic':
            self.u_min = [-omega_max, -amax]
            self.u_max = [omega_max, amax]
        else:
            self.u_min = None
            self.u_max = None

        self._qp_warm_start = None
        self.fc_in = nn.Linear(nFeatures, nHidden1)
        self.res1 = ResidualMLPBlock(nHidden1, nHidden2)
        self.res2 = ResidualMLPBlock(nHidden1, nHidden2)
        self.fc_out = nn.Linear(nHidden1, nCls)  # 只输出 u_nom

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(self.fc_in.weight.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(obs.size(0), -1)

        x = F.relu(self.fc_in(obs))
        x = self.res1(x)
        x = self.res2(x)
        u_nom = self.fc_out(x)
    
        if self.robot_type == 'single_integrator':
            u_safe = self.dCBF_SI(obs, u_nom)
        elif self.robot_type == 'unicycle':
            u_safe = self.dCBF_Unicycle(obs, u_nom)
        elif self.robot_type == 'unicycle_dynamic':
            # unicycle_dynamic path intentionally disabled.
            u_safe = torch.zeros_like(u_nom)
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not supported in BarrierNet")
            
        return u_safe

    def _get_bound_tensor(self, bound, device, dtype):
        if bound is None:
            return None
        if torch.is_tensor(bound):
            return bound.to(device=device, dtype=dtype)
        if np.isscalar(bound):
            values = [float(bound)] * self.nCls
        else:
            values = list(bound)
        return torch.tensor(values, device=device, dtype=dtype)

    def _append_input_constraints(self, G, h, device, dtype):
        u_min = self._get_bound_tensor(self.u_min, device, dtype)
        u_max = self._get_bound_tensor(self.u_max, device, dtype)

        nBatch = G.size(0)
        n = self.nCls
        eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(nBatch, n, n)
        zero_col = torch.zeros(nBatch, n, 1, device=device, dtype=dtype)

        G_list = []
        h_list = []

        G_max = torch.cat([eye, zero_col], dim=2)
        h_max = u_max.unsqueeze(0).expand(nBatch, n)
        G_list.append(G_max)
        h_list.append(h_max)

        G_min = torch.cat([-eye, zero_col], dim=2)
        h_min = (-u_min).unsqueeze(0).expand(nBatch, n)
        G_list.append(G_min)
        h_list.append(h_min)

        G_box = torch.cat(G_list, dim=1)
        h_box = torch.cat(h_list, dim=1)

        G = torch.cat([G, G_box], dim=1)
        h = torch.cat([h, h_box], dim=1)
        return G, h

    def dCBF_SI(self, obs, u_nom): # slack variable
        nBatch = obs.size(0)
        device = self.fc_in.weight.device

        # QP: min ||u - u_nom||^2
        Q = torch.eye(self.nCls + 1, device=device).unsqueeze(0).expand(nBatch, self.nCls + 1, self.nCls + 1)
        Q[:, -1, -1] = self.slack_weight
        p = torch.cat([-2 * u_nom, torch.zeros(nBatch, 1, device=device)], dim=1)
        Q = 2 * Q

        rel_x = obs[:, 6]
        rel_y = obs[:, 7]
        v_hx = obs[:, 8]
        v_hy = obs[:, 9]

        R_safe = self.safe_dist
        dist_sq = rel_x**2 + rel_y**2
        h = dist_sq - R_safe**2

        lg1 = 2 * rel_x
        lg2 = 2 * rel_y
        lf = -2 * rel_x * v_hx - 2 * rel_y * v_hy

        G_cbf = torch.stack([-lg1, -lg2, -torch.ones_like(lg1)], dim=1).to(device).unsqueeze(1)  # (B,1,3)
        G_slack = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).unsqueeze(0).expand(nBatch, 1, 3)

        alpha = torch.full((nBatch,), float(self.alpha), device=device)
        h_qp = (lf + alpha * h).unsqueeze(1).to(device)  # (B,1)
        # G_cbf, h_qp = self._soft_gate_cbf(G_cbf, h_qp, h, device, obs.dtype)

        h_slack = torch.zeros(nBatch, 1, device=device)
        G = torch.cat([G_cbf, G_slack], dim=1)
        h_qp = torch.cat([h_qp, h_slack], dim=1)
        G, h_qp = self._append_input_constraints(G, h_qp, device, obs.dtype)

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=device, dtype=torch.float64)

        if self.training:    
            x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            # If batching is supported by solver, use it, otherwise fallback to QPFunction or single item
            if nBatch == 1:
                 # x31[0] is 1D tensor of size 2
                 self.p1 = alpha[0]
                 x = solver(
                     Q_qp[0], p_qp[0], G_qp[0], h_qp_qp[0],
                     device=device,
                     dtype=torch.float32,
                     warm_start_x=self._qp_warm_start,
                 )
                 self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                 x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        
        return x.to(dtype=obs.dtype)[:, :self.nCls]


    # def dCBF_SI(self, obs, u_nom): # no input constraints
    #     nBatch = obs.size(0)
    #     device = self.fc_in.weight.device

    #     # QP: min ||u - u_nom||^2
    #     Q = torch.eye(self.nCls, device=device).unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
    #     p = -2 * u_nom
    #     Q = 2 * Q

    #     rel_x = obs[:, 6]
    #     rel_y = obs[:, 7]
    #     v_hx = obs[:, 8]
    #     v_hy = obs[:, 9]

    #     R_safe = self.safe_dist
    #     dist_sq = rel_x**2 + rel_y**2
    #     h = dist_sq - R_safe**2

    #     lg1 = 2 * rel_x
    #     lg2 = 2 * rel_y
    #     lf = -2 * rel_x * v_hx - 2 * rel_y * v_hy

    #     G = torch.stack([-lg1, -lg2], dim=1).to(device).unsqueeze(1)  # (B,1,2)

    #     alpha = torch.full((nBatch,), float(self.alpha), device=device)
    #     h_qp = (lf + alpha * h).unsqueeze(1).to(device)  # (B,1)

    #     e = torch.empty(0, device=device, dtype=obs.dtype)

    #     if self.training:    
    #         x = QPFunction(verbose = 0, maxIter=40)(Q, p , G, h_qp, e, e)
    #     else:
    #         # If batching is supported by solver, use it, otherwise fallback to QPFunction or single item
    #         if nBatch == 1:
    #              # x31[0] is 1D tensor of size 2
    #              self.p1 = alpha[0]
    #              x = solver(Q[0], p[0] , G[0], h_qp[0])
    #         else:
    #              x = QPFunction(verbose = 0, maxIter=40)(Q, p , G, h_qp, e, e)
        
    #     return x

    def dCBF_Unicycle(self, obs, u_nom):
        # Lookahead CBF for Unicycle
        nBatch = obs.size(0)
        device = self.fc_in.weight.device
        
        # QP: min ||u - u_nom||^2 (assuming u_nom matches action space [v, w])
        Q = torch.eye(self.nCls, device=device).unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        p = -2 * u_nom
        Q = 2 * Q
        
        # Obs parsing
        theta = obs[:, 4]
        rel_x = obs[:, 6]
        rel_y = obs[:, 7]
        v_hx = obs[:, 8]
        v_hy = obs[:, 9]
        
        epsilon = 0.2 # 0.5?
        R_safe = self.safe_dist + epsilon
        
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        # Lookahead pos relative
        p_L_x = rel_x + epsilon * c
        p_L_y = rel_y + epsilon * s
        
        h = (p_L_x**2 + p_L_y**2) - R_safe**2
        
        # h_dot = 2 * p_L^T * (J u - v_h)
        # J = [[c, -e*s], [s, e*c]]
        # J col 1 (coeff of v): [c, s]
        # J col 2 (coeff of w): [-e*s, e*c]
        
        # Coeffs for u = [v, w]
        # 2 * p_L^T * col1 = 2*(p_L_x*c + p_L_y*s)
        lg_v = 2 * (p_L_x * c + p_L_y * s)
        # 2 * p_L^T * col2 = 2*(p_L_x*(-e*s) + p_L_y*(e*c))
        lg_w = 2 * epsilon * (p_L_y * c - p_L_x * s)
        
        # Lf h = -2 * p_L^T * v_h
        lf = -2 * (p_L_x * v_hx + p_L_y * v_hy)
        
        # Constraint: Lg u >= -Lf - alpha*h
        # -Lg u <= Lf + alpha*h
        # G = [-lg_v, -lg_w]
        
        G = torch.stack([-lg_v, -lg_w], dim=1).to(device).unsqueeze(1) # (B, 1, 2)
        
        alpha = torch.full((nBatch,), float(self.alpha), device=device)
        h_qp = (lf + alpha * h).unsqueeze(1).to(device)
        G, h_qp = self._soft_gate_cbf(G, h_qp, h, device, obs.dtype)
        G, h_qp = self._append_input_constraints(G, h_qp, device, obs.dtype)
        
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

    def dCBF_UnicycleDynamic(self, obs, u_nom):
        # unicycle_dynamic path intentionally disabled.
        return torch.zeros_like(u_nom)


        

        
