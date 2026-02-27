import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from qpth.qp import QPFunction
import numpy as np
from rl.my_classes import test_solver as solver
# from controller.traj_prediction_batch import TrajPredictorTorch as TrajPredictor

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
                 slack_weight=10.0):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        
        self.safe_dist = safe_dist
        self.robot_type = robot_type
        self.slack_weight = slack_weight
        self.alpha = alpha

        self.last_alpha = alpha

        if self.robot_type == 'single_integrator':
            self.u_min = [-vmax, -vmax]
            self.u_max = [vmax, vmax]
            self._cbf_name = "dCBF_SI"
        elif self.robot_type == 'unicycle':
            self.u_min = [-vmax, -omega_max]
            self.u_max = [vmax, omega_max]
            self._cbf_name = "dCBF_Unicycle"
        elif self.robot_type == 'unicycle_dynamic':
            self.u_min = [-omega_max, -amax]
            self.u_max = [omega_max, amax]
            self._cbf_name = "dCBF_UnicycleDynamic"
        else:
            self.u_min = None
            self.u_max = None
            self._cbf_name = "unknown"

        print(
            f"[CBFNet] robot_type={self.robot_type}, safe_dist={self.safe_dist:.3f}, umax={self.u_max}",
            flush=True,
        )
        self._qp_warm_start = None
        # self.predictor = TrajPredictor()

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
        
        x31 = self.fc31(x21) # unominal
        unom = x31

        # x32 = self.fc32(x22) # alpha
        # x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        alpha = 4*torch.sigmoid(self.fc32(x22)).squeeze(-1)  # (B,)
        self.last_alpha = alpha

        # BarrierNet dispatch based on robot type
        if self.robot_type == 'single_integrator':
            x = self.dCBF_SI(obs, unom, alpha)
        elif self.robot_type == 'unicycle':
            x = self.dCBF_Unicycle(obs, unom, alpha)
        elif self.robot_type == 'unicycle_dynamic':
            x = self.dCBF_UnicycleDynamic(obs, unom, alpha)
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not supported in BarrierNet")
               
        return x

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
            # Fallback: disabled dummy obstacle so constraints become 0 <= 0
            rel = torch.zeros((nBatch, 1, 2), device=device, dtype=dtype)
            vel = torch.zeros((nBatch, 1, 2), device=device, dtype=dtype)
            mask = torch.zeros((nBatch, 1), device=device, dtype=dtype)

        return rel, vel, mask
    
    def dCBF_SI(self, obs, unom, alpha):
        nBatch = obs.size(0)
        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.fc1.weight.device) 
        
        p = -2 * unom  # Minimize ||u - u_nom||^2
        Q = Q * 2  # since 1/2 factor in QPFunction

        rel, vel, mask = self._extract_obstacle_blocks(obs)  # (B,K,2), (B,K,2), (B,K)
        rel_x = rel[:, :, 0]
        rel_y = rel[:, :, 1]
        v_hx = vel[:, :, 0]
        v_hy = vel[:, :, 1]

        # current_vel = obs[:, 8:10].cpu().numpy()  # (B,2)
        # human_vel_pred = self.predictor.predict_vel_expectation(current_vel)
        # v_hx = human_vel_pred[:, 0]
        # v_hy = human_vel_pred[:, 1]
        
        # Safety distance (Robot Radius + Human Radius)
        R_safe = self.safe_dist
        # R_safe = r_radius + h_radius + 0.1 # 0.1 buffer

        # Barrier function h(x) = (px - hx)^2 + (py - hy)^2 - R_safe^2
        # rel_x = px - hx, rel_y = py - hy
        dist_sq = rel_x**2 + rel_y**2
        barrier = dist_sq - R_safe**2

        # Gradient of h w.r.t robot position p = [px, py]
        # dh/dpx = 2*(px - hx) = 2*rel_x
        # dh/dpy = 2*(py - hy) = 2*rel_y
        lg1 = 2 * rel_x
        lg2 = 2 * rel_y
        
        # Lie derivative w.r.t to Human Dynamics (Lf)
        # Human is moving, so h changes even if robot is static.
        # \dot{h}_{drift} = (dh/d_hx * \dot{h}_x) + (dh/d_hy * \dot{h}_y)
        # dh/d_hx = -2*(px - hx) = -2*rel_x
        # dh/d_hy = -2*(py - hy) = -2*rel_y
        lf = -2 * rel_x * v_hx - 2 * rel_y * v_hy

        # Construct K CBF constraints (one per obstacle slot):
        # -Lg*u <= Lf + alpha*h
        # mask=0 (dummy obstacle) -> 0 <= 0
        G = torch.stack(
            [-(lg1 * mask), -(lg2 * mask)],
            dim=2,
        ).to(self.fc1.weight.device)  # (B,K,2)
        h_qp = ((lf + alpha.unsqueeze(1) * barrier) * mask).to(self.fc1.weight.device)  # (B,K)
        
        # Solve QP in float64 for better numerical stability, then cast action back.
        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=self.fc1.weight.device, dtype=torch.float64)

        if self.training:    
            x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            # If batching is supported by solver, use it, otherwise fallback to QPFunction or single item
            if nBatch == 1:
                 # x31[0] is 1D tensor of size 2
                 self.p1 = alpha[0]
                 x = solver(
                     Q_qp[0], p_qp[0], G_qp[0], h_qp_qp[0],
                     device=self.fc1.weight.device,
                     dtype=torch.float32,
                     warm_start_x=self._qp_warm_start,
                 )
                 self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                 x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        
        return x.to(dtype=obs.dtype)

    def dCBF_SI_slack(self, obs, unom, alpha):
        nBatch = obs.size(0)
        device = self.fc1.weight.device
        n_actions = unom.size(1)
        n_vars = n_actions + 1

        # QP: min ||u - u_nom||^2 + slack_weight * s^2
        Q = torch.eye(n_vars, device=device).unsqueeze(0).expand(nBatch, n_vars, n_vars)
        Q[:, -1, -1] = self.slack_weight
        p = torch.cat([-2 * unom, torch.zeros(nBatch, 1, device=device)], dim=1)
        Q = 2 * Q

        rel_x = obs[:, 6]
        rel_y = obs[:, 7]
        v_hx = obs[:, 8]
        v_hy = obs[:, 9]

        R_safe = self.safe_dist
        dist_sq = rel_x**2 + rel_y**2
        barrier = dist_sq - R_safe**2

        lg1 = 2 * rel_x
        lg2 = 2 * rel_y
        lf = -2 * rel_x * v_hx - 2 * rel_y * v_hy

        # -Lg*u - s <= Lf + alpha*h
        G_cbf = torch.stack([-lg1, -lg2, -torch.ones_like(lg1)], dim=1).to(device).unsqueeze(1)
        h_qp = (lf + alpha * barrier).unsqueeze(1).to(device)

        # Enforce s >= 0  ->  -s <= 0
        G_slack = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).unsqueeze(0).expand(nBatch, 1, 3)
        h_slack = torch.zeros(nBatch, 1, device=device)

        G = torch.cat([G_cbf, G_slack], dim=1)
        h_qp = torch.cat([h_qp, h_slack], dim=1)

        # Input box constraints
        G, h_qp = self._append_input_constraints(G, h_qp, device, obs.dtype)

        Q_qp = Q.to(dtype=torch.float64)
        p_qp = p.to(dtype=torch.float64)
        G_qp = G.to(dtype=torch.float64)
        h_qp_qp = h_qp.to(dtype=torch.float64)
        e_qp = torch.empty(0, device=device, dtype=torch.float64)

        if self.training:
            x = QPFunction(verbose=0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        else:
            if nBatch == 1:
                self.p1 = alpha[0]
                x = solver(
                    Q_qp[0], p_qp[0], G_qp[0], h_qp_qp[0],
                    device=device,
                    dtype=torch.float32,
                    warm_start_x=self._qp_warm_start,
                )
                self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                x = QPFunction(verbose=0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)

        return x.to(dtype=obs.dtype)[:, :n_actions]

    def dCBF_Unicycle(self, obs, u_nom_xy, alpha):
        # Lookahead CBF for Unicycle with learnable alpha
        nBatch = obs.size(0)
        device = self.fc1.weight.device

        # Obs parsing
        theta = obs[:, 4]
        rel, vel, mask = self._extract_obstacle_blocks(obs)  # (B,K,2), (B,K,2), (B,K)
        v_hx = vel[:, :, 0]
        v_hy = vel[:, :, 1]

        # current_vel = obs[:, 8:10].cpu().numpy()  # (B,2)
        # human_vel_pred = self.predictor.predict_vel_expectation(current_vel)
        # v_hx = human_vel_pred[:, 0]
        # v_hy = human_vel_pred[:, 1]

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

        # Lookahead relative position p_L_rel = p_rel + eps*[cos(theta), sin(theta)]
        heading = torch.stack([c, s], dim=1).unsqueeze(1)  # (B,1,2)
        p_L = rel + epsilon * heading  # (B,K,2)
        h = (p_L[:, :, 0] ** 2 + p_L[:, :, 1] ** 2) - R_safe ** 2

        # CBF: 2*p_L^T*J*u >= -alpha*h + 2*p_L^T*v_h
        # Lg = 2.0 * torch.einsum("bki,bij->bkj", p_L, J)  # (B,K,2)
        Lg = 2.0 * torch.matmul(p_L, J)
        rhs = -alpha.unsqueeze(1) * h + 2.0 * (p_L[:, :, 0] * v_hx + p_L[:, :, 1] * v_hy)  # (B,K)

        # qpth uses G u <= h_qp
        G = (-(Lg * mask.unsqueeze(-1))).to(device)  # (B,K,2)
        h_qp = (-(rhs * mask)).to(device)  # (B,K)

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

    def dCBF_UnicycleDynamic(self, obs, u_nom, alpha):
        # High Order CBF for Unicycle Dynamic with learnable alpha
        nBatch = obs.size(0)
        device = self.fc1.weight.device
        
        Q = Variable(torch.eye(self.nCls)).to(device)
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        p = -2 * u_nom
        Q = 2 * Q
        
        vx = obs[:, 2]
        vy = obs[:, 3]
        theta = obs[:, 4]
        rel, vel, mask = self._extract_obstacle_blocks(obs)  # (B,K,2), (B,K,2), (B,K)
        rel_x = rel[:, :, 0]
        rel_y = rel[:, :, 1]
        v_hx = vel[:, :, 0]
        v_hy = vel[:, :, 1]

        # current_vel = obs[:, 8:10].cpu().numpy()  # (B,2)
        # human_vel_pred = self.predictor.predict_vel_expectation(current_vel)
        # v_hx = human_vel_pred[:, 0]
        # v_hy = human_vel_pred[:, 1]
        
        R_safe = self.safe_dist
        
        # HOCBF terms
        dist_sq = rel_x**2 + rel_y**2
        h = dist_sq - R_safe**2
        
        v_rel_x = vx - v_hx
        v_rel_y = vy - v_hy
        v_rel_sq = v_rel_x**2 + v_rel_y**2
        
        h_dot = 2 * (rel_x * v_rel_x + rel_y * v_rel_y)
        
        c = torch.cos(theta)
        s = torch.sin(theta)
        v_signed = vx * c + vy * s
        
        # J_dyn0 = [-v*s, v*c]
        j0_x = -v_signed * s
        j0_y = v_signed * c
        
        # J_dyn1 = [c, s]
        j1_x = c
        j1_y = s
        
        # Lg Lf h
        lg0 = 2 * (rel_x * j0_x.unsqueeze(1) + rel_y * j0_y.unsqueeze(1))
        lg1 = 2 * (rel_x * j1_x.unsqueeze(1) + rel_y * j1_y.unsqueeze(1))
        
        # Constraint: Lg*u >= - (2*||v_rel||^2 + K*h_dot + gamma^2*h)
        # We assume gamma1 = gamma2 = alpha (learnable)
        # K_gain = 2 * alpha
        # gamma^2 = alpha^2
        
        K_gain = 2 * alpha
        
        rhs_term = 2 * v_rel_sq + K_gain.unsqueeze(1) * h_dot + (alpha * alpha).unsqueeze(1) * h

        G = torch.stack([-(lg0 * mask), -(lg1 * mask)], dim=2).to(device)  # (B,K,2)
        h_qp = (rhs_term * mask).to(device)  # (B,K)
        
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

#     def __init__(self, 
#                  nFeatures = 5, 
#                  nHidden1 = 128, 
#                  nHidden21 = 32, 
#                  nHidden22 = 32, 
#                  nCls = 2, 
#                  device=None):
#         super().__init__()
#         # nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2 

#         self.nFeatures = nFeatures
#         self.nHidden1 = nHidden1
#         self.nHidden21 = nHidden21
#         self.nHidden22 = nHidden22
#         self.nCls = nCls
#         self.device = device

#         self.fc1 = nn.Linear(nFeatures, nHidden1) 
#         self.fc21 = nn.Linear(nHidden1, nHidden21) 
#         self.fc22 = nn.Linear(nHidden1, nHidden22) 
#         self.fc31 = nn.Linear(nHidden21, nCls) 
#         self.fc32 = nn.Linear(nHidden22, nCls) 

#         # QP params.
#         # from previous layers

#     def forward(self, x, sgn):
#         nBatch = x.size(0)

#         # Normal FC network.
#         x = x.view(nBatch, -1)
#         x0 = x
#         x = F.relu(self.fc1(x))
        
#         x21 = F.relu(self.fc21(x))
#         x22 = F.relu(self.fc22(x))

#         x31 = self.fc31(x21)
#         x32 = self.fc32(x22)
#         x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        
#         # BarrierNet
#         x = self.dCBF(x0, x31, x32, sgn, nBatch)
               
#         return x

#     def dCBF(self, x0, x31, x32, sgn, nBatch):

#         Q = Variable(torch.eye(self.nCls))
#         Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
#         px = x0[:,0]
#         py = x0[:,1]
#         theta = x0[:,2]
#         v = x0[:,3]
#         sin_theta = torch.sin(theta)
#         cos_theta = torch.cos(theta)
        
#         barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2

#         if wandb.run is not None:
#             wandb.log({
#                 "barrier_min": torch.min(barrier).item(),
#                 "barrier_avg": torch.mean(barrier).item()
#             }, commit=False)

#         barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
#         Lf2b = 2*v**2
#         LgLfbu1 = torch.reshape(-2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta, (nBatch, 1)) 
#         LgLfbu2 = torch.reshape(2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta, (nBatch, 1))
          
#         G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
#         G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)     
#         h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))).to(self.device) 
#         e = Variable(torch.Tensor()).to(self.device)
            
#         if self.training or sgn == 1:    
#             x = QPFunction(verbose = 0, maxIter=40)(Q , x31 , G , h , e, e)
#         else:
#             self.p1 = x32[0,0]
#             self.p2 = x32[0,1]
#             x = solver(Q[0] , x31[0] , G[0] , h[0] )
        
#         return x
