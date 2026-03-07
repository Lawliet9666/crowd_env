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
                 obs_dim,
                 act_dim,
                 qp_obs_dim,
                 qp_start_timesteps=0,
                 nHidden1 = 256, 
                 nHidden21 = 256, 
                 nHidden22 = 256, 
                 safe_dist = 0.8,
                 alpha = 2.0,
                 beta = 0.2,
                 robot_type='single_integrator',
                 vmax = 3.0, amax = 3.0, omega_max = 3.0,
                 slack_weight=10.0,
                 **kwargs):
        super().__init__()
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = int(act_dim)
        
        self.safe_dist = safe_dist
        self.robot_type = robot_type
        self.slack_weight = slack_weight
        self.alpha = alpha
        self.alpha_max = float(kwargs.get("alpha_max", 4.0))
        self.outputs_real_action = True
        self.actor_obs_dim = int(obs_dim)
        self.qp_obs_dim = int(qp_obs_dim)
        self.qp_start_timesteps = max(0, int(qp_start_timesteps))
        self.current_timestep = 0
        self.anneal_end_timesteps = int(kwargs.get("anneal_end_timesteps", 1_000_000))
        self.annealing_learning_alpha = bool(kwargs.get("annealing_learning_alpha", False))
        self.alpha_anneal_range = kwargs.get("alpha_anneal_range", None)
        if self.qp_obs_dim <= 6 or (self.qp_obs_dim - 6) % 6 != 0:
            raise ValueError(
                f"BarrierNet invalid qp_obs_dim={self.qp_obs_dim}. Expected 6 + 6*K with K>=1."
            )
        self.obs_topk = int((self.qp_obs_dim - 6) // 6)
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
            f"[CBFNet] robot_type={self.robot_type}, safe_dist={self.safe_dist:.3f}, umax={self.u_max}, obs_topk={self.obs_topk}, actor_input=polar, qp_start_timesteps={self.qp_start_timesteps}",
            flush=True,
        )
        self._qp_warm_start = None
        # self.predictor = TrajPredictor()

        self.fc1 = nn.Linear(self.actor_obs_dim, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21) 
        self.fc22 = nn.Linear(nHidden1, nHidden22) 
        self.fc31 = nn.Linear(nHidden21, self.nCls) 
        self.fc32 = nn.Linear(nHidden22, 1) 

    def set_timestep(self, timestep):
        self.current_timestep = max(0, int(timestep))

    @staticmethod
    def _opt_float(v, default):
        return float(default) if v is None else float(v)

    def _anneal_progress(self):
        if not self.training:
            return 1.0
        t1 = int(self.anneal_end_timesteps)
        if t1 <= 0:
            return 1.0
        p = float(self.current_timestep) / float(t1)
        return max(0.0, min(1.0, p))

    def _annealed_bounds(self, *, enabled, default_min, default_max, range_cfg, end_max_from_default=False):
        if not enabled or range_cfg is None:
            lo = float(default_min)
            hi = float(default_max)
        else:
            vals = [float(v) for v in list(range_cfg)]
            if end_max_from_default:
                if len(vals) == 3:
                    lo0, hi0, lo1 = vals
                    hi1 = float(default_max)
                elif len(vals) == 4:
                    lo0, hi0, lo1, _ = vals
                    hi1 = float(default_max)
                else:
                    raise ValueError("alpha_anneal_range must have 3 or 4 numbers: [min_start, max_start, min_end, (optional) max_end].")
            else:
                if len(vals) != 4:
                    raise ValueError("Anneal range must have 4 numbers: [min_start, max_start, min_end, max_end].")
                lo0, hi0, lo1, hi1 = vals
            p = self._anneal_progress()
            lo = lo0 + (lo1 - lo0) * p
            hi = hi0 + (hi1 - hi0) * p
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def _map_alpha_from_sigmoid(self, alpha_logits, default_max):
        lo, hi = self._annealed_bounds(
            enabled=self.annealing_learning_alpha,
            default_min=0.0,
            default_max=float(default_max),
            range_cfg=self.alpha_anneal_range,
            end_max_from_default=True,
        )
        sig = torch.sigmoid(alpha_logits).squeeze(-1)
        return float(lo) + (float(hi) - float(lo)) * sig

    def forward(self, obs_actor, obs_qp=None):
        if isinstance(obs_actor, np.ndarray):
            obs_actor = torch.tensor(obs_actor, dtype=torch.float)
        if obs_qp is None:
            raise ValueError("BarrierNet(v1) requires dual input: obs_actor (polar) and obs_qp (relative).")
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
                f"BarrierNet expected obs_actor dim={self.actor_obs_dim}, got {obs_actor.size(1)}."
            )
        if obs_qp.size(1) != self.qp_obs_dim:
            raise ValueError(
                f"BarrierNet expected obs_qp dim={self.qp_obs_dim}, got {obs_qp.size(1)}."
            )

        x = F.silu(self.fc1(obs_actor))
        x21 = F.silu(self.fc21(x))
        x22 = F.silu(self.fc22(x))
        
        x31 = self.fc31(x21) # unominal
        unom = x31

        # x32 = self.fc32(x22) # alpha
        # x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        alpha = self._map_alpha_from_sigmoid(self.fc32(x22), default_max=self.alpha_max)
        self.last_alpha = alpha

        # Warmup phase: bypass QP during training.
        if self.training and int(self.current_timestep) < int(self.qp_start_timesteps):
            return unom

        # BarrierNet dispatch based on robot type
        if self.robot_type == 'single_integrator':
            x = self.dCBF_SI(obs_qp, unom, alpha)
        elif self.robot_type == 'unicycle':
            x = self.dCBF_Unicycle(obs_qp, unom, alpha)
        elif self.robot_type == 'unicycle_dynamic':
            # unicycle_dynamic path intentionally disabled.
            x = torch.zeros_like(unom)
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
        """Parse fixed relative QP observation: [robot(6), K * (rel_x, rel_y, vx, vy, radius, mask)]."""
        nBatch = obs.size(0)
        if obs.size(1) != self.qp_obs_dim:
            raise ValueError(
                f"BarrierNet(v1) expected relative obs_qp dim={self.qp_obs_dim}, got {obs.size(1)}."
            )
        blocks = obs[:, 6:].reshape(nBatch, self.obs_topk, 6)
        rel = blocks[:, :, 0:2]
        vel = blocks[:, :, 2:4]
        mask = blocks[:, :, 5].clamp(0.0, 1.0)
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

        rel, vel, mask = self._extract_obstacle_blocks(obs)
        rel_x = rel[:, 0, 0]
        rel_y = rel[:, 0, 1]
        v_hx = vel[:, 0, 0]
        v_hy = vel[:, 0, 1]
        active = mask[:, 0]

        R_safe = self.safe_dist
        dist_sq = rel_x**2 + rel_y**2
        barrier = dist_sq - R_safe**2

        lg1 = 2 * rel_x
        lg2 = 2 * rel_y
        lf = -2 * rel_x * v_hx - 2 * rel_y * v_hy

        # -Lg*u - s <= Lf + alpha*h
        G_cbf = torch.stack(
            [-(lg1 * active), -(lg2 * active), -active],
            dim=1,
        ).to(device).unsqueeze(1)
        h_qp = ((lf + alpha * barrier) * active).unsqueeze(1).to(device)

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

        # Original lookahead-space cost (kept for reference):
        # min ||J u - u_nom_xy||^2
        # JT = J.transpose(1, 2)
        # Q = 2.0 * torch.bmm(JT, J)
        # Q = Q + 1e-6 * torch.eye(self.nCls, device=device, dtype=obs.dtype).unsqueeze(0)
        # p = -2.0 * torch.bmm(JT, u_nom_xy.unsqueeze(-1)).squeeze(-1)

        # New control-space cost:
        # min ||u - u_nom_vw||^2
        # Here we interpret actor output as nominal unicycle control [v, w].
        u_nom_vw = u_nom_xy
        Q = 2.0 * torch.eye(self.nCls, device=device, dtype=obs.dtype).unsqueeze(0).expand(
            nBatch, self.nCls, self.nCls
        )
        p = -2.0 * u_nom_vw

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
        # unicycle_dynamic path intentionally disabled.
        return torch.zeros_like(u_nom)

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
#         x = F.silu(self.fc1(x))
        
#         x21 = F.silu(self.fc21(x))
#         x22 = F.silu(self.fc22(x))

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
