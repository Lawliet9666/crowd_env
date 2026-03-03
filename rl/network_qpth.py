import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from qpth.qp import QPFunction
import numpy as np
from rl.my_classes import test_solver as solver

class BarrierNet(nn.Module):
    def __init__(self, 
                 nFeatures, 
                 nCls,
                 nHidden1 = 128, 
                 nHidden21 = 32, 
                 nHidden22 = 32, 
                 safe_dist = 0.8,
                 alpha = 2.0,
                 beta = 0.2,
                 **kwargs):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        self.safe_dist = safe_dist

        self.alpha = alpha
        self.last_alpha = alpha
        self._qp_warm_start = None
        
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

        x = F.silu(self.fc1(obs))
        x21 = F.silu(self.fc21(x))
        x22 = F.silu(self.fc22(x))
        
        x31 = self.fc31(x21) # unominal
        unom = x31

        # x32 = self.fc32(x22) # alpha
        # x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        alpha = 4*torch.sigmoid(self.fc32(x22)).squeeze(-1)  # (B,)
        self.last_alpha = alpha

        # BarrierNet for SI
        x = self.dCBF_SI(obs, unom, alpha)
               
        return x

    def dCBF_SI(self, obs, unom, alpha):
        nBatch = obs.size(0)
        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.fc1.weight.device) 
        
        p = -2 * unom  # Minimize ||u - u_nom||^2
        Q = Q * 2  # since 1/2 factor in QPFunction

        # obs structure: 
        # 0: px-gx, 1: py-gy, 2: vx, 3: vy, 4: theta 5: r_radius
        # 6: px-hx, 7: py-hy, 8: vx_h, 9: vy_h, 10: h_radius
        
        rel_x = obs[:, 6]
        rel_y = obs[:, 7]
        v_hx = obs[:, 8]
        v_hy = obs[:, 9]
        
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

        # Construct QP Constraints: -Lg*u <= Lf + alpha*h
        # Lg = [lg1, lg2] (gradient w.r.t robot position, which for SI is same as control u)
        
        G = torch.stack([-lg1, -lg2], dim=1).to(self.fc1.weight.device) 
        
        # For qpth, G has expected shape (nBatch, nConstraints, nVars) = (N, 1, 2)
        G = G.unsqueeze(1) 
             
        # Gamma is x32[:,0]
        # alpha = x32[:, 0]
        
        # h_qp = Lf + alpha * barrier
        # Shape need to be (nBatch, 1)
        h_qp = (lf + alpha * barrier).unsqueeze(1).to(self.fc1.weight.device) 
        
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
                     dtype=obs.dtype,
                     warm_start_x=self._qp_warm_start,
                 )
                 self._qp_warm_start = x.detach().cpu().numpy().reshape(-1)
            else:
                 x = QPFunction(verbose = 0, maxIter=40)(Q_qp, p_qp, G_qp, h_qp_qp, e_qp, e_qp)
        
        return x.to(dtype=obs.dtype)

        
        
        

# class BarrierNet(nn.Module):
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
