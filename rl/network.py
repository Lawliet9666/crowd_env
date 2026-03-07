import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        out = self.fc2(F.silu(self.fc1(x)))
        return x + out
    
class FCNet(nn.Module):
    def __init__(self, 
                 nFeatures,
                 nCls,
                 nHidden1 = 256, 
                 nHidden21 = 256, 
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
        self.nCls = nCls
        self.safe_dist = safe_dist
        self.outputs_real_action = False


        self.fc1 = nn.Linear(nFeatures, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21) 
        self.fc31 = nn.Linear(nHidden21, nCls) 

        # self.fc_in = nn.Linear(nFeatures, nHidden1)
        # self.res1 = ResidualMLPBlock(nHidden1, nHidden21)
        # self.res2 = ResidualMLPBlock(nHidden1, nHidden21)
        # self.fc_out = nn.Linear(nHidden1, nCls)


        if self.nCls != 1:
            print(f"[FCNet] robot_type={robot_type}, safe_dist={safe_dist:.3f}",flush=True)

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        obs = obs.to(self.fc1.weight.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = obs.reshape(obs.size(0), -1)
        # assert obs.shape[1] == self.nFeatures, \
        #     f"Expected obs feature dim {self.nFeatures}, got {obs.shape[1]}"

        x = F.silu(self.fc1(obs))
        x21 = F.silu(self.fc21(x))
        x31 = self.fc31(x21)
        
        return x31
        
        
