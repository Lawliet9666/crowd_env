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
        self.alpha_fc1 = nn.Linear(self.qp_obs_dim, int(alpha_hidden1))
        self.alpha_fc2 = nn.Linear(int(alpha_hidden1), int(alpha_hidden2))
        self.alpha_out = nn.Linear(int(alpha_hidden2), 1)
        self.alpha_max = float(alpha_max)
        print(
            f"[CBFNet 2nets] alpha_from=relative(qp), alpha_max={self.alpha_max:.3f}",
            flush=True,
        )

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

        # alpha from relative QP observation branch
        xa = F.silu(self.alpha_fc1(obs_qp))
        xa = F.silu(self.alpha_fc2(xa))
        alpha = self._map_alpha_from_sigmoid(self.alpha_out(xa), default_max=self.alpha_max)
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
