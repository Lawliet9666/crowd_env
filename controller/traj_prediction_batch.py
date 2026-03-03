import torch
import torch.nn as nn

class TrajPredictorTorch(nn.Module):
    def __init__(
        self,
        n_components=3,
        lateral_ratio=0.3,
        seed=1234,
        weights=None,
        stds=None,
    ):
        super().__init__()
        self.n_components = n_components
        self.lateral_ratio = float(lateral_ratio)

        if weights is None:
            weights = [0.6, 0.2, 0.2]
        if stds is None:
            stds = [0.1, 0.2, 0.2]

        weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        stds = torch.as_tensor(stds, dtype=torch.float32).reshape(-1)
        variances = stds ** 2

        if weights.numel() != n_components:
            raise ValueError(f"weights size {weights.numel()} != n_components {n_components}")
        if variances.numel() != n_components:
            raise ValueError(f"stds size {variances.numel()} != n_components {n_components}")

        # register_buffer: moved with .to(device), saved in state_dict, but not trainable params
        self.register_buffer("gmm_weights", weights)        # (M,)
        self.register_buffer("gmm_variances", variances)    # (M,)

        # self._rng = torch.Generator()
        # self._rng.manual_seed(seed)

    def _build_component_means(self, current_vel: torch.Tensor) -> torch.Tensor:
        # current_vel: (B,2)
        speed = torch.linalg.norm(current_vel, dim=1, keepdim=True)  # (B,1)
        eps = 1e-6
        f = current_vel / (speed + eps)
        left = torch.stack([-f[:, 1], f[:, 0]], dim=1)  # (B,2)
        lat_mag = self.lateral_ratio * speed

        mu0 = current_vel
        muL = current_vel + lat_mag * left
        muR = current_vel - lat_mag * left

        # Preserve original speed for left/right modes
        muL_norm = torch.linalg.norm(muL, dim=1, keepdim=True)
        muR_norm = torch.linalg.norm(muR, dim=1, keepdim=True)
        scaleL = torch.where(muL_norm > eps, speed / muL_norm, torch.ones_like(muL_norm))
        scaleR = torch.where(muR_norm > eps, speed / muR_norm, torch.ones_like(muR_norm))
        muL = muL * scaleL
        muR = muR * scaleR

        means = torch.stack([mu0, muL, muR], dim=1)  # (B,3,2)

        # If speed is near zero, all modes collapse to current_vel
        mask = (speed.squeeze(1) < eps)
        if torch.any(mask):
            means[mask] = current_vel[mask].unsqueeze(1).expand(-1, self.n_components, 2)

        return means
    
    # def _build_component_means(self, current_vel):
    #     # current_vel: (B,2) or (2,)
    #     if current_vel.dim() == 1:
    #         current_vel = current_vel.unsqueeze(0)
    #     v = current_vel
    #     speed = torch.linalg.norm(v, dim=1, keepdim=True)  # (B,1)

    #     small = speed < 1e-6
    #     f = torch.where(small, v, v / speed)
    #     left = torch.stack([-f[:, 1], f[:, 0]], dim=1)
    #     lat_mag = self.lateral_ratio * speed

    #     mu0 = v
    #     muL = v + lat_mag * left
    #     muR = v - lat_mag * left

    #     for mu in (muL, muR):
    #         mu_norm = torch.linalg.norm(mu, dim=1, keepdim=True)
    #         valid = mu_norm > 1e-6
    #         mu[valid] = mu[valid] / mu_norm[valid] * speed[valid]

    #     means = torch.stack([mu0, muL, muR], dim=1)  # (B,3,2)
    #     return means, speed.squeeze(1)

    @torch.no_grad()
    def predict_gmm(self, current_vel):
        """
        current_vel: torch.Tensor (B,2) or (2,)
        Returns:
          weights:   (B,M)
          means:     (B,M,2)
                    covariances: (B,M)
        """
        if current_vel.dim() == 1:
            current_vel = current_vel.unsqueeze(0)  # (1,2)
        B = current_vel.size(0)
        M = self.n_components
        device = current_vel.device

        means = self._build_component_means(current_vel)
        weights = self.gmm_weights.to(device).unsqueeze(0).expand(B, M).contiguous()
        covariances = self.gmm_variances.to(device).unsqueeze(0).expand(B, M).contiguous()

        return weights, means, covariances

    # @torch.no_grad()
    # def predict_vel(self, current_vel):
    #     """
    #     current_vel: torch.Tensor (B,2) or (2,)
    #     Returns:
    #         sampled_vel: (B,2)
    #     """
    #     if current_vel.dim() == 1:
    #             current_vel = current_vel.unsqueeze(0)
    #     B = current_vel.size(0)
    #     device = current_vel.device

    #     weights, means, covariances = self.predict_gmm(current_vel)
    #     comp_idx = torch.multinomial(weights, num_samples=1, generator=self._rng).squeeze(1)  # (B,)

    #     mean_sel = means[torch.arange(B, device=device), comp_idx]  # (B,2)
    #     var_sel = covariances[torch.arange(B, device=device), comp_idx]  # (B,)
    #     std_sel = torch.sqrt(var_sel).unsqueeze(1)

    #     noise = torch.randn((B, 2), generator=self._rng, device=device)
    #     return mean_sel + noise * std_sel

    @torch.no_grad()
    def predict_vel_expectation(self, current_vel):
        """
        current_vel: torch.Tensor (B,2) or (2,)
        Returns:
            expected_vel: (B,2)
        """
        # if current_vel.dim() == 1:
        #     current_vel = current_vel.unsqueeze(0)
        # weights, means, _ = self.predict_gmm(current_vel)
        # return torch.sum(weights.unsqueeze(-1) * means, dim=1)
        return current_vel
