
from os import error
import numpy as np
from crowd_sim.env.robot.agents import RobotModel

class GMMNoiseModel:
    def __init__(self, weights=None, means=None, stds=None, seed=123, lateral_ratio=0.3):
        if weights is None:
            # Forward has highest probability, left/right are smaller.
            weights = [0.6, 0.2, 0.2]
            # weights = [1.0]
        if means is None:
            means = [np.zeros(2), np.zeros(2), np.zeros(2)]
            # means = [np.zeros(2)]
        if stds is None:
            # Per-component std (spherical): [forward, left, right]
            stds = [0.1, 0.2, 0.2]
            # stds = [0.0]
            
        self.weights_ = np.array(weights)
        self.means_ = np.array(means)
        # Store variances (sigma^2) to align with sklearn's covariances_ (spherical)
        self.covariances_ = np.array(stds) ** 2
        
        # Save base parameters for dynamic updates
        self.base_weights = self.weights_.copy()
        self.base_stds = np.array(stds)
        self.lateral_ratio = float(lateral_ratio)
        
        self.rng = np.random.default_rng(seed)

    def _build_component_means(self, nominal_action):
        v = np.asarray(nominal_action, dtype=float)
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            means = np.stack([v, v, v])
            return means, 0.0

        f = v / speed
        left = np.array([-f[1], f[0]])
        lat_mag = self.lateral_ratio * speed

        mu0 = v
        muL = v + lat_mag * left
        muR = v - lat_mag * left

        # Preserve original speed for left/right modes
        for mu in (muL, muR):
            mu_norm = np.linalg.norm(mu)
            if mu_norm > 1e-6:
                mu[:] = mu / mu_norm * speed

        means = np.stack([mu0, muL, muR], axis=0)
        return means, speed


    def sample(self, nominal_action=None):
        # 1. Select Component
        component_idx = self.rng.choice(len(self.weights_), p=self.weights_)
        
        # 2. Sample velocity from selected component
        means, _ = self._build_component_means(nominal_action)
        mu = means[component_idx]

        # For spherical, covariance contains variances (sigma^2)
        sigma = np.sqrt(self.covariances_[component_idx])

        # Disable speed-based scaling for now (always scale = 1.0)
        sample = self.rng.normal(mu, sigma, size=2)
        
        return sample
    
    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

class SingleIntegrator(RobotModel):
    def __init__(self, dt, radius=0.3, umax=1.0):
        super().__init__(dt)
        self.radius = radius
        self.vmax = umax
        
        # GMM Noise Parameters
        self.gmm = GMMNoiseModel()
        self.val_history = []
        self.u = None


    def reset(self, initial_pos):
        # state: [x, y]
        self.state = np.array(initial_pos, dtype=np.float32)
        self.val_history = []
        self.pos = self.state[0:2]
        self.u = np.zeros(2, dtype=np.float32)


    def step(self, action):
        # action: [vx, vy]
        
        # self.val_history.append(action)
        # if len(self.val_history) > 10:
        #     self.val_history.pop(0)
        # self.gmm.update_distribution(self.val_history, self.dt)

        # Optional: Clip action magnitude
        speed = np.linalg.norm(action)
        if speed > self.vmax:
            action = action / speed * self.vmax
        self.u = np.asarray(action, dtype=np.float32).reshape(-1)

        self.state += self.u * self.dt
        return self.state

    def nominal_controller(self, state, goal):
        """
        Simple PD control to generate a nominal action (velocity).
        state: current position [x, y]
        goal: goal position [gx, gy]
        """
        k_p = 2.0
        error = goal - state
        nominal_action = k_p * error        
        # Clip action magnitude
        action = nominal_action
        speed = np.linalg.norm(action)
        if speed > self.vmax:
            action = action / speed * self.vmax
        return action

    def apply_gmm(self, action, state=None, goal=None):
        """
        Apply GMM perturbation to a nominal action after it is generated.
        Optionally skip noise when close to goal.
        """
        dist = np.linalg.norm(goal - state)
        if dist < self.radius - 0.1:
            return action

        speed = np.linalg.norm(action)
        action = self.gmm.sample(nominal_action=action)

        # speed = np.linalg.norm(action)
        # if speed > self.vmax:
        #     action = action / speed * self.vmax
        
        return action

    def get_state(self):
        return self.state
    
    def get_pos(self):
        return self.state
    

    
