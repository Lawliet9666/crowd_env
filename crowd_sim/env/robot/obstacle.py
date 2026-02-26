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
        # Closed-form normalization factor for left/right components.
        self._lat_scale = 1.0 / np.sqrt(1.0 + self.lateral_ratio * self.lateral_ratio)
        
        self.rng = np.random.default_rng(seed)

    def _build_component_mean(self, nominal_action, component_idx):
        v = np.asarray(nominal_action, dtype=float)
        vx, vy = float(v[0]), float(v[1])
        speed_sq = vx * vx + vy * vy
        if speed_sq < 1e-12:
            return v, 0.0

        speed = np.sqrt(speed_sq)
        idx = int(component_idx)
        if idx == 0:
            return np.array([vx, vy], dtype=float), speed

        r = self.lateral_ratio
        s = self._lat_scale
        if idx == 1:
            mu = np.array([s * (vx - r * vy), s * (vy + r * vx)], dtype=float)
        elif idx == 2:
            mu = np.array([s * (vx + r * vy), s * (vy - r * vx)], dtype=float)
        else:
            raise ValueError(f"Invalid component index: {component_idx}")

        return mu, speed

    # def _sample_component_indices(self, size):
    #     size = int(size)
    #     if size <= 0:
    #         return np.zeros((0,), dtype=np.int64)

    #     n_comp = int(len(self.weights_))
    #     if n_comp == 3:
    #         # Faster than rng.choice(..., p=...) for fixed 3-component mixtures.
    #         u = self.rng.random(size)
    #         t0 = float(self.weights_[0])
    #         t1 = t0 + float(self.weights_[1])
    #         idx = np.zeros(size, dtype=np.int64)
    #         idx[u >= t0] = 1
    #         idx[u >= t1] = 2
    #         return idx

    #     return self.rng.choice(n_comp, size=size, p=self.weights_).astype(np.int64, copy=False)

    def sample(self, nominal_action=None):
        # 1. Select component
        # component_idx = int(self._sample_component_indices(1)[0])
        component_idx = int(self.rng.choice(len(self.weights_), p=self.weights_))

        # 2. Build mean of selected component
        mu, _ = self._build_component_mean(nominal_action, component_idx)

        # 3. Sample velocity from selected spherical covariance (sigma^2)
        sigma = np.sqrt(self.covariances_[component_idx])
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
        # speed = np.linalg.norm(action)
        # if speed > self.vmax:
            # action = action / speed * self.vmax
        return action

    @staticmethod
    def apply_gmm_batch(humans, nominal_actions, states=None, goals=None, radii=None):
        """
        Batch GMM perturbation for multiple humans.
        Each human samples its own component index and Gaussian noise independently.
        """
        actions = np.asarray(nominal_actions, dtype=float)
        if actions.ndim == 1:
            actions = actions.reshape(1, 2)
        if actions.shape[1] != 2:
            raise ValueError(f"Expected nominal_actions with shape (N, 2), got {actions.shape}")

        num_humans = int(actions.shape[0])
        if num_humans == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(humans) != num_humans:
            raise ValueError(f"humans length ({len(humans)}) must match nominal_actions ({num_humans})")

        out = actions.copy()

        if states is not None and goals is not None:
            states_arr = np.asarray(states, dtype=float).reshape(num_humans, 2)
            goals_arr = np.asarray(goals, dtype=float).reshape(num_humans, 2)
            if radii is None:
                radii_arr = np.array([float(getattr(h, "radius", 0.0)) for h in humans], dtype=float)
            else:
                radii_arr = np.asarray(radii, dtype=float).reshape(num_humans)
            dist = np.linalg.norm(goals_arr - states_arr, axis=1)
            noise_mask = dist >= (2.0 * radii_arr)
        else:
            noise_mask = np.ones((num_humans,), dtype=bool)

        noisy_idx = np.nonzero(noise_mask)[0]
        for i in noisy_idx:
            out[i] = humans[i].gmm.sample(nominal_action=out[i])

        return out.astype(np.float32, copy=False)

    def apply_gmm(self, action, state=None, goal=None):
        """
        Apply GMM perturbation to a nominal action after it is generated.
        Optionally skip noise when close to goal.
        """
        dist = np.linalg.norm(goal - state)
        if dist < self.radius*2:
            return action

        # speed = np.linalg.norm(action)
        action = self.gmm.sample(nominal_action=action)

        # speed = np.linalg.norm(action)
        # if speed > self.vmax:
        #     action = action / speed * self.vmax
        
        return action

    def get_state(self):
        return self.state
    
    def get_pos(self):
        return self.state
    

    
