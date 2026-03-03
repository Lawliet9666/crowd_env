import numpy as np
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_sim.utils import absolute_obs_to_relative

class SFAgentState:
    __slots__ = ("px", "py", "gx", "gy", "vx", "vy", "radius", "v_pref")

    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 0.0
        self.v_pref = 0.0

class SocialForceHelper:
    def __init__(self, dt, sf_params=None, max_humans=0):
        self.sf_params = sf_params or {}
        
        class DummyConfig:
            def __init__(self, sf_params, dt):
                class SFConfig:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                self.sf = SFConfig(sf_params) if isinstance(sf_params, dict) else sf_params
                self.env = type('EnvConfig', (), {'dt': dt})()
                
        self.policy = SOCIAL_FORCE(DummyConfig(self.sf_params, dt))
        self._self_state = SFAgentState()
        self._neighbor_pool = [SFAgentState() for _ in range(max(1, int(max_humans) + 1))]

    def reset(self):
        pass

    def set_seed(self, seed):
        self.policy.set_seed(seed)

    def _ensure_neighbor_pool(self, required):
        required = max(1, int(required))
        current = len(self._neighbor_pool)
        if current >= required:
            return
        self._neighbor_pool.extend(SFAgentState() for _ in range(required - current))

    @staticmethod
    def _val_at(values, idx):
        if np.isscalar(values):
            return float(values)
        return float(values[idx])

    @staticmethod
    def _fill_state(state, pos, vel, goal, radius, v_pref):
        state.px = float(pos[0])
        state.py = float(pos[1])
        state.gx = float(goal[0])
        state.gy = float(goal[1])
        state.vx = float(vel[0])
        state.vy = float(vel[1])
        state.radius = float(radius)
        state.v_pref = float(v_pref)

    @staticmethod
    def _expand_param(values, n):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return np.full((n,), float(arr), dtype=float)
        arr = arr.reshape(-1)
        if arr.size != n:
            raise ValueError(f"Expected size {n}, got {arr.size}")
        return arr

    def action_for_human(
        self,
        human_idx,
        human_positions,
        human_vels,
        human_goals,
        human_radii,
        human_vprefs,
        robot_pos=None,
        robot_vel=None,
        robot_radius=0.3,
        robot_vpref=1.0,
        include_robot=None,
    ):
        num_humans = len(human_positions)
        if include_robot is None:
            include_robot = bool(self.sf_params.get("avoid_robot", True))
        include_robot = bool(include_robot) and robot_pos is not None and robot_vel is not None

        self._fill_state(
            self._self_state,
            pos=human_positions[human_idx],
            vel=human_vels[human_idx],
            goal=human_goals[human_idx],
            radius=self._val_at(human_radii, human_idx),
            v_pref=self._val_at(human_vprefs, human_idx),
        )

        num_neighbors = (num_humans - 1) + (1 if include_robot else 0)
        self._ensure_neighbor_pool(num_neighbors)

        n = 0
        if include_robot:
            self._fill_state(
                self._neighbor_pool[n],
                pos=robot_pos,
                vel=robot_vel,
                goal=(0.0, 0.0),
                radius=robot_radius,
                v_pref=robot_vpref,
            )
            n += 1

        for j in range(num_humans):
            if j == human_idx:
                continue
            self._fill_state(
                self._neighbor_pool[n],
                pos=human_positions[j],
                vel=human_vels[j],
                goal=human_goals[j],
                radius=self._val_at(human_radii, j),
                v_pref=self._val_at(human_vprefs, j),
            )
            n += 1

        action = self.policy.predict_from_states(self._self_state, self._neighbor_pool[:n])
        return np.array([action.vx, action.vy], dtype=np.float32)

    def action_for_humans(
        self,
        human_positions,
        human_vels,
        human_goals,
        human_radii,
        human_vprefs,
        robot_pos=None,
        robot_vel=None,
        robot_radius=0.3,
        robot_vpref=1.0,
        include_robot=None,
    ):
        num_humans = len(human_positions)
        if num_humans == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if include_robot is None:
            include_robot = bool(self.sf_params.get("avoid_robot", True))
        include_robot = bool(include_robot) and robot_pos is not None and robot_vel is not None

        pos = np.asarray(human_positions, dtype=float).reshape(num_humans, 2)
        vel = np.asarray(human_vels, dtype=float).reshape(num_humans, 2)
        goals = np.asarray(human_goals, dtype=float).reshape(num_humans, 2)
        radii = self._expand_param(human_radii, num_humans)
        v_pref = np.maximum(self._expand_param(human_vprefs, num_humans), 0.0)

        # Pull force to goal (vectorized over humans).
        goal_delta = goals - pos
        dist_goal = np.linalg.norm(goal_delta, axis=1)
        desired_vel = np.zeros((num_humans, 2), dtype=float)
        valid_goal = dist_goal > 1e-5
        if np.any(valid_goal):
            desired_vel[valid_goal] = (
                goal_delta[valid_goal] / dist_goal[valid_goal, None]
            ) * v_pref[valid_goal, None]
        curr_delta = self.policy.KI * (desired_vel - vel)

        # Build neighbor set: all humans (+ optional robot) for pairwise interactions.
        if include_robot:
            all_pos = np.vstack([pos, np.asarray(robot_pos, dtype=float).reshape(1, 2)])
            all_radii = np.concatenate([radii, np.array([float(robot_radius)], dtype=float)])
        else:
            all_pos = pos
            all_radii = radii

        # Pairwise deltas from human i to neighbor j.
        delta = pos[:, None, :] - all_pos[None, :, :]   # (N, M, 2)
        dist = np.linalg.norm(delta, axis=2)            # (N, M)

        # Exclude self-self interaction for human neighbors.
        valid_neighbor = np.ones_like(dist, dtype=bool)
        valid_neighbor[np.arange(num_humans), np.arange(num_humans)] = False

        B = max(float(self.policy.B), 1e-5)
        sum_r = radii[:, None] + all_radii[None, :]
        non_overlap = valid_neighbor & (dist > 1e-5)
        overlap = valid_neighbor & (~non_overlap)

        interaction = np.zeros((num_humans, 2), dtype=float)

        if np.any(non_overlap):
            force_mag = self.policy.A * np.exp((sum_r - dist) / B)
            unit = np.zeros_like(delta)
            unit[non_overlap] = delta[non_overlap] / dist[non_overlap, None]
            interaction += np.sum(force_mag[:, :, None] * unit * non_overlap[:, :, None], axis=1)

        if np.any(overlap):
            # Keep behavior consistent with scalar implementation: random repulsion on exact overlap.
            overlap_force = self.policy.A * np.exp(sum_r / B)
            rand_angles = self.policy.rng.uniform(-np.pi, np.pi, size=dist.shape)
            rand_dirs = np.stack([np.cos(rand_angles), np.sin(rand_angles)], axis=-1)
            interaction += np.sum(
                overlap_force[:, :, None] * rand_dirs * overlap[:, :, None], axis=1
            )

        total_delta = (curr_delta + interaction) * float(self.policy.config.env.dt)
        new_vel = vel + total_delta

        speed = np.linalg.norm(new_vel, axis=1)
        clip_mask = speed > v_pref
        if np.any(clip_mask):
            new_vel[clip_mask] = new_vel[clip_mask] / speed[clip_mask, None] * v_pref[clip_mask, None]

        return new_vel.astype(np.float32, copy=False)

    def action_for_robot(
        self,
        robot_pos,
        robot_vel,
        robot_goal,
        robot_radius,
        robot_vpref,
        human_positions,
        human_vels,
        human_goals,
        human_radii,
        human_vprefs,
    ):
        num_humans = len(human_positions)
        self._fill_state(
            self._self_state,
            pos=robot_pos,
            vel=robot_vel,
            goal=robot_goal,
            radius=robot_radius,
            v_pref=robot_vpref,
        )

        self._ensure_neighbor_pool(num_humans)
        for j in range(num_humans):
            self._fill_state(
                self._neighbor_pool[j],
                pos=human_positions[j],
                vel=human_vels[j],
                goal=human_goals[j],
                radius=self._val_at(human_radii, j),
                v_pref=self._val_at(human_vprefs, j),
            )

        action = self.policy.predict_from_states(self._self_state, self._neighbor_pool[:num_humans])
        return np.array([action.vx, action.vy], dtype=np.float32)

def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

class SocialForceController:
    """
    Robot-side Social Force controller for evaluation in main_opt.py.
    """

    def __init__(self, config_file, env=None):
        self.robot_type = config_file.robot_params["type"]
        env_params = config_file.env_params
        human_params = config_file.human_params
        dt = float(env_params["dt"])
        sf_params = human_params.get("sf", {})
        self.max_obstacles_obs = int(env_params.get("max_obstacles_obs", human_params.get("num_humans", 10)))
        self.robot_vpref = float(config_file.robot_params.get("vmax", 1.0))
        hv = human_params.get("vmax", 1.0)
        self.human_vpref = float(hv if np.isscalar(hv) else hv[1])

        self.k_omega = float(env_params.get("unicycle_k_omega", 2.0))
        self._helper = SocialForceHelper(
            dt=dt,
            sf_params=sf_params,
            max_humans=self.max_obstacles_obs,
        )
        print(f"[SocialForceController] robot_type={self.robot_type}", flush=True)

    def _parse_observation(self, obs):
        obs = absolute_obs_to_relative(obs).astype(float, copy=False).reshape(-1)

        goal_rel = obs[0:2]      # robot_pos - goal_pos
        robot_vel = obs[2:4]     # world-frame velocity
        theta = float(obs[4])
        robot_radius = float(obs[5])

        if obs.size >= 12 and (obs.size - 6) % 6 == 0:
            blocks = obs[6:].reshape(-1, 6)
            # absolute_obs_to_relative stores obstacle relative position as (p_r - p_h).
            # Social-force helper expects neighbor positions in robot-centered frame as (p_h - p_r).
            human_rel_positions = -blocks[:, 0:2].astype(float)
            human_vels = blocks[:, 2:4].astype(float)
            human_radii = blocks[:, 4].astype(float)
            human_masks = np.clip(blocks[:, 5].astype(float), 0.0, 1.0)
            return goal_rel, robot_vel, theta, robot_radius, human_rel_positions, human_vels, human_radii, human_masks

        raise ValueError(f"Unsupported observation shape for SocialForceController: {obs.shape}")

    def get_action(self, obs=None):
        if obs is None:
            raise ValueError("SocialForceController.get_action requires observation input")

        goal_rel, robot_vel, theta, robot_radius, human_rel_positions, human_vels, human_radii, human_masks = self._parse_observation(obs)

        if human_rel_positions.shape[0] > 0:
            m = human_masks.reshape(-1, 1)
            far = np.array([1e3, 1e3], dtype=float)
            human_rel_positions = np.where(m > 0.5, human_rel_positions, far)
            human_vels = np.where(m > 0.5, human_vels, 0.0)
            human_radii = np.where(human_masks > 0.5, human_radii, 0.0)
            human_vprefs = np.where(human_masks > 0.5, self.human_vpref, 0.0).astype(float)
        else:
            human_vprefs = np.zeros((0,), dtype=float)

        n = human_rel_positions.shape[0]
        human_goals = np.zeros((n, 2), dtype=float)

        action_xy = self._helper.action_for_robot(
            robot_pos=np.zeros(2, dtype=float),
            robot_vel=robot_vel,
            robot_goal=-goal_rel,
            robot_radius=robot_radius,
            robot_vpref=self.robot_vpref,
            human_positions=human_rel_positions,
            human_vels=human_vels,
            human_goals=human_goals,
            human_radii=human_radii,
            human_vprefs=human_vprefs,
        )

        if self.robot_type == "single_integrator":
            return action_xy.astype(np.float32)

        if self.robot_type in ("unicycle", "unicycle_dynamic"):
            vx, vy = float(action_xy[0]), float(action_xy[1])
            v = vx * np.cos(theta) + vy * np.sin(theta)
            heading = np.arctan2(vy, vx)
            omega = self.k_omega * _wrap_to_pi(heading - theta)
            if self.robot_type == "unicycle":
                return np.array([v, omega], dtype=np.float32)
            return np.array([omega, 0.0], dtype=np.float32)
