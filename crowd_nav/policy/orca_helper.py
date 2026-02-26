import numpy as np
import rvo2
from crowd_nav.policy.orca import ORCA


class OrcaAgentState:
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


class ORCAHelper:
    def __init__(self, dt, orca_params=None, max_humans=0):
        self.orca_params = orca_params or {}
        self.policy = ORCA(self.orca_params)
        self.policy.time_step = dt
        self._self_state = OrcaAgentState()
        self._neighbor_pool = [OrcaAgentState() for _ in range(max(1, int(max_humans) + 1))]
        self._multi_sim = None
        self._multi_total_agents = 0

    def reset(self):
        self.policy.sim = None
        self._multi_sim = None
        self._multi_total_agents = 0

    def _ensure_neighbor_pool(self, required):
        required = max(1, int(required))
        current = len(self._neighbor_pool)
        if current >= required:
            return
        self._neighbor_pool.extend(OrcaAgentState() for _ in range(required - current))

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
            include_robot = bool(self.orca_params.get("avoid_robot", True))
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
            include_robot = bool(self.orca_params.get("avoid_robot", True))
        include_robot = bool(include_robot) and robot_pos is not None and robot_vel is not None

        total_agents = num_humans + (1 if include_robot else 0)
        max_neighbors = max(total_agents - 1, 0)
        params = (
            self.policy.neighbor_dist,
            max_neighbors,
            self.policy.time_horizon,
            self.policy.time_horizon_obst,
        )

        if self._multi_sim is None or self._multi_total_agents != total_agents:
            self._multi_sim = rvo2.PyRVOSimulator(
                self.policy.time_step,
                *params,
                self.policy.radius if self.policy.radius is not None else 0.3,
                self.policy.max_speed,
            )
            for i in range(num_humans):
                max_speed = self._val_at(human_vprefs, i)
                self._multi_sim.addAgent(
                    (float(human_positions[i][0]), float(human_positions[i][1])),
                    *params,
                    float(self._val_at(human_radii, i)) + 0.01 + self.policy.safety_space,
                    max_speed,
                    (float(human_vels[i][0]), float(human_vels[i][1])),
                )
            if include_robot:
                self._multi_sim.addAgent(
                    (float(robot_pos[0]), float(robot_pos[1])),
                    *params,
                    float(robot_radius) + 0.01 + self.policy.safety_space,
                    float(robot_vpref),
                    (float(robot_vel[0]), float(robot_vel[1])),
                )
            self._multi_total_agents = total_agents
        else:
            for i in range(num_humans):
                self._multi_sim.setAgentPosition(i, (float(human_positions[i][0]), float(human_positions[i][1])))
                self._multi_sim.setAgentVelocity(i, (float(human_vels[i][0]), float(human_vels[i][1])))
            if include_robot:
                self._multi_sim.setAgentPosition(num_humans, (float(robot_pos[0]), float(robot_pos[1])))
                self._multi_sim.setAgentVelocity(num_humans, (float(robot_vel[0]), float(robot_vel[1])))

        for i in range(num_humans):
            goal_vec = np.array(
                [
                    float(human_goals[i][0]) - float(human_positions[i][0]),
                    float(human_goals[i][1]) - float(human_positions[i][1]),
                ],
                dtype=float,
            )
            speed = np.linalg.norm(goal_vec)
            v_pref = max(0.0, float(self._val_at(human_vprefs, i)))
            if speed > 1e-6 and v_pref > 0.0:
                pref_speed = min(speed, v_pref)
                pref_vel = goal_vec / speed * pref_speed
            else:
                pref_vel = np.zeros(2, dtype=float)
            self._multi_sim.setAgentPrefVelocity(i, (float(pref_vel[0]), float(pref_vel[1])))

        if include_robot:
            # Humans avoid robot as a dynamic obstacle; robot intent is unknown here.
            self._multi_sim.setAgentPrefVelocity(num_humans, (0.0, 0.0))

        self._multi_sim.doStep()
        actions = np.zeros((num_humans, 2), dtype=np.float32)
        for i in range(num_humans):
            vx, vy = self._multi_sim.getAgentVelocity(i)
            actions[i, 0] = float(vx)
            actions[i, 1] = float(vy)
        return actions

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


class ORCAController:
    """
    Robot-side ORCA controller for evaluation in main_opt.py.
    """

    def __init__(self, config_file, env=None):
        # Keep env arg for API compatibility, but this controller is observation-driven.
        self.robot_type = config_file.robot_params["type"]
        env_params = config_file.env_params
        human_params = config_file.human_params
        dt = float(env_params["dt"])
        orca_params = human_params.get("orca", {})
        self.max_obstacles_obs = int(env_params.get("max_obstacles_obs", human_params.get("num_humans", 10)))
        self.robot_vpref = float(config_file.robot_params.get("vmax", 1.0))
        hv = human_params.get("vmax", 1.0)
        self.human_vpref = float(hv if np.isscalar(hv) else hv[1])

        self.k_omega = float(env_params.get("unicycle_k_omega", 2.0))
        self._helper = ORCAHelper(
            dt=dt,
            orca_params=orca_params,
            max_humans=self.max_obstacles_obs,
        )
        print(f"[ORCAController] robot_type={self.robot_type}", flush=True)

    def _parse_observation(self, obs):
        obs = np.asarray(obs, dtype=float).reshape(-1)

        goal_rel = obs[0:2]      # robot_pos - goal_pos
        robot_vel = obs[2:4]     # world-frame velocity
        theta = float(obs[4])
        robot_radius = float(obs[5])

        # Flat observation format: 6 + K*6 with [rel_x, rel_y, vx, vy, radius, mask]
        if obs.size >= 12 and (obs.size - 6) % 6 == 0:
            blocks = obs[6:].reshape(-1, 6)
            human_rel_positions = blocks[:, 0:2].astype(float)
            human_vels = blocks[:, 2:4].astype(float)
            human_radii = blocks[:, 4].astype(float)
            human_masks = np.clip(blocks[:, 5].astype(float), 0.0, 1.0)
            return goal_rel, robot_vel, theta, robot_radius, human_rel_positions, human_vels, human_radii, human_masks

        # # Legacy single-obstacle format: [.., rel_rx_hx, rel_ry_hy, hvx, hvy, h_radius]
        # # where rel is (p_r - p_h). Convert to (p_h - p_r) by negating.
        # if obs.size >= 11:
        #     rel_r_minus_h = obs[6:8]
        #     hv = obs[8:10]
        #     hr = float(obs[10])
        #     # No mask in legacy format; if rel and vel are both ~0 treat as "no visible obstacle".
        #     if np.linalg.norm(rel_r_minus_h) < 1e-8 and np.linalg.norm(hv) < 1e-8:
        #         human_positions = np.zeros((0, 2), dtype=float)
        #         human_vels = np.zeros((0, 2), dtype=float)
        #         human_radii = np.zeros((0,), dtype=float)
        #     else:
        #         human_positions = np.asarray([[-rel_r_minus_h[0], -rel_r_minus_h[1]]], dtype=float)
        #         human_vels = np.asarray([[hv[0], hv[1]]], dtype=float)
        #         human_radii = np.asarray([hr], dtype=float)
        #     return goal_rel, robot_vel, theta, robot_radius, human_positions, human_vels, human_radii

        raise ValueError(f"Unsupported observation shape for ORCAController: {obs.shape}")

    def get_action(self, obs=None):
        if obs is None:
            raise ValueError("ORCAController.get_action requires observation input")

        goal_rel, robot_vel, theta, robot_radius, human_rel_positions, human_vels, human_radii, human_masks = self._parse_observation(obs)

        # Keep fixed K obstacle slots: masked slots become far-away dummy neighbors.
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
            # Positions are relative (p_h - p_r) in robot-centered translated frame.
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

        raise ValueError(f"Unsupported robot type for ORCA controller: {self.robot_type}")
