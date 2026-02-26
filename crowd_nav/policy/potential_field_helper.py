import numpy as np

from crowd_nav.policy.potential_field import POTENTIAL_FIELD


class PFAgentState:
    __slots__ = ("px", "py", "gx", "gy", "vx", "vy", "radius", "v_pref", "is_robot")

    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 0.0
        self.v_pref = 0.0
        self.is_robot = False


class PotentialFieldHelper:
    def __init__(self, dt, pf_params=None, max_humans=0):
        self.pf_params = pf_params or {}

        class DummyConfig:
            def __init__(self, pf_params, dt):
                class PFConfig:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)

                self.pf = PFConfig(pf_params) if isinstance(pf_params, dict) else pf_params
                self.env = type("EnvConfig", (), {"dt": dt})()

        self.policy = POTENTIAL_FIELD(DummyConfig(self.pf_params, dt))
        self._self_state = PFAgentState()
        self._neighbor_pool = [PFAgentState() for _ in range(max(1, int(max_humans) + 1))]

    def reset(self):
        pass

    def _ensure_neighbor_pool(self, required):
        required = max(1, int(required))
        current = len(self._neighbor_pool)
        if current >= required:
            return
        self._neighbor_pool.extend(PFAgentState() for _ in range(required - current))

    @staticmethod
    def _val_at(values, idx):
        if np.isscalar(values):
            return float(values)
        return float(values[idx])

    @staticmethod
    def _fill_state(state, pos, vel, goal, radius, v_pref, is_robot=False):
        state.px = float(pos[0])
        state.py = float(pos[1])
        state.gx = float(goal[0])
        state.gy = float(goal[1])
        state.vx = float(vel[0])
        state.vy = float(vel[1])
        state.radius = float(radius)
        state.v_pref = float(v_pref)
        state.is_robot = bool(is_robot)

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
            include_robot = bool(self.pf_params.get("avoid_robot", True))
        include_robot = bool(include_robot) and robot_pos is not None and robot_vel is not None

        self._fill_state(
            self._self_state,
            pos=human_positions[human_idx],
            vel=human_vels[human_idx],
            goal=human_goals[human_idx],
            radius=self._val_at(human_radii, human_idx),
            v_pref=self._val_at(human_vprefs, human_idx),
            is_robot=False,
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
                is_robot=True,
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
                is_robot=False,
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

        actions = np.zeros((num_humans, 2), dtype=np.float32)
        for i in range(num_humans):
            actions[i] = self.action_for_human(
                human_idx=i,
                human_positions=human_positions,
                human_vels=human_vels,
                human_goals=human_goals,
                human_radii=human_radii,
                human_vprefs=human_vprefs,
                robot_pos=robot_pos,
                robot_vel=robot_vel,
                robot_radius=robot_radius,
                robot_vpref=robot_vpref,
                include_robot=include_robot,
            )
        return actions
