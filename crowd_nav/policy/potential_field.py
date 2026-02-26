import numpy as np

from crowd_nav.policy.policy import Policy
from crowd_sim.env.robot.action import ActionXY


class POTENTIAL_FIELD(Policy):
    def __init__(self, config):
        super().__init__(config)
        cfg = config.get("pf", config) if isinstance(config, dict) else getattr(config, "pf", {})
        if not isinstance(cfg, dict):
            cfg = {}
        self.name = "potential_field"

        self.human_margin = float(cfg.get("human_margin", 0.1))
        self.human_influence = float(cfg.get("human_influence", 0.5))
        self.human_gain = float(cfg.get("human_gain", 1.5))
        self.robot_margin = float(cfg.get("robot_margin", 0.15))
        self.robot_influence = float(cfg.get("robot_influence", 0.8))
        self.robot_gain = float(cfg.get("robot_gain", 2.0))
        self.goal_eps = float(cfg.get("goal_eps", 1e-6))

    def predict(self, state):
        return self.predict_from_states(state.self_state, state.human_states)

    @staticmethod
    def _repulsion(delta_x, delta_y, dist, min_dist, influence, gain):
        trigger_dist = min_dist + influence
        if dist <= 1e-8 or dist >= trigger_dist:
            return 0.0, 0.0
        rep = (trigger_dist - dist) * gain
        return (delta_x / dist) * rep, (delta_y / dist) * rep

    def predict_from_states(self, self_state, human_states):
        # Desired velocity toward goal.
        goal_dx = self_state.gx - self_state.px
        goal_dy = self_state.gy - self_state.py
        goal_dist = np.hypot(goal_dx, goal_dy)
        if goal_dist > self.goal_eps:
            vd_x = (goal_dx / goal_dist) * self_state.v_pref
            vd_y = (goal_dy / goal_dist) * self_state.v_pref
        else:
            vd_x = 0.0
            vd_y = 0.0

        # Repulsion from neighbors (humans and optionally robot).
        for other_state in human_states:
            delta_x = self_state.px - other_state.px
            delta_y = self_state.py - other_state.py
            dist = np.hypot(delta_x, delta_y)

            is_robot = bool(getattr(other_state, "is_robot", False))
            margin = self.robot_margin if is_robot else self.human_margin
            influence = self.robot_influence if is_robot else self.human_influence
            gain = self.robot_gain if is_robot else self.human_gain
            min_dist = self_state.radius + other_state.radius + margin

            rep_x, rep_y = self._repulsion(delta_x, delta_y, dist, min_dist, influence, gain)
            vd_x += rep_x
            vd_y += rep_y

        # Clamp to preferred speed.
        v_pref = max(0.0, float(self_state.v_pref))
        speed = np.hypot(vd_x, vd_y)
        if speed > v_pref > 1e-8:
            scale = v_pref / speed
            vd_x *= scale
            vd_y *= scale

        return ActionXY(vd_x, vd_y)
