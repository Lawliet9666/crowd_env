import numpy as np

from crowd_sim.utils import absolute_obs_to_relative


class NominalController:
    """
    Pure PID-style go-to-goal controller (no collision avoidance).
    This controller only tracks the goal and ignores obstacle terms.
    """

    def __init__(self, config_file, env=None):
        robot_params = config_file.robot_params
        ctrl_params = config_file.controller_params
        env_params = config_file.env_params

        self.env = env
        self.robot_type = str(robot_params["type"])
        self.robot_radius = float(robot_params["radius"])
        self.dt = float(env_params.get("dt", 0.1))

        self.vmax = float(robot_params.get("vmax", 1.0))
        self.omega_max = float(robot_params.get("omega_max", np.pi / 2.0))
        self.amax = float(robot_params.get("amax", 2.0))

        nominal_cfg = ctrl_params.get("nominal", {})
        self.kp_xy = float(nominal_cfg.get("kp_xy", nominal_cfg.get("goal_k", 1.0)))
        self.ki_xy = float(nominal_cfg.get("ki_xy", 0.0))
        self.kd_xy = float(nominal_cfg.get("kd_xy", 0.2))
        self.i_clip = float(nominal_cfg.get("i_clip", 2.0))
        self.goal_tol = float(nominal_cfg.get("goal_tol", 0.15))
        self.k_omega = float(nominal_cfg.get("k_omega", env_params.get("unicycle_k_omega", 2.0)))
        self.k_acc = float(nominal_cfg.get("k_acc", 2.0))
        self._int_err = np.zeros(2, dtype=np.float64)

        print(
            (
                f"[NominalController] robot_type={self.robot_type}, vmax={self.vmax:.3f}, "
                f"omega_max={self.omega_max:.3f}, kp_xy={self.kp_xy:.3f}, "
                f"ki_xy={self.ki_xy:.3f}, kd_xy={self.kd_xy:.3f}"
            ),
            flush=True,
        )

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _clip_xy_speed(self, v_xy):
        v = np.asarray(v_xy, dtype=float).reshape(2)
        speed = np.linalg.norm(v)
        if speed > self.vmax and speed > 1e-8:
            v = v / speed * self.vmax
        return v

    def _compute_xy_command(self, goal_rel, robot_vel):
        # Goal vector in world frame.
        goal_vec = -np.asarray(goal_rel, dtype=np.float64).reshape(2)
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= self.goal_tol:
            self._int_err[:] = 0.0
            return np.zeros(2, dtype=np.float32)

        # PID in XY velocity command space:
        #   u = Kp * e + Ki * integral(e) + Kd * d(e)/dt
        # Here d(e)/dt = -v_robot because goal is static.
        self._int_err += goal_vec * self.dt
        self._int_err = np.clip(self._int_err, -self.i_clip, self.i_clip)
        d_err = -np.asarray(robot_vel, dtype=np.float64).reshape(2)

        u_xy = (
            self.kp_xy * goal_vec
            + self.ki_xy * self._int_err
            + self.kd_xy * d_err
        )
        u_xy = self._clip_xy_speed(u_xy)
        return u_xy.astype(np.float32)

    def get_action(self, obs):
        # obs = absolute_obs_to_relative(obs)

        goal_rel = np.asarray(obs[0:2], dtype=np.float64)
        robot_vel = np.asarray(obs[2:4], dtype=np.float64)
        theta = float(np.float64(obs[4]))
        u_xy = self._compute_xy_command(goal_rel, robot_vel)

        if self.robot_type == "single_integrator":
            return u_xy.astype(np.float32)

        heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        speed = float(np.linalg.norm(u_xy))

        if speed <= 1e-8:
            heading_des = theta
            v_des = 0.0
        else:
            heading_des = float(np.arctan2(u_xy[1], u_xy[0]))
            v_des = float(np.dot(u_xy, heading))

        omega = self.k_omega * self._wrap_to_pi(heading_des - theta)
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        v_des = float(np.clip(v_des, -self.vmax, self.vmax))

        if self.robot_type == "unicycle":
            return np.array([v_des, omega], dtype=np.float32)

        if self.robot_type == "unicycle_dynamic":
            # unicycle_dynamic path intentionally disabled.
            return np.zeros(2, dtype=np.float32)

        raise ValueError(f"Unsupported robot type for NominalController: {self.robot_type}")
