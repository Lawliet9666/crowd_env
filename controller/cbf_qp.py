import numpy as np
import cvxpy as cp
from controller.traj_prediction import TrajPredictor
from crowd_sim.utils import parse_obstacles, absolute_obs_to_relative

    
class CBFQPController:
    def __init__(self, config_file, env=None):
        # Load params from Config
        robot_params = config_file.robot_params
        human_params = config_file.human_params
        ctrl_params = config_file.controller_params

        self.env = env
        self.robot = env.robot
        self.robot_type = env.robot_type

        self.robot_radius = robot_params['radius']
        self.human_radius = human_params['radius']
        self.safe_dist = ctrl_params['safety_margin'] + self.robot_radius + self.human_radius   # Use discomfort_dist as safe_dist
        self.alpha = ctrl_params['cbf_alpha']
        self._base_safe_dist = float(self.safe_dist)
        self._base_alpha = float(self.alpha)
        
        # Handle umax depending on robot type
        if self.env.robot_type == 'single_integrator':
            self.umax = robot_params['vmax'] # For single integrator, u is v
        elif self.env.robot_type == 'unicycle':
            self.umax = [robot_params['vmax'], robot_params['omega_max']]
        elif self.env.robot_type == 'unicycle_dynamic':
            self.umax = [robot_params['vmax'], robot_params['amax'], robot_params['omega_max']]

        gmm_params = dict(human_params.get("gmm", {}))
        self.predictor = TrajPredictor(gmm_params=gmm_params)
        # Warm-start buffers: reuse previous QP solution as next initialization
        self._u_prev = None
        self.infeasible = False


        print(
            f"[CBFQPController] robot_type={self.robot_type}, safe_dist={self.safe_dist:.3f}, umax={self.umax}",
            flush=True,
        )

    def set_adaptive_params(self, alpha=None, safe_dist=None):
        if alpha is not None:
            self.alpha = float(alpha)
        if safe_dist is not None:
            self.safe_dist = float(safe_dist)

    def reset_adaptive_params(self):
        self.alpha = float(self._base_alpha)
        self.safe_dist = float(self._base_safe_dist)

    def _warm_start_var(self, var, prev_u):
        if prev_u is None:
            return
        var.value = np.asarray(prev_u, dtype=float).reshape(-1)

    def _solve_qp(self, prob):
        # Prefer OSQP with warm start for sequential timesteps.
        try:
            return prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            return prob.solve(warm_start=True)

    def _fallback_action(self, dim=2):
        if self._u_prev is not None:
            return self._u_prev.astype(np.float32).copy()
        return np.zeros(dim, dtype=np.float32)



    def get_action(self, obs):
        # Input Observation (Common for all robots):
        # 1. Robot FullState: [px-gx, py-gy, vx, vy, theta, r_radius] -> 6 dims
        # 2. Human ObservableState: [px-hx, py-hy, vx_h, vy_h, h_radius] -> 5 dims

        obs = absolute_obs_to_relative(obs)
        self.infeasible = False

        # Use float64 inside QP math for numerical stability.
        goal_rel = np.asarray(obs[0:2], dtype=np.float64)
        theta = float(np.float64(obs[4]))
        obstacle_rels, obstacle_vels, obstacle_radii, obstacle_masks = parse_obstacles(obs)

        
        if self.env.robot_type == "single_integrator":
            # 2. CBF Constraints
            # h(x) = distance^2 - min_safe_distance^2 >= 0
            # distance^2 = ||p_rob - p_hum||^2 = ||human_rel||^2
            # Objective: Minimize ||u - u_nom||^2
            R = self.safe_dist  # min safe distance

            u = cp.Variable(2)
            constraints = []

            u_nom = self.robot.nominal_input(goal_rel, theta) 
            u_nom = u_nom.flatten()
            objective = cp.Minimize(cp.sum_squares(u - u_nom))

            # One CBF constraint per observed obstacle.
            for i in range(obstacle_rels.shape[0]):
                human_rel = obstacle_rels[i]
                human_vel_curr = obstacle_vels[i]
                m = float(obstacle_masks[i])
                human_vel_pred = self.predictor.predict_vel_expectation(human_vel_curr)
                # Keep original safe distance behavior by default.
                # If per-obstacle radii are desired, replace R with:
                # self.alpha + self.robot_radius + obstacle_radii[i]
                h_val = np.dot(human_rel, human_rel) - R**2
                lhs = m * (2 * human_rel @ u)
                rhs = m * (-self.alpha * h_val + 2 * np.dot(human_rel, human_vel_pred))
                constraints.append(lhs >= rhs)
            
            v_limit = self.umax if np.isscalar(self.umax) else self.umax[0]
            constraints.append(u[0] <= v_limit)
            constraints.append(u[0] >= -v_limit)
            constraints.append(u[1] <= v_limit)
            constraints.append(u[1] >= -v_limit)
            
            self._warm_start_var(u, self._u_prev)
            prob = cp.Problem(objective, constraints)
            result = self._solve_qp(prob)
            if prob.status not in ["infeasible", "unbounded", None] and u.value is not None:
                u_sol = np.asarray(u.value, dtype=np.float64).reshape(-1)
                self._u_prev = u_sol
                return u_sol.astype(np.float32)
            else:
                self.infeasible = True
                return self._fallback_action(dim=2)

        elif self.env.robot_type == 'unicycle':
            # --- Unicycle Lookahead CBF ---
            u = cp.Variable(2) # [v, omega]
            constraints = []

            epsilon = 0.2 # Lookahead distance
            R =  self.safe_dist + epsilon

            c, s = np.cos(theta), np.sin(theta)
            # Jacobian mapping [v, w] -> [vx_L, vy_L]
            J = np.array([[c, -epsilon*s],
                          [s, epsilon*c]])
            
            # u_nom = self.robot.nominal_input(goal_rel, theta) 
            # u_nom = u_nom.flatten()
            # objective = cp.Minimize(cp.sum_squares(u - u_nom)) # (v-v_nom)^2 + (w - w_nom)^2

            u_nom = self.robot.nominal_input_SI(goal_rel, theta) 
            u_nom = u_nom.flatten()
            objective = cp.Minimize(cp.sum_squares(J @ u - u_nom)) # (vx_L - vx_nom)^2 + (vy_L - vy_nom)^2

            # One lookahead-CBF constraint per observed obstacle.
            for i in range(obstacle_rels.shape[0]):
                human_rel = obstacle_rels[i]
                human_vel_curr = obstacle_vels[i]
                m = float(obstacle_masks[i])
                human_vel_pred = self.predictor.predict_vel_expectation(human_vel_curr)

                # Lookahead Position Relative to Human
                # p_L = p + eps*[c, s]
                # p_L_rel = (p - h) + eps*[c, s]
                p_L_rel = human_rel + epsilon * np.array([c, s])
                # CBF on Lookahead point
                # Constraint: 2*p_L_rel^T * J * u >= -alpha*h + 2*p_L_rel^T * v_hum
                h_val = np.dot(p_L_rel, p_L_rel) - R**2
                lhs = m * (2 * p_L_rel @ (J @ u))
                rhs = m * (-self.alpha * h_val + 2 * np.dot(p_L_rel, human_vel_pred))
                constraints.append(lhs >= rhs)
            
            # Input Limits
            v_max, w_max = self.umax
            constraints += [u[0] <= v_max, u[0] >= 0]     # v in [0, vmax] usually, or [-vmax, vmax]
            constraints += [u[1] <= w_max, u[1] >= -w_max] # w limits
            
            self._warm_start_var(u, self._u_prev)
            prob = cp.Problem(objective, constraints)
            result = self._solve_qp(prob)
            if prob.status not in ["infeasible", "unbounded", None] and u.value is not None:
                u_sol = np.asarray(u.value, dtype=np.float64).reshape(-1)
                self._u_prev = u_sol
                return u_sol.astype(np.float32)
            else:
                self.infeasible = True
                return self._fallback_action(dim=2)


        elif self.robot_type == 'unicycle_dynamic':
            return np.zeros(2, dtype=np.float32)
 
