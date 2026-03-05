import numpy as np
import cvxpy as cp
from controller.traj_prediction import TrajPredictor
from scipy.stats import norm

from crowd_sim.utils import absolute_obs_to_relative, parse_obstacles

# helper
def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
    
class CVaRBFQPController:
    def __init__(self, config_file, env=None):
        # Load params from Config
        robot_params = config_file.robot
        human_params = config_file.human
        ctrl_params = config_file.controller

        self.env = env
        self.robot = env.robot
        self.robot_type = env.robot_type


        self.robot_radius = robot_params['radius']
        self.human_radius = human_params['radius']
        self.safe_dist = ctrl_params['safety_margin'] + self.robot_radius + self.human_radius  # Use discomfort_dist as safe_dist
        self.alpha = ctrl_params['cbf_alpha']
        # self.beta = ctrl_params.get('cvar_beta', 0.1) # 1.0 = no risk aversion (default 0.1)
        self.beta = ctrl_params['cvar_beta'] # 1.0 = no risk aversion (default 0.1)
        self._base_safe_dist = float(self.safe_dist)
        self._base_alpha = float(self.alpha)
        self._base_beta = float(self.beta)
        
        # Handle umax depending on robot type
        if self.robot_type == 'single_integrator':
            self.umax = robot_params['vmax'] # For single integrator, u is v
        elif self.robot_type == 'unicycle':
            self.umax = [robot_params['vmax'], robot_params['omega_max']]
        elif self.robot_type == 'unicycle_dynamic':
            self.umax = [robot_params['vmax'], robot_params['amax'], robot_params['omega_max']]

        gmm_params = dict(human_params.get("gmm", {}))
        self.predictor = TrajPredictor(gmm_params=gmm_params)
        # Warm-start buffer: reuse previous QP solution as initialization.
        self._u_prev = None
        self.infeasible = False
        print(
            f"[DRCVaRBFQPController] robot_type={self.robot_type}, safe_dist={self.safe_dist:.3f}, alpha={self.alpha}, beta={self.beta}, umax={self.umax}",
            flush=True,
        )

    def set_adaptive_params(self, alpha=None, beta=None, safe_dist=None):
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            # Beta must be in (0, 1)
            self.beta = float(np.clip(beta, 1e-4, 1.0 - 1e-4))
        if safe_dist is not None:
            self.safe_dist = float(safe_dist)

    def reset_adaptive_params(self):
        self.alpha = float(self._base_alpha)
        self.beta = float(self._base_beta)
        self.safe_dist = float(self._base_safe_dist)

    def _warm_start_var(self, var, prev_u):
        if prev_u is None:
            return
        var.value = np.asarray(prev_u, dtype=float).reshape(-1)

    def _solve_qp(self, prob):
        try:
            return prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            return prob.solve(warm_start=True)

    def _fallback_action(self, dim=2):
        if self._u_prev is not None:
            return self._u_prev.astype(np.float32).copy()
        return np.zeros(dim, dtype=np.float32)

    def compute_drcvar_component_risk(self, p_rel, mu_v, sigma_sq, scaling=1.0):
        """
        Computes the Risk term (Cost) for one Gaussian component using DR-CVaR (or Standard Risk Metric).
        Z = -2 * p_rel^T * v_human
        We generally want: LHS >= -alpha*h - Risk
        """
        
        # CVaR Coefficient: phi(Phi^-1(1-beta)) / beta
        # We compute this once or here
        inv_cdf = norm.ppf(1 - self.beta)
        pdf_val = norm.pdf(inv_cdf)
        coeff = pdf_val / self.beta
        
        # mu_z = -2 * p^T * mu_v
        mu_z = -2 * scaling * np.dot(p_rel, mu_v)
        
        # var_z = 4 * p^T * Sigma * p
        # Sigma = sigma_sq * I
        var_z = 4 * (scaling ** 2) * sigma_sq * np.dot(p_rel, p_rel)
        sigma_z = np.sqrt(var_z) if var_z > 1e-8 else 1e-8
        
        expected_tail_val = mu_z - sigma_z * coeff
        return expected_tail_val


    def get_action(self, obs):
        # 1. Parse Observation
        obs = absolute_obs_to_relative(obs)
        self.infeasible = False

        goal_rel = np.asarray(obs[0:2], dtype=np.float64)
        theta = float(np.float64(obs[4]))
        obstacle_rels, obstacle_vels, obstacle_radii, obstacle_masks = parse_obstacles(obs)
        active_idx = np.where(obstacle_masks > 0.5)[0]

        if self.robot_type == "single_integrator":
            # --- Single Integrator DR-CVaR ---
            R = self.safe_dist

            u = cp.Variable(2)
            constraints = []

            u_nom = self.robot.nominal_input(goal_rel, theta) if (self.robot and hasattr(self.robot, 'nominal_input')) else np.zeros(2)
            u_nom = u_nom.flatten()
            objective = cp.Minimize(cp.sum_squares(u - u_nom))

            # One robust-constraint set per obstacle block (mask-aware).
            for k in range(obstacle_rels.shape[0]):
                human_rel = obstacle_rels[k]
                human_vel_curr = obstacle_vels[k]
                m = float(obstacle_masks[k])
                weights, means, variances = self.predictor.predict_gmm(human_vel_curr)

                h_val = np.dot(human_rel, human_rel) - R**2
                lhs = m * (2 * human_rel @ u)
                for i in range(len(weights)):
                    risk_val = self.compute_drcvar_component_risk(human_rel, means[i], variances[i])
                    rhs = m * (-self.alpha * h_val - risk_val)
                    constraints.append(lhs >= rhs)
                
            v_limit = self.umax if np.isscalar(self.umax) else self.umax[0]
            constraints.append(u[0] <= v_limit)
            constraints.append(u[0] >= -v_limit)
            constraints.append(u[1] <= v_limit)
            constraints.append(u[1] >= -v_limit)
            
            self._warm_start_var(u, self._u_prev)
            prob = cp.Problem(objective, constraints)

            self._solve_qp(prob)
            if prob.status not in ["infeasible", "unbounded", None] and u.value is not None:
                u_sol = np.asarray(u.value, dtype=np.float64).reshape(-1)
                self._u_prev = u_sol
                return u_sol.astype(np.float32)
            else:
                self.infeasible = True
                return self._fallback_action(dim=2)

        elif self.robot_type == "unicycle":
            # --- Unicycle Lookahead DR-CVaR ---
            u = cp.Variable(2) # [v, omega]
            constraints = []

            epsilon = 0.2
            R = self.safe_dist + epsilon
            
            c, s = np.cos(theta), np.sin(theta)
            J = np.array([[c, -epsilon*s],
                          [s, epsilon*c]])
            
            # u_nom = self.robot.nominal_input(goal_rel, theta) if hasattr(self.robot, 'nominal_input') else np.zeros(2)
            # u_nom = u_nom.flatten()
            # objective = cp.Minimize(cp.sum_squares(u - u_nom)) # (v-v_nom)^2 + (w - w_nom)^2
            
            u_nom = self.robot.nominal_input_SI(goal_rel, theta) if hasattr(self.robot, 'nominal_input_SI') else np.zeros(2)
            u_nom = u_nom.flatten()
            objective = cp.Minimize(cp.sum_squares(J @ u - u_nom)) # (vx_L - vx_nom)^2 + (vy_L - vy_nom)^2
            
            # One robust lookahead-constraint set per obstacle block (mask-aware).
            for k in range(obstacle_rels.shape[0]):
                human_rel = obstacle_rels[k]
                human_vel_curr = obstacle_vels[k]
                m = float(obstacle_masks[k])
                p_L_rel = human_rel + epsilon * np.array([c, s])
                weights, means, variances = self.predictor.predict_gmm(human_vel_curr)

                h_val = np.dot(p_L_rel, p_L_rel) - R**2
                lhs = m * (2 * p_L_rel @ (J @ u))
                for i in range(len(weights)):
                    risk_val = self.compute_drcvar_component_risk(p_L_rel, means[i], variances[i])
                    rhs = m * (-self.alpha * h_val - risk_val)
                    constraints.append(lhs >= rhs)
                
            self._warm_start_var(u, self._u_prev)
            prob = cp.Problem(objective, constraints)
            self._solve_qp(prob)
            if prob.status not in ["infeasible", "unbounded", None] and u.value is not None:
                u_sol = np.asarray(u.value, dtype=np.float64).reshape(-1)
                self._u_prev = u_sol
                return u_sol.astype(np.float32)
            else:
                self.infeasible = True
                return self._fallback_action(dim=2)

        elif self.robot_type == "unicycle_dynamic":
            return np.zeros(2, dtype=np.float32)
