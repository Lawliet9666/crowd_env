import numpy as np
import cvxpy as cp
from scipy.optimize import brentq
from scipy.stats import norm

from crowd_sim.utils import absolute_obs_to_relative, parse_obstacles
from controller.cvarbf_qp import CVaRBFQPController


class AdaptiveCVaRBFQPController(CVaRBFQPController):
    """
    Keep the same CVaR-QP logic as CVaRBFQPController, except beta is searched
    adaptively per step in [0, base_beta] with step 0.05.
    Supports single_integrator and unicycle.
    """

    def __init__(self, config_file, env=None):
        super().__init__(config_file=config_file, env=env)
        if self.robot_type not in {"single_integrator", "unicycle"}:
            raise ValueError(
                "AdaptiveCVaRBFQPController supports only single_integrator/unicycle, "
                f"got: {self.robot_type}"
            )
        self.beta =0.99
        raw = np.arange(0.0, float(self.beta) + 1e-9, 0.02).tolist()
        raw.append(float(self.beta))
        # Search beta from aggressive to conservative: base_beta -> 0.
        self.beta_candidates = sorted(
            set(round(float(np.clip(b, 0.0, float(self.beta))), 4) for b in raw),
            reverse=True, # False means search from high to low (conservative to aggressive)
        )
        self.last_feasible_beta = float(self.beta)
        self.last_selected_beta = float(self.beta)
        self._safe_dist_dyn = float(self._base_safe_dist)
        self._safe_dist_smoothing = 0.3
        self.last_collision_cone_risk = 0.0
        self.last_selected_safe_dist = float(self._base_safe_dist)

        print(
            f"[AdaptiveCVaRBFQPController] robot_type={self.robot_type}, beta_max={self.beta:.3f}, candidates={self.beta_candidates}",
            flush=True,
        )
        self.infeasible = False

    @staticmethod
    def _clip_beta(beta):
        return float(np.clip(beta, 1e-4, 0.99))

    def _compute_cvar_term_with_beta(self, p_rel, weights, means, variances, beta):
        beta = self._clip_beta(beta)
        p_rel_sq = np.dot(p_rel, p_rel)
        m_k = []
        s_k = []

        for i in range(len(weights)):
            mu_val = -2.0 * np.dot(p_rel, means[i])
            var_val = 4.0 * variances[i] * p_rel_sq
            s_val = np.sqrt(var_val) if var_val > 1e-8 else 1e-8
            m_k.append(mu_val)
            s_k.append(s_val)

        m_k = np.asarray(m_k, dtype=np.float64)
        s_k = np.asarray(s_k, dtype=np.float64)

        def tail_prob_func(q):
            z = (q - m_k) / s_k
            return np.sum(weights * norm.cdf(z)) - beta

        min_q = np.min(m_k - 4.0 * s_k)
        max_q = np.max(m_k + 4.0 * s_k)
        try:
            q_star = brentq(tail_prob_func, min_q, max_q)
        except ValueError:
            q_star = min_q if tail_prob_func(min_q) > 0 else max_q

        z_star = (q_star - m_k) / s_k
        beta_k = norm.cdf(z_star)
        pdf_k = norm.pdf(z_star)
        return np.sum(weights * (beta_k * m_k - s_k * pdf_k)) / beta

    def _compute_collision_cone_risk(self, obs):
        """
        Compute a per-step collision-cone risk in [0, 1] from the current
        observation. Higher means more likely head-on conflict.
        """
        obs_rel = absolute_obs_to_relative(obs)
        robot_vel = np.asarray(obs_rel[2:4], dtype=np.float64).reshape(-1)
        obstacle_rels, obstacle_vels, _, obstacle_masks = parse_obstacles(obs_rel)

        base_R = max(float(self._base_safe_dist), 1e-6)
        max_risk = 0.0

        for i in range(obstacle_rels.shape[0]):
            m = float(obstacle_masks[i])
            if m <= 0.5:
                continue

            p_rel = np.asarray(obstacle_rels[i], dtype=np.float64).reshape(-1)
            p_sq = float(np.dot(p_rel, p_rel))
            if p_sq <= base_R * base_R:
                return 1.0

            human_vel_curr = np.asarray(obstacle_vels[i], dtype=np.float64).reshape(-1)
            human_vel_pred = self.predictor.predict_vel_expectation(human_vel_curr)
            human_vel_pred = np.asarray(human_vel_pred, dtype=np.float64).reshape(-1)
            if human_vel_pred.size < 2 or robot_vel.size < 2:
                continue

            v_rel = robot_vel[:2] - human_vel_pred[:2]
            v_sq = float(np.dot(v_rel, v_rel))
            if v_sq <= 1e-8:
                continue

            pv = float(np.dot(p_rel, v_rel))
            if pv >= 0.0:
                # Moving away (or roughly tangent): low immediate collision-cone risk.
                continue

            # Collision-cone discriminant-like margin:
            # > 0 implies relative velocity lies inside collision cone.
            cone_val = pv * pv - v_sq * (p_sq - base_R * base_R)
            cone_scale = np.sqrt(v_sq * (p_sq + base_R * base_R)) + 1e-8
            cone_norm = cone_val / cone_scale
            cone_arg = np.clip(6.0 * cone_norm, -50.0, 50.0)
            cone_risk = 1.0 / (1.0 + np.exp(-cone_arg))

            closing = np.clip(-pv / (np.sqrt(p_sq * v_sq) + 1e-8), 0.0, 1.0)
            dist = np.sqrt(p_sq)
            dist_gate = np.clip((2.5 * base_R) / (dist + 1e-8), 0.0, 1.0)

            risk_i = float(np.clip(cone_risk * closing * dist_gate, 0.0, 1.0))
            if risk_i > max_risk:
                max_risk = risk_i

        return float(np.clip(max_risk, 0.0, 1.0))

    def _update_adaptive_safe_dist(self, obs):
        """
        Map collision-cone risk to safe distance multiplier in [1.0, 1.5].
        """
        current_step = int(getattr(self.env, "current_step", 0)) if self.env is not None else -1
        if current_step == 0 or self._u_prev is None:
            self._safe_dist_dyn = float(self._base_safe_dist)

        risk = self._compute_collision_cone_risk(obs)
        target_scale = 1.0 + 0.5 * risk
        target_safe_dist = float(self._base_safe_dist) * target_scale

        smooth = float(np.clip(self._safe_dist_smoothing, 0.0, 1.0))
        self._safe_dist_dyn = (1.0 - smooth) * float(self._safe_dist_dyn) + smooth * target_safe_dist
        self._safe_dist_dyn = float(
            np.clip(self._safe_dist_dyn, float(self._base_safe_dist), 1.5 * float(self._base_safe_dist))
        )

        self.safe_dist = self._safe_dist_dyn
        self.last_collision_cone_risk = float(risk)
        self.last_selected_safe_dist = float(self._safe_dist_dyn)

    def _solve_unicycle_for_beta(self, obs, beta):
        obs = absolute_obs_to_relative(obs)
        goal_rel = np.asarray(obs[0:2], dtype=np.float64)
        theta = float(np.float64(obs[4]))
        obstacle_rels, obstacle_vels, obstacle_radii, obstacle_masks = parse_obstacles(obs)

        u = cp.Variable(2)  # [v, omega]

        epsilon = 0.2
        R = self.safe_dist + epsilon
        c, s = np.cos(theta), np.sin(theta)
        J = np.array([[c, -epsilon * s], [s, epsilon * c]], dtype=np.float64)

        u_nom = (
            self.robot.nominal_input_SI(goal_rel, theta)
            if hasattr(self.robot, "nominal_input_SI")
            else np.zeros(2)
        )
        u_nom = np.asarray(u_nom, dtype=np.float64).flatten()
        objective = cp.Minimize(cp.sum_squares(J @ u - u_nom))

        constraints = []
        for i in range(obstacle_rels.shape[0]):
            human_rel = obstacle_rels[i]
            human_vel_curr = obstacle_vels[i]
            m = float(obstacle_masks[i])
            p_L_rel = human_rel + epsilon * np.array([c, s], dtype=np.float64)

            weights, means, variances = self.predictor.predict_gmm(human_vel_curr)
            risk_term = self._compute_cvar_term_with_beta(
                p_rel=p_L_rel,
                weights=weights,
                means=means,
                variances=variances,
                beta=beta,
            )

            h_val = np.dot(p_L_rel, p_L_rel) - R ** 2
            lhs = m * (2.0 * p_L_rel @ (J @ u))
            rhs = m * (-self.alpha * h_val - risk_term)
            constraints.append(lhs >= rhs)

        # Keep exactly the same input limits as cvarbf_qp.py (unicycle branch).
        v_max, w_max = self.umax
        constraints += [u[0] <= v_max, u[0] >= 0.0]
        constraints += [u[1] <= w_max, u[1] >= -w_max]

        self._warm_start_var(u, self._u_prev)
        prob = cp.Problem(objective, constraints)
        self._solve_qp(prob)

        if prob.status not in ["infeasible", "unbounded", None] and u.value is not None:
            return np.asarray(u.value, dtype=np.float64).reshape(-1)
        return None

    def _solve_single_integrator_for_beta(self, obs, beta):
        obs = absolute_obs_to_relative(obs)
        goal_rel = np.asarray(obs[0:2], dtype=np.float64)
        theta = float(np.float64(obs[4]))
        obstacle_rels, obstacle_vels, obstacle_radii, obstacle_masks = parse_obstacles(obs)

        R = self.safe_dist
        u = cp.Variable(2)

        u_nom = (
            self.robot.nominal_input(goal_rel, theta)
            if hasattr(self.robot, "nominal_input")
            else np.zeros(2)
        )
        u_nom = np.asarray(u_nom, dtype=np.float64).flatten()
        objective = cp.Minimize(cp.sum_squares(u - u_nom))

        constraints = []
        for i in range(obstacle_rels.shape[0]):
            human_rel = obstacle_rels[i]
            human_vel_curr = obstacle_vels[i]
            m = float(obstacle_masks[i])

            weights, means, variances = self.predictor.predict_gmm(human_vel_curr)
            risk_term = self._compute_cvar_term_with_beta(
                p_rel=human_rel,
                weights=weights,
                means=means,
                variances=variances,
                beta=beta,
            )

            h_val = np.dot(human_rel, human_rel) - R ** 2
            lhs = m * (2.0 * human_rel @ u)
            rhs = m * (-self.alpha * h_val - risk_term)
            constraints.append(lhs >= rhs)

        v_limit = self.umax if np.isscalar(self.umax) else self.umax[0]
        constraints += [u[0] <= v_limit, u[0] >= -v_limit]
        constraints += [u[1] <= v_limit, u[1] >= -v_limit]

        self._warm_start_var(u, self._u_prev)
        prob = cp.Problem(objective, constraints)
        self._solve_qp(prob)

        if prob.status not in ["infeasible", "unbounded", None] and u.value is not None:
            return np.asarray(u.value, dtype=np.float64).reshape(-1)
        return None

    def get_action(self, obs):
        self.infeasible = False
        self._update_adaptive_safe_dist(obs)
        for beta in self.beta_candidates:
            if self.robot_type == "single_integrator":
                u_sol = self._solve_single_integrator_for_beta(obs, beta)
            elif self.robot_type == "unicycle":
                u_sol = self._solve_unicycle_for_beta(obs, beta)
            else:
                u_sol = None
            if u_sol is not None:
                beta_sol = self._clip_beta(beta)
                self.beta = beta_sol
                self.last_feasible_beta = beta_sol
                self.last_selected_beta = beta_sol
                self._u_prev = u_sol
                return u_sol.astype(np.float32)

        self.last_selected_beta = None
        self.infeasible = True
        return self._fallback_action(dim=2)
