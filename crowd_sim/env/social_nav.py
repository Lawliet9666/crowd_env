import gymnasium as gym
from gymnasium import spaces
import numpy as np
from crowd_sim.env.robot.robot import SingleIntegrator, Unicycle, UnicycleDynamic
from crowd_sim.env.robot.obstacle import SingleIntegrator as HumanIntegrator
from crowd_nav.policy.social_force_helper import SocialForceHelper
from crowd_nav.policy.potential_field_helper import PotentialFieldHelper
from crowd_sim.utils import sample_point_in_disk


class SocialNav(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, config_file=None):
        super(SocialNav, self).__init__()
        self.render_mode = render_mode
        
        # Load params from Config
        env_params = config_file.env
        self.dt = env_params['dt']
        self.max_steps = env_params['max_steps']
        self.sensing_radius = float(env_params.get('sensing_radius', 5.0))
        self.max_obstacles_obs = int(env_params.get('max_obstacles_obs', 10))
        self.normalize_obs = bool(env_params.get('normalize_obs', False))
        self.rl_xy_to_unicycle = bool(env_params.get('rl_xy_to_unicycle', False))
        self.unicycle_k_omega = float(env_params.get('unicycle_k_omega', 2.0))

        # --- 0. Initialize Robot Model ---
        robot_params = config_file.robot
        self.robot_radius = robot_params['radius']
        self.robot_pos = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.robot_theta = 0.0
        self.goal_pos = np.zeros(2)
        self.robot_traj = []

        self.robot_type = robot_params['type']
        if self.robot_type == 'single_integrator':
            robot_u_max = robot_params['vmax']
            self.robot = SingleIntegrator(self.dt, self.robot_radius, umax=robot_u_max)
        elif self.robot_type == 'unicycle':
            # input is v and omega, so robot_u_max is 2 dimensions: [v_max, w_max]
            robot_u_max = [robot_params['vmax'], robot_params['omega_max']]
            self.robot = Unicycle(self.dt, self.robot_radius, umax=robot_u_max)
        elif self.robot_type == 'unicycle_dynamic':
            # input is acc and omega, so robot_u_max is: [v_max, w_max, acc_max]
            robot_u_max = [robot_params['vmax'],  robot_params['omega_max'], robot_params['amax']]
            self.robot = UnicycleDynamic(self.dt, self.robot_radius, umax=robot_u_max)
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")
            

        # --- 1. Initialize Humans ---
        human_params = config_file.human
        self.num_humans = int(human_params.get('num_humans', 1))

        self.human_radii = np.zeros(self.num_humans, dtype=float)  
        self.human_positions = np.zeros((self.num_humans, 2), dtype=float)
        self.human_vels = np.zeros((self.num_humans, 2), dtype=float)
        self.human_goals = np.zeros((self.num_humans, 2), dtype=float)
        self.human_trajs = [[] for _ in range(self.num_humans)]
        self.human_traj_steps = []
        self.human_vmaxs = np.zeros(self.num_humans, dtype=float) 

        self.human_vmax_min, self.human_vmax = map(float, human_params["vmax"])
        # self.human_radius_min, self.human_radius_max = map(float, human_params["radius"])
        self.human_radius_min = float(human_params["radius"])
        self.human_radius_max = float(human_params["radius"])

        self.human_gmm_params = dict(human_params.get("gmm", {}))
        self.humans = [
            HumanIntegrator(self.dt, gmm_params=self.human_gmm_params)
            for _ in range(self.num_humans)
        ]
    
        self.arena_size = human_params.get('arena_size', 6.0)
        self.human_circle_radius = self.arena_size * np.sqrt(2) 

        self.human_policy_name = human_params.get('policy', 'nominal')
        self.human_use_gmm = bool(human_params.get('use_gmm', True))
        self.random_goal_changing = bool(human_params.get('random_goal_changing', False))
        self.goal_change_chance = float(np.clip(human_params.get('goal_change_chance', 0.0), 0.0, 1.0))
        self.end_goal_changing = bool(human_params.get('end_goal_changing', False))
        self.end_goal_change_chance = float(np.clip(human_params.get('end_goal_change_chance', 1.0), 0.0, 1.0))
        self.current_scenario = None

        self.orca_params = human_params.get('orca', {})
        self.sf_params = human_params.get('sf', {})
        self.pf_params = human_params.get('pf', {})
        self.orca_helper = None
        self.sf_helper = None
        self.pf_helper = None
        if self.human_policy_name == 'orca':
            try:
                from crowd_nav.policy.orca_helper import ORCAHelper
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "ORCA policy requested but optional dependency is missing. "
                    "Install Python-RVO2 (rvo2) to use human.policy='orca'."
                ) from exc
            self.orca_helper = ORCAHelper(
                dt=self.dt,
                orca_params=self.orca_params,
                max_humans=self.num_humans,
            )
        elif self.human_policy_name == 'social_force':
            self.sf_helper = SocialForceHelper(
                dt=self.dt,
                sf_params=self.sf_params,
                max_humans=self.num_humans,
            )
        elif self.human_policy_name == 'potential_field':
            self.pf_helper = PotentialFieldHelper(
                dt=self.dt,
                pf_params=self.pf_params,
                max_humans=self.num_humans,
            )

        # --- Base Observation (absolute protocol) ---
        # Robot+goal block: [rx, ry, gx, gy, rvx, rvy, theta, r_radius] -> 8 dims
        # Local sensing block (per obstacle): [hx, hy, hvx, hvy, hradius, mask] -> 6 dims
        # Total: 8 + K * 6
        self.obs_dim = 8 + self.max_obstacles_obs * 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = self.robot.action_space
        if self.robot_type == 'unicycle' and self.rl_xy_to_unicycle:
            vmax = float(self.robot.vmax)
            self.action_space = spaces.Box(
                low=np.array([-vmax, -vmax], dtype=np.float32),
                high=np.array([vmax, vmax], dtype=np.float32),
                dtype=np.float32,
            )

        # --- Reward Parameters (CrowdNav++) ---
        self.current_step = 0
        rew_params = config_file.reward
        self.success_reward = rew_params['success_reward']
        self.collision_penalty = rew_params['collision_penalty']
        self.discomfort_dist = rew_params['discomfort_dist']
        self.discomfort_penalty_factor = rew_params['discomfort_penalty_factor']
        self.potential_factor = rew_params['potential_factor']
        self.back_factor = rew_params['back_factor']
        self.spin_factor = rew_params['spin_factor']
        self.constant_penalty = rew_params['constant_penalty']
        # Optional smooth safety shaping near humans (disabled by default).
        self.safe_shaping_weight = rew_params.get('safe_shaping_weight', 0.0)
        self.safe_shaping_band = rew_params.get('safe_shaping_band', 0.6)

        self.max_abs_pot_reward = self.dt * self.robot.vmax * self.potential_factor
        # maximum absolute value of rotation penalty (only meaningful for unicycle robots)
        if self.robot_type == 'unicycle':
            self.max_abs_rot_penalty = self.spin_factor * (max(abs(self.robot.w_min), self.robot.w_max) * self.dt) ** 2
        else:
            self.max_abs_rot_penalty = 0.0
        self.max_abs_back_penalty = self.back_factor * max(abs(self.robot.vmin), self.robot.vmax)

    def reset(self, seed=None, options=None):
        # 1. Basic Environment Reset & Seeding
        super().reset(seed=seed)
        self.current_step = 0
        self.current_scenario = options.get("scenario") if isinstance(options, dict) else options
        
        if self.orca_helper is not None:
            self.orca_helper.reset()
        if self.pf_helper is not None:
            self.pf_helper.reset()
            
        # 2. Initialize Trajectory Storage
        self.robot_traj = []
        self.human_trajs = [[] for _ in range(self.num_humans)]
        self.human_traj_steps = []
        
        # 3. Initialize Robot State (Position, Goal, Theta)
        (
            self.robot_pos,
            self.goal_pos,
            self.robot_theta,
            self.robot_vel,
            self.human_positions,
            self.human_goals,
            self.human_vels,
            self.human_vmaxs,
            self.human_radii,
        ) = self._init_robot_humans(options=options)
        self._seed_social_force_model()
        if self.human_use_gmm:
            self._seed_human_noise_models()

        # 4. Reset Internal Dynamics Models
        # Robot: [x, y, theta, v]
        self.robot.reset([self.robot_pos[0], self.robot_pos[1], self.robot_theta, 0.0])
        # Humans
        for i, human in enumerate(self.humans):
            human.vmax = self.human_vmaxs[i]
            human.radius = self.human_radii[i]
            human.reset(self.human_positions[i])
        
        # 6. Initialize Metrics & History
        self.prev_dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        self.robot_traj.append(self.robot_pos.copy())
        self.human_traj_steps.append(self.human_positions.copy())
            
        return self._get_obs(), {}
    
    def _seed_human_noise_models(self, seed=None):
        max_seed = np.iinfo(np.uint32).max
        if seed is not None:
            base_seed = int(seed) % max_seed
        else:
            base_seed = int(self.np_random.integers(0, max_seed))
        for i, human in enumerate(self.humans):
            human.gmm.set_seed((base_seed + i) % max_seed)

    def _seed_social_force_model(self, seed=None):
        if self.sf_helper is None:
            return

        max_seed = np.iinfo(np.uint32).max
        if seed is not None:
            sf_seed = int(seed) % max_seed
        else:
            sf_seed = int(self.np_random.integers(0, max_seed))
        self.sf_helper.set_seed(sf_seed)

    def step(self, action):
        # 1. robot position update
        self._update_robot_state(action)

        # 2. human position update (with optional goal changing)
        self._update_human_states()

        # 3. Hard wall clamp: keep robot/humans inside [-half_extent, half_extent].
        self._clip_positions_to_wall()
        
        # 4. calculate distances
        dist_to_goal, min_dist = self._compute_distances(self.human_positions)
        
        # 5. claculate reward and done
        reward, done, info = self._compute_reward_and_done(dist_to_goal, min_dist, self.robot.u)
            
        return self._get_obs(), reward, done, False, info

    def _get_wall_half_extent(self):
        return float(self.human_circle_radius + getattr(self, "human_init_noise_range", 0.0))

    def _clip_positions_to_wall(self):
        half_extent = self._get_wall_half_extent()
        lo, hi = -half_extent, half_extent

        np.clip(self.robot_pos, lo, hi, out=self.robot_pos)
        np.clip(self.human_positions, lo, hi, out=self.human_positions)

        # Keep internal robot state and trajectory buffers aligned with clipped positions.
        if hasattr(self, "robot_state") and self.robot_state is not None:
            self.robot_state[0:2] = self.robot_pos
        if self.robot_traj:
            self.robot_traj[-1] = self.robot_pos.copy()
        if self.human_traj_steps:
            self.human_traj_steps[-1] = self.human_positions.copy()

    def _init_robot_humans(self, options=None):
        if options and options.get("scenario") == "crossing":
            robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii = \
            self.crossing_init_robot_humans()
        else:
            # Robot starts near left-bottom corner and goals near right-top corner.
            robot_pos = sample_point_in_disk(
                self.np_random, center=[-5.5, -5.5], radius=1.0, arena_size=self.arena_size
            )
            goal_pos = sample_point_in_disk(
                self.np_random, center=[5.5, 5.5], radius=1.0, arena_size=self.arena_size
            )
            robot_theta = self.np_random.uniform(-np.pi, np.pi)

            min_pair_dist = 6.0
            human_positions = np.zeros((self.num_humans, 2), dtype=float)
            human_goals = np.zeros((self.num_humans, 2), dtype=float)

            safe_dist_init = self.robot_radius + self.human_radius_max + self.discomfort_dist

            for i in range(self.num_humans):
                found = False
                for _ in range(50):
                    p = sample_point_in_disk(
                        self.np_random, center=[0.0, 0.0], radius=self.human_circle_radius, arena_size=self.arena_size
                    )
                    g = sample_point_in_disk(
                        self.np_random, center=[0.0, 0.0], radius=self.human_circle_radius, arena_size=self.arena_size
                    )
                    if np.linalg.norm(p - g) >= min_pair_dist and np.linalg.norm(robot_pos - p) >= safe_dist_init and np.linalg.norm(goal_pos - g) >= safe_dist_init:
                        human_positions[i] = p
                        human_goals[i] = g
                        found = True
                        break

                if not found:
                    angle = self.np_random.uniform(0.0, 2.0 * np.pi)
                    p = np.array([
                        self.human_circle_radius * np.cos(angle),
                        self.human_circle_radius * np.sin(angle),
                    ], dtype=float)
                    human_positions[i] = p
                    human_goals[i] = -p

            human_vmaxs = self.np_random.uniform(self.human_vmax_min, self.human_vmax, size=self.num_humans)
            human_radii = self.np_random.uniform(self.human_radius_min, self.human_radius_max, size=self.num_humans)
            human_vels = np.zeros((self.num_humans, 2), dtype=float)
            robot_vel = np.zeros(2)

        return robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii

    def _update_human_states(self):
        # optional human goal update
        self._update_obstacle_goals()

        if self.human_policy_name == 'orca':
            nominal_actions = self.orca_helper.action_for_humans(
                human_positions=self.human_positions,
                human_vels=self.human_vels,
                human_goals=self.human_goals,
                human_radii=self.human_radii,
                human_vprefs=self.human_vmaxs,
                robot_pos=self.robot_pos,
                robot_vel=self.robot_vel,
                robot_radius=self.robot_radius,
                robot_vpref=float(getattr(self.robot, "vmax", 1.0)),
            )
        elif self.human_policy_name == 'social_force':
            nominal_actions = self.sf_helper.action_for_humans(
                human_positions=self.human_positions,
                human_vels=self.human_vels,
                human_goals=self.human_goals,
                human_radii=self.human_radii,
                human_vprefs=self.human_vmaxs,
                robot_pos=self.robot_pos,
                robot_vel=self.robot_vel,
                robot_radius=self.robot_radius,
                robot_vpref=float(getattr(self.robot, "vmax", 1.0)),
            )
        elif self.human_policy_name == 'potential_field':
            nominal_actions = self.pf_helper.action_for_humans(
                human_positions=self.human_positions,
                human_vels=self.human_vels,
                human_goals=self.human_goals,
                human_radii=self.human_radii,
                human_vprefs=self.human_vmaxs,
                robot_pos=self.robot_pos,
                robot_vel=self.robot_vel,
                robot_radius=self.robot_radius,
                robot_vpref=float(getattr(self.robot, "vmax", 1.0)),
            )
        else:
            nominal_actions = np.zeros((self.num_humans, 2), dtype=np.float32)
            for i, human in enumerate(self.humans):
                nominal_actions[i] = human.nominal_controller(self.human_positions[i], self.human_goals[i])

        if self.human_use_gmm:
            exec_actions = HumanIntegrator.apply_gmm_batch(
                humans=self.humans,
                nominal_actions=nominal_actions,
                states=self.human_positions,
                goals=self.human_goals,
                radii=self.human_radii,
            )
        else:
            exec_actions = np.asarray(nominal_actions, dtype=np.float32)

        # 3. Human state update (vectorized).
        self.human_vels, self.human_positions = HumanIntegrator.step_batch(
            actions=exec_actions,
            vmaxs=self.human_vmaxs,
            positions=self.human_positions,
            dt=self.dt,
        )
        self.human_traj_steps.append(self.human_positions.copy())

    def _update_robot_state(self, action):
        if self.robot_type == 'unicycle' and self.rl_xy_to_unicycle:
            action_to_robot = self._xy_to_unicycle_action(action) # convert [vx, vy] to [v, omega]
        else:
            action_to_robot = action # v,w for unicycle, vx,vy for single integrator

        self.robot.step(action_to_robot)

        self.robot_state = self.robot.get_state()
        if self.robot_type == 'single_integrator':
            self.robot_pos = self.robot_state
            self.robot_vel = self.robot.u
            self.robot_theta = 0.0
        elif self.robot_type == 'unicycle':
            self.robot_pos = self.robot_state[0:2]
            self.robot_vel = np.array([
                self.robot.u[0] * np.cos(self.robot_state[2]),
                self.robot.u[0] * np.sin(self.robot_state[2])
            ])
            self.robot_theta = self.robot_state[2]
        elif self.robot_type == 'unicycle_dynamic':
            self.robot_pos = self.robot_state[0:2]
            v = self.robot_state[3]
            self.robot_vel = np.array([
                v * np.cos(self.robot_state[2]),
                v * np.sin(self.robot_state[2])
            ])
            self.robot_theta = self.robot_state[2]
        self.robot_traj.append(self.robot_pos.copy())

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _xy_to_unicycle_action(self, action):
        """
        Convert world-frame [vx, vy] command into unicycle [v, omega].
        """
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size < 2:
            return np.zeros(2, dtype=np.float32)

        vxy = np.array([float(a[0]), float(a[1])], dtype=float)
        speed_xy = np.linalg.norm(vxy)
        vmax = float(self.robot.vmax)
        if speed_xy > vmax and speed_xy > 1e-8:
            vxy = vxy / speed_xy * vmax
            speed_xy = vmax

        theta = float(self.robot_theta)
        vx_cmd, vy_cmd = float(vxy[0]), float(vxy[1])

        # Project desired XY velocity on robot heading to get linear speed.
        v = vx_cmd * np.cos(theta) + vy_cmd * np.sin(theta)

        if speed_xy <= 1e-8:
            omega = 0.0
        else:
            theta_des = np.arctan2(vy_cmd, vx_cmd)
            theta_err = self._wrap_angle(theta_des - theta)
            omega = self.unicycle_k_omega * theta_err

        v = float(np.clip(v, self.robot.vmin, self.robot.vmax))
        omega = float(np.clip(omega, self.robot.w_min, self.robot.w_max))
        return np.array([v, omega], dtype=np.float32)

    def _compute_distances(self, human_positions=None):
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        targets = human_positions if human_positions is not None else self.human_positions
        if len(targets) == 0:
            min_clearance = float('inf')
        else:
            rel = targets - self.robot_pos
            dists = np.linalg.norm(rel, axis=1)
            radii = self.human_radii if self.human_radii.size == dists.size else np.full(dists.size, self.human_radius_max)
            clearances = dists - self.robot_radius - radii
            min_clearance = float(np.min(clearances))

        return dist_to_goal, min_clearance

    def _compute_reward_and_done(self, dist_to_goal, min_dist, u=None):
        reward = 0
        done = False
        info = {"is_success": False, "is_collision": False, "is_timeout": False}
        
        if self.robot_type == 'unicycle':
            goal_radius = 0.6
        else:
            goal_radius = self.robot.radius

        self.current_step += 1
        if self.current_step >= self.max_steps and not done:
            done = True
            info["is_timeout"] = True
            reward = 0
        elif min_dist < 0:
            reward = self.collision_penalty
            done = True
            info["is_collision"] = True
        elif dist_to_goal <  goal_radius:    
            reward = self.success_reward
            done = True
            info["is_success"] = True

        elif min_dist < self.discomfort_dist:
            reward = (min_dist - self.discomfort_dist) * self.discomfort_penalty_factor * self.dt
            done = False
        else:
            potential_reward = self.potential_factor * (self.prev_dist_to_goal - dist_to_goal)
            reward = potential_reward
            reward = np.clip(reward, -self.max_abs_pot_reward, self.max_abs_pot_reward)
            done = False
            
        self.prev_dist_to_goal = dist_to_goal

        # reward = reward + self.constant_penalty
        # add a rotational penalty
        r_spin = -self.spin_factor * (u[1] * self.dt) ** 2
        # r_spin = -self.spin_factor * u[1] ** 2
        r_spin = np.clip(r_spin, -self.max_abs_rot_penalty, self.max_abs_rot_penalty)

        # add a penalty for going backwards
        if u[0] < 0:
            r_back = -self.back_factor * abs(u[0])
        else:
            r_back = 0.
        r_back = np.clip(r_back, -self.max_abs_back_penalty, self.max_abs_back_penalty)

        # r_safe = self._compute_safe_shaping_reward(min_dist) if not done else 0.0
        # reward = reward + r_safe + r_spin + r_back + self.constant_penalty
        reward = reward + r_spin + r_back + self.constant_penalty
        
        # r_safe = self._compute_safe_shaping_reward(min_dist) if not done else 0.0

        # reward = reward + r_safe + self.constant_penalty
        # scale down the reward to keep it in a reasonable range for RL training
        reward = reward / 10.0

        return reward, done, info

    def _compute_safe_shaping_reward(self, min_dist):
        """
        Smooth penalty when clearance is close to the safety boundary.
        Returns 0 when far enough; negative value as clearance shrinks.
        """
        if self.safe_shaping_weight <= 0.0:
            return 0.0

        band = max(float(self.safe_shaping_band), 1e-6)
        start = float(self.discomfort_dist)

        if min_dist >= start + band:
            return 0.0

        x = np.clip((start + band - float(min_dist)) / band, 0.0, 1.0)
        smooth = x * x * (3.0 - 2.0 * x)  # smoothstep in [0, 1]
        return -float(self.safe_shaping_weight) * float(smooth)

    def _normalize_obs(self, obs):
        """
        Scale observation features into roughly comparable ranges.
        """
        x = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        if x.size == 0:
            return x

        k = int(self.max_obstacles_obs)
        robot_v_scale = max(float(getattr(self.robot, "vmax", 1.0)), 1e-6)
        human_v_scale = max(float(self.human_vmax), 1e-6)
        pos_scale = max(float(self._get_wall_half_extent()), 1e-6)
        radius_scale = max(float(self.human_radius_max), float(self.robot_radius), 1e-6)

        # Robot+goal block: [rx, ry, gx, gy, rvx, rvy, theta, radius]
        x[0:4] /= pos_scale
        x[4:6] /= robot_v_scale
        x[6] /= np.pi
        x[7] /= radius_scale

        # Obstacle blocks: [hx, hy, hvx, hvy, radius, mask]
        if x.size >= 8 + 6 * k:
            blocks = x[8:8 + 6 * k].reshape(k, 6)
            blocks[:, 0:2] /= pos_scale
            blocks[:, 2:4] /= human_v_scale
            blocks[:, 4] /= radius_scale
            # mask stays 0/1

        return np.clip(x, -5.0, 5.0)

    def _get_obs(self):
        # --- Robot+goal absolute block ---
        robot_state = np.array([
            self.robot_pos[0], self.robot_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            self.robot_vel[0], self.robot_vel[1],
            self.robot_theta,
            self.robot_radius
        ], dtype=float)

        k = self.max_obstacles_obs
        obs_blocks = np.zeros((k, 6), dtype=float)  # dummy blocks: [0, 0, 0, 0, 0, 0]

        # Use robot-relative distances for visibility/ranking only.
        rel = self.robot_pos - self.human_positions
        dists = np.linalg.norm(rel, axis=1)
        visible_idx = np.where(dists <= self.sensing_radius)[0]
        if visible_idx.size > 0:
            order = np.argsort(dists[visible_idx])
            selected = visible_idx[order[:k]]
            rows = np.arange(selected.size)
            obs_blocks[rows, 0:2] = self.human_positions[selected]
            obs_blocks[rows, 2:4] = self.human_vels[selected]
            obs_blocks[rows, 4] = self.human_radii[selected]
            obs_blocks[rows, 5] = 1.0  # mask

        obs = np.concatenate([robot_state, obs_blocks.reshape(-1)]).astype(np.float32)
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        return obs

    def render(self):
        if self.render_mode is None:
            return
        plt = self._get_plt()

        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            if self.render_mode == "human":
                plt.ion()
                plt.show()

        self.ax.clear()
        half_extent = self._get_wall_half_extent()-3 # TODO CNN should use bound: self._get_wall_half_extent()
        self.ax.set_xlim(-half_extent, half_extent)
        self.ax.set_ylim(-half_extent, half_extent)
        self.ax.set_aspect('equal')
        
        # Draw Goal
        goal = plt.Circle(self.goal_pos, 0.2, color='blue', alpha=0.5, label='Goal')
        self.ax.add_artist(goal)
        self.ax.text(self.goal_pos[0], self.goal_pos[1], 'G', color='white', ha='center', va='center')
        
        # Draw Trajectories
        if len(self.robot_traj) > 1:
            robot_traj_arr = np.array(self.robot_traj)
            self.ax.plot(robot_traj_arr[:, 0], robot_traj_arr[:, 1], 'b-', alpha=0.5, linewidth=1)

        # Human trajectories (time-major cache: [T, N, 2]).
        if len(self.human_traj_steps) > 1:
            traj = np.asarray(self.human_traj_steps, dtype=float)
            if traj.ndim == 3 and traj.shape[1] == self.num_humans:
                for i in range(self.num_humans):
                    human_traj_arr = traj[:, i, :]
                    self.ax.plot(
                        human_traj_arr[:, 0],
                        human_traj_arr[:, 1],
                        color='red',
                        alpha=0.35,
                        linewidth=1,
                    )

        # Draw Robot
        robot = plt.Circle(self.robot_pos, self.robot_radius, color='blue', alpha=0.5, label='Robot')
        self.ax.add_artist(robot)
        
        # Draw Humans
        for i in range(self.num_humans):
            human = plt.Circle(self.human_positions[i], self.human_radii[i], color='red', alpha=0.5)
            self.ax.add_artist(human)
        
        self.ax.legend(loc='upper right')
        # self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.grid(False)
        self.ax.set_title(f"Step: {self.current_step}")

        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.01) # Slow down visualization
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            # Prefer buffer_rgba as it handles high-DPI (Retina) scaling properly by preserving shape
            if hasattr(self.fig.canvas, "buffer_rgba"):
                rgba = np.asarray(self.fig.canvas.buffer_rgba())
                return rgba[:, :, :3].copy()
            # Fallback for older backends
            if hasattr(self.fig.canvas, "tostring_rgb"):
                data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return data

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt = self._get_plt()
            plt.close(self.fig)
            self.fig = None

    @staticmethod
    def _get_plt():
        import matplotlib.pyplot as plt
        return plt

    def _update_obstacle_goals(self):
        """
        Optional stochastic goal update for humans (obstacles).
        Two independent events:
        1) random_goal_changing: can trigger regardless of goal reaching status.
        2) end_goal_changing: can trigger when the human reaches its current goal.
        """
        if self.current_scenario == "crossing":
            return
        if not (self.random_goal_changing or self.end_goal_changing):
            return
        n = int(self.num_humans)
        if n <= 0:
            return

        # Humans with effectively zero preferred speed do not change goals.
        active = np.asarray(self.human_vmaxs, dtype=float).reshape(n) > 1e-8
        if not np.any(active):
            return

        random_trigger = np.zeros((n,), dtype=bool)
        if self.random_goal_changing:
            random_vals = self.np_random.random(n)
            random_trigger = random_vals <= float(self.goal_change_chance)

        end_goal_trigger = np.zeros((n,), dtype=bool)
        if self.end_goal_changing:
            dist_to_goal = np.linalg.norm(self.human_positions - self.human_goals, axis=1)
            near_goal = dist_to_goal <= (self.human_radii + 0.1)
            end_vals = self.np_random.random(n)
            end_goal_trigger = near_goal & (end_vals <= float(self.end_goal_change_chance))

        trigger = active & (random_trigger | end_goal_trigger)
        triggered_idx = np.flatnonzero(trigger)
        for i in triggered_idx:
            self.human_goals[i] = self._sample_new_human_goal(int(i))

    def _sample_new_human_goal(self, idx):
        # Sample goal uniformly in disk with only one hard constraint:
        # ||current_position - new_goal|| >= 6.0
        p = np.asarray(self.human_positions[idx], dtype=float)
        min_pair_dist = 6.0
        r = float(self.human_circle_radius)

        for _ in range(50):
            candidate = sample_point_in_disk(
                self.np_random, center=[0.0, 0.0], radius=r, arena_size=self.arena_size
            )
            if np.linalg.norm(p - candidate) >= min_pair_dist:
                return candidate

        # Fallback: put goal on circle boundary opposite to current position.
        p_norm = np.linalg.norm(p)
        if p_norm > 1e-8:
            candidate = -(p / p_norm) * r
            if np.linalg.norm(p - candidate) >= min_pair_dist:
                return candidate
        return np.asarray(self.human_goals[idx], dtype=float).copy()
    
    def crossing_init_robot_humans(self):
        robot_pos = np.array([-5.5, 0.0])
        goal_pos = np.array([5.5, 0.0])
        robot_theta = 0.0

        base_y = 0.0
        offsets = np.linspace(-1.0, 1.0, self.num_humans) if self.num_humans > 1 else np.array([0.0])
        human_positions = np.zeros((self.num_humans, 2), dtype=float)
        human_goals = np.zeros((self.num_humans, 2), dtype=float)
        for i in range(self.num_humans):
            human_positions[i] = np.array([5.0, base_y + offsets[i]])
            human_goals[i] = np.array([-5.0, base_y + offsets[i]])

        human_vmaxs = np.full((self.num_humans,), 1.8, dtype=float)
        human_radii = np.full((self.num_humans,), 1.0, dtype=float)
        human_vels = np.zeros((self.num_humans, 2), dtype=float)
        robot_vel = np.zeros(2)
        return robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii
