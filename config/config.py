import numpy as np


class BaseConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        return dict(self)


class Config:
    def __init__(self):
        # Environment Parameters
        self.env = BaseConfig()
        self.env.name = "social_nav_var_num"
        self.env.dt = 0.1
        self.env.max_steps = 400
        self.env.sensing_radius = 20.0
        self.env.max_obstacles_obs = 1
        self.env.normalize_obs = False
        # For pure RL + unicycle only:
        # when enabled, policy outputs [vx, vy], env.step converts to [v, omega].
        # self.env.rl_xy_to_unicycle = True
        self.env.rl_xy_to_unicycle = False  # when unicycle+rl, this is set to True in main_vec.py

        self.env.unicycle_k_omega = 4.0

        # Human Parameters
        self.human = BaseConfig()
        # self.human.radius = (0.3, 0.5)  # can be a single value or a range (min, max)
        self.human.radius = 0.3  # can be a single value or a range (min, max)
        # self.human.radius = 0.3
        self.human.vmax = (0.5, 1.5)
        self.human.arena_size = 6.0
        self.human.policy = "social_force"  # 'nominal', 'orca', 'social_force', or 'potential_field'
        self.human.num_humans = 20
        self.human.human_num_range = 0
        self.human.randomize_attributes = True
        # Whether to apply GMM perturbation to human actions in env.step.
        self.human.use_gmm = True

        # Human goal changing behavior
        self.human.random_goal_changing = False
        self.human.goal_change_chance = 0.5
        self.human.end_goal_changing = True
        self.human.end_goal_change_chance = 1.0
        self.human.orca = {
            "neighbor_dist": 10.0,
            "time_horizon": 5.0,
            "time_horizon_obst": 5.0,
            "safety_space": 0.15,
            "avoid_robot": False,
        }
        self.human.sf = {
            "A": 8.0,
            "B": 0.2,
            "KI": 3.0,
            "avoid_robot": False,
        }
        self.human.pf = {
            "human_margin": 0.1,
            "human_influence": 0.5,
            "human_gain": 1.5,
            "robot_margin": 0.15,
            "robot_influence": 0.8,
            "robot_gain": 2.0,
            "avoid_robot": False,
        }

        # Robot Parameters
        self.robot = BaseConfig()
        self.robot.radius = 0.3
        self.robot.vmax = 1.0
        self.robot.amax = 2.0
        self.robot.omega_max = np.pi / 2
        self.robot.type = "unicycle"  # 'single_integrator', 'unicycle', 'unicycle_dynamic'

        # Optimization controller parameters
        self.controller = BaseConfig()
        self.controller.cbf_alpha = 2.0  # for CBF and CVaR-CBF Controller
        self.controller.cvar_beta = 0.5  # for CVaR-BFQP Controller
        self.controller.safety_margin = 0.10  # similar to discomfort distance but used in controller

        # Reward Parameters
        self.reward = BaseConfig()
        self.reward.success_reward = 20
        self.reward.collision_penalty = -20
        self.reward.discomfort_dist = 0.25  # used in reward calculation
        self.reward.discomfort_penalty_factor = 10

        # self.reward.safe_shaping_weight = 0.0  # set >0 to enable smooth near-collision penalty
        self.reward.safe_shaping_weight = 0.3  # set >0 to enable smooth near-collision penalty
        self.reward.safe_shaping_band = 0.6    # active range: [discomfort_dist, discomfort_dist + band]

        if self.robot.type == 'unicycle':
            self.reward.potential_factor = 3.0  # increase potential reward for dynamic robot to encourage faster goal-reaching
            self.reward.back_factor = 0.1
            # self.reward.spin_factor = 0.05
            self.reward.spin_factor = 4.5
        elif self.robot.type == 'single_integrator':
            self.reward.potential_factor = 2.0
            self.reward.back_factor = 0.0
            self.reward.spin_factor = 0.0
        self.reward.constant_penalty = -0.025

        # Deprecated: PPO hyperparameters are provided via CLI args (see config/arguments.py).
        # Keep this object for backward compatibility only.
        self.hyperparameters = BaseConfig()

        # Backwards-compatible aliases
        self.env_params = self.env
        self.human_params = self.human
        self.robot_params = self.robot
        self.controller_params = self.controller
        self.reward_params = self.reward
