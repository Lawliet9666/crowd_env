import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import Config as CrowdSimConfig
from crowd_sim.utils import absolute_obs_to_polar, absolute_obs_to_relative
from crowd_sim.env.social_nav import SocialNav
from crowd_sim.env.social_nav_var_num import SocialNavVarNum
from gymnasium import spaces


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _to_polar_obs(obs: np.ndarray, topk: int = 5, farest_dist: float = 6.0) -> np.ndarray:
    """
    Convert absolute obs to minimal polar format.
    Robot: [dist_to_goal, angle_to_goal, robot_speed]
    Per human (top-k nearest): [clearance, angle, rel_vel_radial, rel_vel_tangential]
    - mask=0 (far/padded): clearance=farest_dist, angle=0, rel_vel_radial=0, rel_vel_tangential=0
    - mask=1: clearance truncated to max farest_dist; angles relative to robot heading
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.size < 8 or (obs.size - 8) % 6 != 0:
        return obs

    rx, ry, gx, gy, rvx, rvy, rtheta, r_radius = obs[:8]
    k_in = (obs.size - 8) // 6
    blocks = obs[8:].reshape(k_in, 6)
    k = min(k_in, topk)

    # --- Robot block ---
    to_goal = np.array([gx - rx, gy - ry], dtype=np.float32)
    dist_to_goal = float(np.linalg.norm(to_goal))
    if dist_to_goal > 1e-8:
        goal_dir = np.arctan2(to_goal[1], to_goal[0])
        angle_to_goal = _wrap_angle(goal_dir - rtheta)
    else:
        angle_to_goal = 0.0
    robot_linear_speed = float(np.linalg.norm([rvx, rvy]))
    robot_block = np.array([dist_to_goal, angle_to_goal, robot_linear_speed], dtype=np.float32)
    # print("robot_block", robot_block)
    # --- Top-k human blocks: [clearance, angle, rel_vel_radial] ---
    cos_t, sin_t = np.cos(rtheta), np.sin(rtheta)
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)

    out_blocks = np.full((topk, 4), [farest_dist, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(k):
        hx, hy, hvx, hvy, h_radius, mask = blocks[i]
        if mask < 0.5:
            out_blocks[i] = [farest_dist, 0.0, 0.0, 0.0]
            continue
        rel_pos = np.array([hx - rx, hy - ry], dtype=np.float32)
        dist_cc = float(np.linalg.norm(rel_pos))
        clearance = dist_cc - r_radius - h_radius
        clearance = min(clearance, farest_dist)
        if dist_cc < 1e-8:
            angle = 0.0
        else:
            world_angle = np.arctan2(rel_pos[1], rel_pos[0])
            angle = _wrap_angle(world_angle - rtheta)
        rel_vel = np.array([hvx - rvx, hvy - rvy], dtype=np.float32)
        rel_vel_robot = rot @ rel_vel
        rel_vel_radial = rel_vel_robot[0]
        rel_vel_tangential = rel_vel_robot[1]
        out_blocks[i] = [clearance, angle, rel_vel_radial, rel_vel_tangential]
    # print(out_blocks)
    return np.concatenate([robot_block, out_blocks.ravel()]).astype(np.float32)


class SocialNavVarNumPloar(SocialNavVarNum):
    """Wrapper that converts observations to relative polar (radius, angle) format."""

    def __init__(self, *args, topk: int = 5, farest_dist: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.farest_dist = farest_dist
        self.obs_dim = 3 + topk * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        return _to_polar_obs(obs, topk=self.topk, farest_dist=self.farest_dist)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return self._preprocess_obs(obs), reward, done, truncated, info


class SocialNavVarNumHybrid(SocialNavVarNum):
    """Return concatenated actor polar obs and QP relative obs."""

    def __init__(self, *args, topk: int = 5, farest_dist: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = int(topk)
        self.farest_dist = float(farest_dist)
        self.actor_obs_dim = 3 + self.topk * 4
        self.qp_obs_dim = 6 + self.topk * 6
        self.obs_dim = self.actor_obs_dim + self.qp_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        polar_obs = absolute_obs_to_polar(
            obs,
            topk=self.topk,
            farest_dist=self.farest_dist,
        )
        relative_obs = absolute_obs_to_relative(obs, topk=self.topk)
        return np.concatenate([polar_obs, relative_obs]).astype(np.float32)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return self._preprocess_obs(obs), reward, done, truncated, info


def build_env(env_name: str, render_mode: str, config: CrowdSimConfig, *args, **kwargs):
    if env_name == "social_nav":
        return SocialNav(render_mode=render_mode, config_file=config)
    if env_name == "social_nav_var_num":
        return SocialNavVarNum(render_mode=render_mode, config_file=config)
    if env_name == "social_nav_var_num_ploar":
        topk = kwargs.get("topk", 5)
        farest_dist = kwargs.get("farest_dist", 5)
        return SocialNavVarNumPloar(
            render_mode=render_mode,
            config_file=config,
            topk=topk,
            farest_dist=farest_dist,
        )
    if env_name == "social_nav_var_num_hybrid":
        topk = kwargs.get("topk", 5)
        farest_dist = kwargs.get("farest_dist", 5)
        return SocialNavVarNumHybrid(
            render_mode=render_mode,
            config_file=config,
            topk=topk,
            farest_dist=farest_dist,
        )
    raise ValueError(f"Unknown env: {env_name}")


if __name__ == "__main__":
    config = CrowdSimConfig()
    config.env.max_obstacles_obs = 20
    env = build_env("social_nav_var_num_ploar", None, config)
    obs, info = env.reset()
    done = False
    while not done:
        action = np.array([0.0, 0.0])
        obs, reward, done, truncated, info = env.step(action)
