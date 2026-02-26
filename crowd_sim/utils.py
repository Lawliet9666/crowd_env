import json
import os

import numpy as np
from config.config import Config
import torch
 

def parse_obstacles(obs):
    """
    Parse obstacle blocks from observation.
    Supports:
    1) New format: [robot(6), K * (rel_x, rel_y, vx, vy, radius, mask)]
    2) Legacy format: [robot(6), rel_x, rel_y, vx, vy, radius]
    Returns:
        rels: (N, 2), vels: (N, 2), radii: (N,), masks: (N,)
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)

    # New K-obstacle format
    if obs.size >= 12 and (obs.size - 6) % 6 == 0:
        blocks = obs[6:].reshape(-1, 6)
        rels = blocks[:, 0:2].astype(np.float64)
        vels = blocks[:, 2:4].astype(np.float64)
        radii = blocks[:, 4].astype(np.float64)
        masks = np.clip(blocks[:, 5].astype(np.float64), 0.0, 1.0)
        return (
            rels,
            vels,
            radii,
            masks,
        )

    # Legacy single-obstacle format
    if obs.size >= 11:
        return (
            obs[6:8].reshape(1, 2).astype(np.float64),
            obs[8:10].reshape(1, 2).astype(np.float64),
            np.array([float(obs[10])], dtype=np.float64),
            np.ones((1,), dtype=np.float64),
        )

    # No obstacle info in observation
    return (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )


def sample_point_in_disk(rng, center, radius, arena_size=None, max_tries=256):
    center = np.asarray(center, dtype=float)
    for _ in range(max_tries):
        rr = radius * np.sqrt(rng.uniform(0.0, 1.0))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        p = center + rr * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        if arena_size is None:
            return p
        if (-arena_size <= p[0] <= arena_size) and (-arena_size <= p[1] <= arena_size):
            return p
    if arena_size is None:
        return center.copy()
    return np.clip(center, -arena_size, arena_size)


def build_env(env_name: str, render_mode: str, config: Config):
    from crowd_sim.env.social_nav import SocialNav
    from crowd_sim.env.social_nav_var_num import SocialNavVarNum

    if env_name == "social_nav":
        return SocialNav(render_mode=render_mode, config_file=config)
    if env_name == "social_nav_var_num":
        return SocialNavVarNum(render_mode=render_mode, config_file=config)
    raise ValueError(f"Unknown env: {env_name}")

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if torch is not None:
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def dump_test_config(test_save_dir, config, hyperparameters=None, extra=None):
    os.makedirs(test_save_dir, exist_ok=True)
    dst = os.path.join(test_save_dir, "test_config.json")
    config_dict = {
        "config": {
            "env": to_jsonable(dict(config.env)),
            "human": to_jsonable(dict(config.human)),
            "robot": to_jsonable(dict(config.robot)),
            "controller": to_jsonable(dict(config.controller)),
            "reward": to_jsonable(dict(config.reward)),
        },
    }
    if hyperparameters is not None:
        config_dict["hyperparameters"] = to_jsonable(hyperparameters)
    if extra is not None:
        config_dict["extra"] = to_jsonable(extra)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved test config snapshot: {dst}", flush=True)


def dump_train_config(save_dir, args, config, hyperparameters=None, extra=None):
    os.makedirs(save_dir, exist_ok=True)

    payload = {
        "args": to_jsonable(vars(args) if hasattr(args, "__dict__") else args),
        "config": {
            "env": to_jsonable(dict(config.env)),
            "human": to_jsonable(dict(config.human)),
            "robot": to_jsonable(dict(config.robot)),
            "controller": to_jsonable(dict(config.controller)),
            "reward": to_jsonable(dict(config.reward)),
        },
    }

    if hyperparameters is not None:
        payload["hyperparameters"] = to_jsonable(hyperparameters)
    if extra is not None:
        payload["extra"] = to_jsonable(extra)

    with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
