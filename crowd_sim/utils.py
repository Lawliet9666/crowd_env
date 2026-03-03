import json
import os

import numpy as np
from config.config import Config
 

def is_absolute_obs_dim(obs_dim: int) -> bool:
    dim = int(obs_dim)
    return dim >= 8 and (dim - 8) % 6 == 0


def is_polar_obs_dim(obs_dim: int) -> bool:
    dim = int(obs_dim)
    return dim >= 3 and (dim - 3) % 4 == 0


def _validate_topk(topk, allow_none: bool = False):
    if topk is None:
        if allow_none:
            return None
        raise ValueError("topk must be provided")
    k = int(topk)
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    return k


def relative_obs_dim_from_env_dim(obs_dim: int, topk=None) -> int:
    dim = int(obs_dim)
    k = _validate_topk(topk, allow_none=True)
    if is_absolute_obs_dim(dim):
        if k is None:
            # abs format: 8 + K*6 -> legacy rel format: 6 + K*6
            return dim - 2
        return 6 + 6 * k
    if dim >= 6 and (dim - 6) % 6 == 0:
        if k is None:
            return dim
        return 6 + 6 * k
    return dim


def polar_obs_dim_from_env_dim(obs_dim: int, topk=None) -> int:
    dim = int(obs_dim)
    k = _validate_topk(topk, allow_none=True)
    if is_absolute_obs_dim(dim):
        if k is None:
            k = (dim - 8) // 6
        return 3 + 4 * k
    if is_polar_obs_dim(dim):
        if k is None:
            return dim
        return 3 + 4 * k
    return dim


def hybrid_obs_dim_from_env_dim(obs_dim: int, topk=None) -> int:
    return int(
        relative_obs_dim_from_env_dim(obs_dim, topk=topk)
        + polar_obs_dim_from_env_dim(obs_dim, topk=topk)
    )


def absolute_obs_to_relative(obs, topk=None):
    """
    Convert observation from absolute format to legacy relative format.

    Absolute format (1D):
      [rx, ry, gx, gy, rvx, rvy, rtheta, rr, (hx, hy, hvx, hvy, hr, mask)*K]

    Legacy relative format (1D):
      [goal_rel_x, goal_rel_y, rvx, rvy, rtheta, rr, (rel_x, rel_y, hvx, hvy, hr, mask)*K]
    """
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.size >= 8 and (x.size - 8) % 6 == 0:
        k_in = (x.size - 8) // 6
        k_out = _validate_topk(topk, allow_none=True)
        if k_out is None:
            k_out = k_in
        out = np.zeros((6 + 6 * k_out,), dtype=np.float32)

        rx, ry, gx, gy, rvx, rvy, rtheta, rr = x[:8]
        out[0] = rx - gx
        out[1] = ry - gy
        out[2] = rvx
        out[3] = rvy
        out[4] = rtheta
        out[5] = rr

        take = min(k_in, k_out)
        if take > 0:
            blocks = x[8:].reshape(k_in, 6)[:take]
            out_blocks = np.zeros((k_out, 6), dtype=np.float32)
            out_blocks[:take, 0] = rx - blocks[:, 0]
            out_blocks[:take, 1] = ry - blocks[:, 1]
            out_blocks[:take, 2:6] = blocks[:, 2:6]
            out[6:] = out_blocks.reshape(-1)
        return out

    if x.size >= 6 and (x.size - 6) % 6 == 0:
        k_out = _validate_topk(topk, allow_none=True)
        if k_out is None:
            return x
        out = np.zeros((6 + 6 * k_out,), dtype=np.float32)
        k_in = (x.size - 6) // 6
        out[:6] = x[:6]
        take = min(k_in, k_out)
        if take > 0:
            blocks = x[6:].reshape(k_in, 6)[:take]
            out_blocks = np.zeros((k_out, 6), dtype=np.float32)
            out_blocks[:take, :] = blocks
            out[6:] = out_blocks.reshape(-1)
        return out

    raise ValueError(f"Unsupported observation length for abs->rel conversion: {x.size}")


def absolute_obs_batch_to_relative(obs_batch, topk=None):
    """
    Batch version of absolute_obs_to_relative.
    Input can be shape (N, D) absolute observations or already-relative (N, D_rel).
    """
    arr = np.asarray(obs_batch, dtype=np.float32)

    if arr.ndim == 1:
        return absolute_obs_to_relative(arr, topk=topk)
    if arr.ndim != 2:
        raise ValueError(f"Expected obs batch with ndim 1 or 2, got shape {arr.shape}")

    n, d = arr.shape
    if d >= 8 and (d - 8) % 6 == 0:
        k_in = (d - 8) // 6
        k_out = _validate_topk(topk, allow_none=True)
        if k_out is None:
            k_out = k_in
        out = np.zeros((n, 6 + 6 * k_out), dtype=np.float32)

        rx = arr[:, 0:1]
        ry = arr[:, 1:2]
        gx = arr[:, 2:3]
        gy = arr[:, 3:4]

        out[:, 0:1] = rx - gx
        out[:, 1:2] = ry - gy
        out[:, 2:6] = arr[:, 4:8]

        take = min(k_in, k_out)
        if take > 0:
            blocks = arr[:, 8:].reshape(n, k_in, 6)[:, :take, :]
            out_blocks = np.zeros((n, k_out, 6), dtype=np.float32)
            out_blocks[:, :take, 0] = rx - blocks[:, :, 0]
            out_blocks[:, :take, 1] = ry - blocks[:, :, 1]
            out_blocks[:, :take, 2:6] = blocks[:, :, 2:6]
            out[:, 6:] = out_blocks.reshape(n, 6 * k_out)

        return out

    if d >= 6 and (d - 6) % 6 == 0:
        k_out = _validate_topk(topk, allow_none=True)
        if k_out is None:
            return arr
        k_in = (d - 6) // 6
        out = np.zeros((n, 6 + 6 * k_out), dtype=np.float32)
        out[:, :6] = arr[:, :6]
        take = min(k_in, k_out)
        if take > 0:
            blocks = arr[:, 6:].reshape(n, k_in, 6)[:, :take, :]
            out_blocks = np.zeros((n, k_out, 6), dtype=np.float32)
            out_blocks[:, :take, :] = blocks
            out[:, 6:] = out_blocks.reshape(n, 6 * k_out)
        return out

    raise ValueError(f"Unsupported batch observation width for abs->rel conversion: {d}")


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def absolute_obs_to_polar(obs, topk: int = 5, farest_dist: float = 5.0):
    """
    Convert absolute-format observation to compact polar format.

    Output layout:
      [dist_to_goal, angle_to_goal, robot_speed,
       (clearance, angle, rel_vel_radial, rel_vel_tangential) * topk]
    """
    x = np.asarray(obs, dtype=np.float32).reshape(-1)

    if x.size < 8 or (x.size - 8) % 6 != 0:
        return x

    rx, ry, gx, gy, rvx, rvy, rtheta, rr = x[:8]
    blocks = x[8:].reshape(-1, 6)

    topk = _validate_topk(topk)
    farest_dist = float(farest_dist)
    k = min(blocks.shape[0], topk)

    to_goal = np.array([gx - rx, gy - ry], dtype=np.float32)
    dist_to_goal = float(np.linalg.norm(to_goal))
    if dist_to_goal > 1e-8:
        goal_dir = np.arctan2(to_goal[1], to_goal[0])
        angle_to_goal = _wrap_angle(float(goal_dir - rtheta))
    else:
        angle_to_goal = 0.0

    robot_speed = float(np.linalg.norm([rvx, rvy]))
    robot_block = np.array([dist_to_goal, angle_to_goal, robot_speed], dtype=np.float32)

    cos_t, sin_t = np.cos(rtheta), np.sin(rtheta)
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
    out_blocks = np.full((topk, 4), [farest_dist, 0.0, 0.0, 0.0], dtype=np.float32)

    for i in range(k):
        hx, hy, hvx, hvy, hr, mask = blocks[i]
        if mask < 0.5:
            continue

        rel_pos = np.array([hx - rx, hy - ry], dtype=np.float32)
        dist_cc = float(np.linalg.norm(rel_pos))
        clearance = min(dist_cc - rr - hr, farest_dist)
        if dist_cc > 1e-8:
            world_angle = np.arctan2(rel_pos[1], rel_pos[0])
            angle = _wrap_angle(float(world_angle - rtheta))
        else:
            angle = 0.0

        rel_vel = np.array([hvx - rvx, hvy - rvy], dtype=np.float32)
        rel_vel_robot = rot @ rel_vel
        out_blocks[i] = [clearance, angle, rel_vel_robot[0], rel_vel_robot[1]]

    return np.concatenate([robot_block, out_blocks.reshape(-1)]).astype(np.float32)


def absolute_obs_batch_to_polar(obs_batch, topk: int = 5, farest_dist: float = 5.0):
    arr = np.asarray(obs_batch, dtype=np.float32)
    if arr.ndim == 1:
        return absolute_obs_to_polar(arr, topk=topk, farest_dist=farest_dist)
    if arr.ndim != 2:
        raise ValueError(f"Expected obs batch with ndim 1 or 2, got shape {arr.shape}")
    return np.stack(
        [absolute_obs_to_polar(row, topk=topk, farest_dist=farest_dist) for row in arr],
        axis=0,
    )


def absolute_obs_to_hybrid(obs, topk: int = 5, farest_dist: float = 5.0):
    rel = absolute_obs_to_relative(obs, topk=topk)
    polar = absolute_obs_to_polar(obs, topk=topk, farest_dist=farest_dist)
    return np.concatenate([rel, polar], axis=0).astype(np.float32)


def absolute_obs_batch_to_hybrid(obs_batch, topk: int = 5, farest_dist: float = 5.0):
    arr = np.asarray(obs_batch, dtype=np.float32)
    if arr.ndim == 1:
        return absolute_obs_to_hybrid(arr, topk=topk, farest_dist=farest_dist)
    if arr.ndim != 2:
        raise ValueError(f"Expected obs batch with ndim 1 or 2, got shape {arr.shape}")
    return np.stack(
        [absolute_obs_to_hybrid(row, topk=topk, farest_dist=farest_dist) for row in arr],
        axis=0,
    )


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


def build_env(
    env_name: str,
    render_mode: str,
    config: Config,
    obs_preprocess: str = "relative",
    obs_topk: int = 5,
    obs_farest_dist: float = 5.0,
):
    # obs_topk/obs_farest_dist are accepted for interface consistency; conversion
    # is handled in policy preprocessing (PPO/SAC/eval), not by env wrapper.
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
    type_module = type(obj).__module__
    type_name = type(obj).__name__
    if type_module.startswith("torch"):
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            return obj.detach().cpu().tolist()
        if type_name == "device":
            return str(obj)
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
