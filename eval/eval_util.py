"""
Minimal evaluation utilities for trained policies.
"""

import json
import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from crowd_sim.utils import absolute_obs_to_polar, absolute_obs_to_relative


class RLEvalActorAdapter:
    """Deterministic RL action mapping consistent with PPO tanh-squash."""

    def __init__(self, actor, action_space, device):
        self.actor = actor
        self.device = device
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        self.scale = 0.5 * (high - low)
        self.bias = 0.5 * (high + low)
        self.deterministic = True

    def get_action(self, obs):
        obs_qp_t = None
        if isinstance(obs, (tuple, list)) and len(obs) == 2:
            obs_actor, obs_qp = obs
            obs_t = torch.as_tensor(obs_actor, dtype=torch.float32, device=self.device)
            obs_qp_t = torch.as_tensor(obs_qp, dtype=torch.float32, device=self.device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if obs_qp_t is None:
                mean = self.actor(obs_t)
            else:
                mean = self.actor(obs_t, obs_qp_t)
        if torch.is_tensor(mean):
            mean = mean.detach().cpu().numpy()
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        action = self.bias + self.scale * np.tanh(mean)
        return action, 1.0

    def __getattr__(self, name):
        return getattr(self.actor, name)


def resolve_episode_seed(base_seed, episode_index):
    if base_seed is None:
        return None
    return int(base_seed) + int(episode_index)


def build_obs_preprocess_fn(obs_topk=5, obs_farest_dist=5.0, needs_qp_relative=False):
    topk = int(obs_topk)
    farest_dist = float(obs_farest_dist)
    use_qp_relative = bool(needs_qp_relative)

    def _preprocess(obs):
        obs_polar = absolute_obs_to_polar(obs, topk=topk, farest_dist=farest_dist)
        if not use_qp_relative:
            return obs_polar
        obs_rel = absolute_obs_to_relative(obs, topk=topk)
        return (obs_polar, obs_rel)

    return _preprocess


def _compute_action(
    actor,
    obs,
    obs_preprocess_fn=None,
):
    if obs_preprocess_fn is not None:
        obs = obs_preprocess_fn(obs)
    if hasattr(actor, "deterministic"):
        actor.deterministic = True
    out = actor.get_action(obs)
    action = out[0] if isinstance(out, (tuple, list)) else out

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _reset_actor_episode_cache(actor):
    targets = [actor]
    seen = set()
    while len(targets) > 0:
        obj = targets.pop()
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        if hasattr(obj, "_qp_warm_start"):
            obj._qp_warm_start = None
        if hasattr(obj, "_u_prev"):
            obj._u_prev = None
        if hasattr(obj, "infeasible"):
            obj.infeasible = False

        nested_actor = getattr(obj, "actor", None)
        if nested_actor is not None and nested_actor is not obj:
            targets.append(nested_actor)
        nested_module = getattr(obj, "module", None)
        if nested_module is not None and nested_module is not obj:
            targets.append(nested_module)


def _log_summary(ep_len, ep_ret, ep_num, ep_collision, ep_success):
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {round(ep_len, 2)}", flush=True)
    print(f"Episodic Return: {round(ep_ret, 2)}", flush=True)
    print(f"Collision: {'YES' if ep_collision else 'NO'}", flush=True)
    success_flag = (ep_success and (not ep_collision))
    print(f"Success: {'YES' if success_flag else 'NO'}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def _safe_scalar(val, default=np.nan):
    if val is None:
        return default
    if isinstance(val, (list, tuple)) and len(val) > 0:
        val = val[0]
    if torch.is_tensor(val):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return default
        val = val.flatten()[0]
    try:
        return float(val)
    except Exception:
        return default

def _control_plot_bound(env, action_series):
    env_ref = getattr(env, "unwrapped", env)
    robot = getattr(env_ref, "robot", None)
    robot_type = str(getattr(env_ref, "robot_type", getattr(robot, "type", ""))).lower()

    if robot_type == "single_integrator":
        vmax = float(getattr(robot, "vmax", 1.0))
        return max(abs(vmax), 1e-3)
    if robot_type == "unicycle":
        vmax = float(getattr(robot, "vmax", 1.0))
        wmax = float(getattr(robot, "w_max", getattr(robot, "omega_max", vmax)))
        return max(abs(vmax), abs(wmax), 1e-3)
    if robot_type == "unicycle_dynamic":
        wmax = float(getattr(robot, "w_max", getattr(robot, "omega_max", 1.0)))
        vmax = float(getattr(robot, "vmax", 1.0))
        return max(abs(vmax), abs(wmax), 1e-3)

    if action_series is not None and action_series.size > 0:
        finite = np.abs(action_series[np.isfinite(action_series)])
        if finite.size > 0:
            return max(float(np.max(finite)), 1e-3)
    return 1.0


def _compose_frames(env_frames, metrics, dt, env=None):
    frames = []
    T = len(env_frames)
    time_axis = np.arange(T) * dt
    beta_series = np.array([_safe_scalar(v) for v in metrics.get("beta", [])], dtype=float)
    delta_r_series = np.array([_safe_scalar(v) for v in metrics.get("delta_r", [])], dtype=float)
    action_exec = metrics.get("action_exec", [])
    u_series = np.array(action_exec, dtype=float) if action_exec else np.full((T, 2), np.nan, dtype=float)
    u_bound = _control_plot_bound(env, u_series) if env is not None else 1.0

    for t in range(T):
        fig = plt.figure(figsize=(9.6, 6.2))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], wspace=0.06, hspace=0.34)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.11)
        title_fs = 16
        axis_label_fs = 14
        tick_fs = 12
        legend_fs = 10

        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.imshow(env_frames[t])
        ax_env.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])
        for ax in (ax1, ax2, ax3):
            ax.tick_params(axis="both", labelsize=tick_fs)

        ax1.plot(time_axis[:t+1], beta_series[:t+1], color="tab:blue")
        ax1.set_title(r"$\beta$", fontsize=title_fs)
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, alpha=0.2)

        ax2.plot(time_axis[:t+1], delta_r_series[:t+1], color="tab:purple")
        ax2.set_title(r"$\Delta R$", fontsize=title_fs)
        ax2.grid(True, alpha=0.2)
        ax2.set_ylim(0.7, 1.6)

        ax3.plot(time_axis[:t+1], u_series[:t+1, 0], label=r"$u[0]$", color="tab:orange")
        ax3.plot(time_axis[:t+1], u_series[:t+1, 1], label=r"$u[1]$", color="tab:green")
        ax3.set_title(r"$u$", fontsize=title_fs)
        ax3.set_xlabel(r"$t$ (s)", fontsize=axis_label_fs)
        ax3.grid(True, alpha=0.2)
        ax3.set_ylim(-u_bound, u_bound)
        ax3.legend(loc="upper right", fontsize=legend_fs)

        fig.canvas.draw()
        if hasattr(fig.canvas, "buffer_rgba"):
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = rgba[:, :, :3].copy()
        else:
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            frame = data.reshape(h, w, 3)
        plt.close(fig)
        frames.append(frame)
    return frames


def _init_metrics(track_signals):
    if not track_signals:
        return None
    return {
        "beta": [],
        "delta_r": [],
        "action_exec": [],
    }


def _render_step(env, frames):
    if env.render_mode is None:
        return
    if env.render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    else:
        env.render()


def _record_actor_metrics(metrics, actor):
    if metrics is None:
        return

    metrics["beta"].append(getattr(actor, "last_beta", getattr(actor, "beta", None)))
    metrics["delta_r"].append(getattr(actor, "last_r_safe", getattr(actor, "safe_dist", None)))


def _record_executed_action(metrics, env, fallback_action):
    if metrics is None:
        return

    robot = getattr(env, "robot", None)
    if robot is None and hasattr(env, "unwrapped"):
        robot = getattr(env.unwrapped, "robot", None)

    if robot is not None and getattr(robot, "u", None) is not None:
        executed = robot.u
    else:
        executed = fallback_action
    metrics["action_exec"].append(np.array(executed, dtype=float))


def run_one_episode(
    actor,
    env,
    seed=None,
    reset_options=None,
    track_signals=False,
    collect_frames=False,
    obs_preprocess_fn=None,
):
    """Run a single episode and return step-level and episode-level results."""
    if seed is None and reset_options is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed, options=reset_options)
    _reset_actor_episode_cache(actor)

    done = False
    ep_len = 0
    ep_ret = 0.0
    ep_collision = False
    ep_success = False
    ep_timeout = False
    ep_infeasible = False

    frames = []
    metrics = _init_metrics(track_signals)

    while not done:
        ep_len += 1

        if collect_frames:
            _render_step(env, frames)

        action = _compute_action(
            actor,
            obs,
            obs_preprocess_fn=obs_preprocess_fn,
        )
        _record_actor_metrics(metrics, actor)
        if bool(getattr(actor, "infeasible", False)):
            ep_infeasible = True
            _record_executed_action(metrics, env, action)
            # done = True
            # break

        obs, rew, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        _record_executed_action(metrics, env, action)

        ep_collision = ep_collision or bool(info.get("is_collision", False))
        ep_success = ep_success or bool(info.get("is_success", False))
        ep_timeout = ep_timeout or bool(info.get("is_timeout", False))
        ep_ret += float(rew)

    return {
        "ep_len": ep_len,
        "ep_ret": ep_ret,
        "ep_collision": ep_collision,
        "ep_success": ep_success,
        "ep_timeout": ep_timeout,
        "ep_infeasible": ep_infeasible,
        "frames": frames,
        "metrics": metrics,
    }


def eval_policy(
    policy,
    env,
    max_episodes=50,
    save_path=None,
    base_seed=None,
    method=None,
    obs_preprocess_fn=None,
    visualize_episodes=20,
):
    total_episodes = 0
    success_count = 0
    collision_count = 0
    infeasible_count = 0

    actor = policy
   
    mode = (method or "").lower()
    track_signals = ("cbf" in mode) or ("cvar" in mode)
    for ep_num in range(max_episodes):
        seed = resolve_episode_seed(base_seed, ep_num)
        result = run_one_episode(
            actor,
            env,
            seed=seed,
            reset_options=None,
            track_signals=track_signals,
            collect_frames=True,
            obs_preprocess_fn=obs_preprocess_fn,
        )
        ep_len = result["ep_len"]
        ep_ret = result["ep_ret"]
        ep_collision = result["ep_collision"]
        ep_success = result["ep_success"]
        ep_infeasible = result["ep_infeasible"]
        frames = result["frames"]
        metrics = result["metrics"]

        _log_summary(ep_len, ep_ret, ep_num, ep_collision, ep_success)

        success_flag = ep_success and (not ep_collision)
        if len(frames) > 0 and save_path:
            os.makedirs(save_path, exist_ok=True)

            # Save the first N episodes regardless of success/failure.
            if ep_num < visualize_episodes:
                succ_bit = 1 if success_flag else 0
                coll_bit = 1 if ep_collision else 0
                gif_name = f"eval_ep_{ep_num}_succ_{succ_bit}_coll_{coll_bit}.gif"
                full_path = os.path.join(save_path, gif_name)

                if track_signals and metrics is not None:
                    dt = getattr(env, "dt", 1.0)
                    composed = _compose_frames(frames, metrics, dt, env=env)
                    imageio.mimsave(full_path, composed, fps=10)
                else:
                    imageio.mimsave(full_path, frames, fps=10)
                print(f"Saved evaluation animation to {full_path}")

        total_episodes += 1
        if ep_success and not ep_collision:
            success_count += 1
        if ep_collision:
            collision_count += 1
        if ep_infeasible:
            infeasible_count += 1

    print("\n\n-------------------- Evaluation Summary --------------------")
    print(f"Total Episodes: {total_episodes}")
    if total_episodes > 0:
        success_rate = success_count / total_episodes
        collision_rate = collision_count / total_episodes
        infeasible_rate = infeasible_count / total_episodes
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Collision Rate: {collision_rate * 100:.2f}%")
        print(f"Infeasible Rate: {infeasible_rate * 100:.2f}%")
    else:
        success_rate = None
        collision_rate = None
        infeasible_rate = None
        print("Success Rate: N/A")
    print("------------------------------------------------------------")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        summary = {
            "total_episodes": total_episodes,
            "success_count": success_count,
            "collision_count": collision_count,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "infeasible_count": infeasible_count,
            "infeasible_rate": infeasible_rate,
        }

        log_payload = {
            "config": {
                "policy_class": actor.__class__.__name__,
                "render_mode": getattr(env, "render_mode", None),
                "max_episodes": max_episodes,
                "base_seed": base_seed,
                "method": mode,
            },
            "results": summary,
        }
        with open(os.path.join(save_path, "eval_log.json"), "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)


def run_crossing_scenario(
    policy,
    env,
    save_path=None,
    obs_preprocess_fn=None,
):
    actor = policy
    result = run_one_episode(
        actor=actor,
        env=env,
        seed=None,
        reset_options={"scenario": "crossing"},
        track_signals=False,
        collect_frames=True,
        obs_preprocess_fn=obs_preprocess_fn,
    )
    frames = result["frames"]
    is_collision = bool(result["ep_collision"])
    is_success = bool(result["ep_success"])

    if save_path and len(frames) > 0:
        os.makedirs(save_path, exist_ok=True)
        imageio.mimsave(os.path.join(save_path, "crossing.gif"), frames, fps=10)

    print(f"Crossing scenario finished. Collision: {is_collision}, Success: {is_success}")
