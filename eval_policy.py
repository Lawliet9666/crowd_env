"""
Minimal evaluation utilities for trained policies.
"""

import json
import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch


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
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean = self.actor(obs_t)
        if torch.is_tensor(mean):
            mean = mean.detach().cpu().numpy()
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        action = self.bias + self.scale * np.tanh(mean)
        return action, 1.0

    def __getattr__(self, name):
        return getattr(self.actor, name)

def _compute_action(actor, obs):
    if hasattr(actor, "deterministic"):
        actor.deterministic = True
    out = actor.get_action(obs)
    action = out[0] if isinstance(out, (tuple, list)) else out

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


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

def _setup_unom_hook(actor):
    target = None
    if hasattr(actor, "fc31"):
        target = actor.fc31
    elif hasattr(actor, "fc_out"):
        target = actor.fc_out
    if target is None:
        return None, None
    holder = {"val": None}

    def _hook(_module, _inputs, output):
        if torch.is_tensor(output):
            holder["val"] = output.detach().cpu().numpy()
        else:
            holder["val"] = output

    handle = target.register_forward_hook(_hook)
    return handle, holder


def _build_unom_series(values, actions):
    T = len(actions)
    u = np.full((T, 2), np.nan, dtype=float)
    any_unom = False
    for i in range(T):
        val = values[i] if i < len(values) else None
        if val is None:
            continue
        arr = np.asarray(val).reshape(-1)
        if arr.size >= 1:
            u[i, 0] = arr[0]
        if arr.size >= 2:
            u[i, 1] = arr[1]
        any_unom = True
    if not any_unom:
        for i in range(T):
            act = np.asarray(actions[i]).reshape(-1)
            if act.size >= 1:
                u[i, 0] = act[0]
            if act.size >= 2:
                u[i, 1] = act[1]
    return u, any_unom


def _compose_frames(env_frames, metrics, method, dt):
    frames = []
    T = len(env_frames)
    time_axis = np.arange(T) * dt
    alpha_series = np.array([_safe_scalar(v) for v in metrics.get("alpha", [])], dtype=float)
    beta_series = np.array([_safe_scalar(v) for v in metrics.get("beta", [])], dtype=float)
    rsafe_series = np.array([_safe_scalar(v) for v in metrics.get("r_safe", [])], dtype=float)
    action_exec = metrics.get("action_exec", None)
    action_list = action_exec if action_exec is not None else metrics.get("action", [])
    unom_series, has_unom = _build_unom_series(metrics.get("unom", []), action_list)
    action_series = np.array(action_list, dtype=float) if action_list else None

    # Determine plot layout based on method string
    method_str = (method or "").lower()
    is_cbf = "cbf" in method_str
    is_cvar = "cvar" in method_str
    
    # If strictly CBF (and not CVaR), use 2 rows. Otherwise (CVaR or others) use 3.
    rows = 2 if (is_cbf and not is_cvar) else 3
    for t in range(T):
        fig = plt.figure(figsize=(10, 5) if rows == 2 else (10, 6))
        gs = fig.add_gridspec(rows, 2, width_ratios=[1.4, 1.0], wspace=0.35, hspace=0.4)

        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.imshow(env_frames[t])
        ax_env.axis("off")

        if rows == 2:
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 1])
            ax1.plot(time_axis[:t+1], alpha_series[:t+1], color="tab:blue")
            ax1.set_title("gamma/alpha")
            ax1.grid(True, alpha=0.2)
            ax1.set_ylim(-0.1, 4.1)

            ax2.plot(time_axis[:t+1], unom_series[:t+1, 0], label="u_nom[0]" if has_unom else "u[0]", color="tab:orange")
            ax2.plot(time_axis[:t+1], unom_series[:t+1, 1], label="u_nom[1]" if has_unom else "u[1]", color="tab:green")
            if action_series is not None and action_series.shape[0] >= t + 1:
                ax2.plot(time_axis[:t+1], action_series[:t+1, 0], label="u[0]", color="tab:orange", linestyle="--", alpha=0.8)
                ax2.plot(time_axis[:t+1], action_series[:t+1, 1], label="u[1]", color="tab:green", linestyle="--", alpha=0.8)
            ax2.set_title("u_nom vs u" if has_unom else "u")
            ax2.set_xlabel("t (s)")
            ax2.grid(True, alpha=0.2)
            ax2.legend(loc="upper right", fontsize=8)
        else:
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, 1])

            ax1.plot(time_axis[:t+1], beta_series[:t+1], color="tab:blue")
            ax1.set_title("beta")
            ax1.set_ylim(-0.1, 1.1)
            ax1.grid(True, alpha=0.2)

            ax2.plot(time_axis[:t+1], rsafe_series[:t+1], color="tab:purple")
            ax2.set_title("r_safe")
            ax2.grid(True, alpha=0.2)
            ax2.set_ylim(0.5,2.0)

            ax3.plot(time_axis[:t+1], unom_series[:t+1, 0], label="u_nom[0]" if has_unom else "u[0]", color="tab:orange")
            ax3.plot(time_axis[:t+1], unom_series[:t+1, 1], label="u_nom[1]" if has_unom else "u[1]", color="tab:green")
            if action_series is not None and action_series.shape[0] >= t + 1:
                ax3.plot(time_axis[:t+1], action_series[:t+1, 0], label="u[0]", color="tab:orange", linestyle="--", alpha=0.8)
                ax3.plot(time_axis[:t+1], action_series[:t+1, 1], label="u[1]", color="tab:green", linestyle="--", alpha=0.8)
            ax3.set_title("u_nom vs u" if has_unom else "u")
            ax3.set_xlabel("t (s)")
            ax3.grid(True, alpha=0.2)
            ax3.legend(loc="upper right", fontsize=8)

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
        "alpha": [],
        "beta": [],
        "r_safe": [],
        "unom": [],
        "action": [],
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


def _record_actor_metrics(metrics, actor, action, unom_holder):
    if metrics is None:
        return

    metrics["action"].append(np.array(action, dtype=float))
    metrics["unom"].append(unom_holder.get("val") if unom_holder is not None else None)
    metrics["alpha"].append(getattr(actor, "last_alpha", getattr(actor, "alpha", None)))
    metrics["beta"].append(getattr(actor, "last_beta", getattr(actor, "beta", None)))
    metrics["r_safe"].append(getattr(actor, "last_r_safe", getattr(actor, "safe_dist", None)))


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


def rollout(actor, env, base_seed=0, track_signals=False, unom_holder=None):
    ep_cnt = 0
    while True:
        obs, _ = env.reset(seed=base_seed + ep_cnt)
        ep_cnt += 1

        done = False
        ep_len = 0
        ep_ret = 0
        ep_collision = False
        ep_success = False

        frames = []
        metrics = _init_metrics(track_signals)

        while not done:
            ep_len += 1

            _render_step(env, frames)

            action = _compute_action(actor, obs)
            _record_actor_metrics(metrics, actor, action, unom_holder)

            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            _record_executed_action(metrics, env, action)

            if info.get("is_collision", False):
                ep_collision = True
            if info.get("is_success", False):
                ep_success = True

            ep_ret += rew

        yield ep_len, ep_ret, ep_collision, ep_success, frames, metrics


def eval_policy(
    policy,
    env,
    max_episodes=50,
    save_path=None,
    base_seed=None,
    method=None,
    visualize_episodes=20,
):
    total_episodes = 0
    success_count = 0
    collision_count = 0

    actor = policy
   
    mode = (method or "").lower()
    track_signals = ("cbf" in mode) or ("cvar" in mode)
    hook_handle = None
    unom_holder = None
    if track_signals:
        hook_handle, unom_holder = _setup_unom_hook(actor)

    try:
        for ep_num, (ep_len, ep_ret, ep_collision, ep_success, frames, metrics) in enumerate(
            rollout(
                actor,
                env,
                base_seed=base_seed,
                track_signals=track_signals,
                unom_holder=unom_holder,
            )
        ):
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
                        composed = _compose_frames(frames, metrics, mode, dt)
                        imageio.mimsave(full_path, composed, fps=10)
                    else:
                        imageio.mimsave(full_path, frames, fps=10)
                    print(f"Saved evaluation animation to {full_path}")

            total_episodes += 1
            if ep_success and not ep_collision:
                success_count += 1
            if ep_collision:
                collision_count += 1

            if total_episodes >= max_episodes:
                break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    print("\n\n-------------------- Evaluation Summary --------------------")
    print(f"Total Episodes: {total_episodes}")
    if total_episodes > 0:
        success_rate = success_count / total_episodes
        collision_rate = collision_count / total_episodes
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Collision Rate: {collision_rate * 100:.2f}%")
    else:
        success_rate = None
        collision_rate = None
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


def run_crossing_scenario(policy, env, save_path=None):
    actor = policy

    obs, _ = env.reset(options={"scenario": "crossing"})
    done = False
    frames = []
    is_collision = False
    is_success = False

    while not done:
        _render_step(env, frames)
        action = _compute_action(actor, obs)

        obs, _, terminated, truncated, info = env.step(action)
        if info.get("is_collision", False):
            is_collision = True
        if info.get("is_success", False):
            is_success = True
        done = terminated | truncated

    if save_path and len(frames) > 0:
        os.makedirs(save_path, exist_ok=True)
        imageio.mimsave(os.path.join(save_path, "crossing.gif"), frames, fps=10)

    print(f"Crossing scenario finished. Collision: {is_collision}, Success: {is_success}")
