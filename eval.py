"""
Evaluate all actor checkpoints in one run directory using multiple seeds.
Default behavior: 10 seeds x 50 episodes per seed for every checkpoint.
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from config.config import Config
from crowd_sim.env.social_nav import SocialNav
from crowd_sim.env.social_nav_var_num import SocialNavVarNum
from crowd_sim.utils import absolute_obs_to_relative, relative_obs_dim_from_env_dim
from eval_policy import RLEvalActorAdapter
from crowd_nav.policy_utils import get_policy_class


def build_env(env_name: str, config: Config):
    name = (env_name or "").strip()
    if name in ("social_nav", "SocialNav"):
        return SocialNav(render_mode=None, config_file=config)
    if name in ("social_nav_var_num", "SocialNavVarNum"):
        return SocialNavVarNum(render_mode=None, config_file=config)
    return gym.make(name, render_mode=None, config_file=config)


def _set_config_from_saved(config: Config, saved_cfg: Dict[str, Any]) -> None:
    for section_name in ("env", "human", "robot", "controller", "reward"):
        section_dict = saved_cfg.get(section_name, {})
        section = getattr(config, section_name)
        for key, value in section_dict.items():
            section[key] = value


def _extract_step(path: str) -> int:
    base = os.path.basename(path).lower()
    m = re.search(r"(?:^|[_-])step[_-]?(\d+)", base)
    return int(m.group(1)) if m else -1


def discover_checkpoints(run_dir: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    all_ckpts = sorted(glob.glob(os.path.join(run_dir, "*.pth")))

    for path in all_ckpts:
        fname = os.path.basename(path)
        lower = fname.lower()

        # Keep only actor checkpoints; explicitly skip critic and optimizer artifacts.
        if "actor" not in lower:
            continue
        if "critic" in lower:
            continue
        if any(tok in lower for tok in ("optim", "optimizer", "sched", "scaler")):
            continue

        is_ema = "ema" in lower

        is_best = "best" in lower
        if "sac" in lower:
            algo = "sac"
        elif "ppo" in lower:
            algo = "ppo"
        else:
            algo = "actor"

        kind = f"{algo}_{'ema' if is_ema else 'raw'}"
        if is_best:
            kind += "_best"

        entries.append(
            {
                "path": path,
                "step": _extract_step(path),
                "kind": kind,
                "algo": algo,
                "is_ema": is_ema,
                "is_best": is_best,
            }
        )

    def _sort_key(item):
        step = item["step"] if item["step"] >= 0 else 10**18
        kind_rank = 0 if item["is_ema"] else 1
        return (0 if item["is_best"] else 1, step, kind_rank, item["path"])

    entries.sort(key=_sort_key)
    return entries


def _to_action(actor, obs):
    obs_rel = absolute_obs_to_relative(obs)

    if hasattr(actor, "deterministic"):
        actor.deterministic = True

    if hasattr(actor, "get_action"):
        out = actor.get_action(obs_rel)
    elif isinstance(actor, torch.nn.Module):
        with torch.no_grad():
            dev = next(actor.parameters()).device
            obs_t = torch.as_tensor(obs_rel, dtype=torch.float32, device=dev).unsqueeze(0)
            out = actor(obs_t)
            if torch.is_tensor(out) and out.ndim >= 2:
                out = out[0]
    else:
        out = actor(obs_rel)

    action = out[0] if isinstance(out, (tuple, list)) else out
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


def evaluate_actor(actor, env, episodes: int, base_seed: Optional[int]) -> Dict[str, Any]:
    returns = []
    lens = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    for ep in range(episodes):
        if base_seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=base_seed + ep)

        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = False
        ep_collision = False
        ep_timeout = False

        while not done:
            action = _to_action(actor, obs)
            obs, rew, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(rew)
            ep_len += 1
            ep_success = ep_success or bool(info.get("is_success", False))
            ep_collision = ep_collision or bool(info.get("is_collision", False))
            ep_timeout = ep_timeout or bool(info.get("is_timeout", False))

        returns.append(ep_ret)
        lens.append(ep_len)
        if ep_success and not ep_collision:
            success_count += 1
        if ep_collision:
            collision_count += 1
        if ep_timeout:
            timeout_count += 1

    total = max(episodes, 1)
    return {
        "total_episodes": episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / total,
        "collision_rate": collision_count / total,
        "timeout_rate": timeout_count / total,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_ep_len": float(np.mean(lens)) if lens else 0.0,
    }


def _parse_seed_list(seed_text: str) -> List[int]:
    parts = [p.strip() for p in seed_text.split(",") if p.strip()]
    if not parts:
        raise ValueError(
            "Empty --eval_seeds. Example: --eval_seeds 100,200,300,400,500,600,700,800,900,1000"
        )
    seeds = [int(p) for p in parts]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--eval_seeds contains duplicate values.")
    return seeds


def _policy_kwargs_from_config(cfg: Config) -> Dict[str, Any]:
    safe_dist = cfg.controller.safety_margin + cfg.human.radius + cfg.robot.radius
    return {
        "robot_type": cfg.robot.type,
        "safe_dist": safe_dist,
        "alpha": cfg.controller.cbf_alpha,
        "beta": cfg.controller.cvar_beta,
        "vmax": cfg.robot.vmax,
        "amax": cfg.robot.amax,
        "omega_max": cfg.robot.omega_max,
    }


def _summarize_metric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": float(np.min(arr)) if n else 0.0,
        "max": float(np.max(arr)) if n else 0.0,
    }


def _needs_rl_adapter(method: str) -> bool:
    return (method or "").lower().startswith("rl")


def _aggregate_per_seed(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_episodes = int(sum(r["total_episodes"] for r in per_seed))
    success_count = int(sum(r["success_count"] for r in per_seed))
    collision_count = int(sum(r["collision_count"] for r in per_seed))
    timeout_count = int(sum(r["timeout_count"] for r in per_seed))

    denom = max(total_episodes, 1)
    totals = {
        "total_episodes": total_episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / denom,
        "collision_rate": collision_count / denom,
        "timeout_rate": timeout_count / denom,
    }

    aggregate = {
        "success_rate": _summarize_metric([r["success_rate"] for r in per_seed]),
        "collision_rate": _summarize_metric([r["collision_rate"] for r in per_seed]),
        "timeout_rate": _summarize_metric([r["timeout_rate"] for r in per_seed]),
        "avg_return": _summarize_metric([r["avg_return"] for r in per_seed]),
        "avg_ep_len": _summarize_metric([r["avg_ep_len"] for r in per_seed]),
    }

    return {"totals": totals, "aggregate": aggregate}


def _choose_best(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    def _key(row: Dict[str, Any]):
        agg = row["aggregate"]
        return (
            agg["success_rate"]["mean"],
            -agg["collision_rate"]["mean"],
            agg["avg_return"]["mean"],
        )

    return max(valid, key=_key)


def _load_run_config(run_dir: str) -> Dict[str, Any]:
    run_cfg_path = os.path.join(run_dir, "train_config.json")
    if not os.path.isfile(run_cfg_path):
        raise FileNotFoundError(f"train_config.json not found: {run_cfg_path}")

    with open(run_cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_model",
        required=True,
        type=str,
        help="Run folder name under trained_models/default (e.g. 20260221_234406_unicycle_rl)",
    )
    parser.add_argument(
        "--eval_seeds",
        type=str,
        default="100,200,300,400,500,600,700,800,900,1000",
        help="Comma-separated base seeds. Default uses 10 distinct seeds.",
    )
    parser.add_argument(
        "--episodes_per_seed",
        type=int,
        default=50,
        help="Episodes evaluated for each seed on each checkpoint.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="checkpoint_eval_all_multiseed.json",
        help="Output JSON filename in run directory.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_dir = os.path.abspath(os.path.join("trained_models", "default", args.actor_model))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes_per_seed must be > 0")

    eval_seeds = _parse_seed_list(args.eval_seeds)
    run_cfg = _load_run_config(run_dir)

    method = run_cfg.get("args", {}).get("method", "rl")
    env_name = run_cfg.get("args", {}).get("env_name", "social_nav_var_num")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    cfg = Config()
    _set_config_from_saved(cfg, run_cfg.get("config", {}))
    env = build_env(env_name, cfg)

    PolicyClass = get_policy_class(method)
    obs_dim = relative_obs_dim_from_env_dim(env.observation_space.shape[0])
    act_dim = env.action_space.shape[0]
    policy_kwargs = _policy_kwargs_from_config(cfg)

    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"No actor checkpoints found in {run_dir}")

    print(
        f"Evaluating {len(checkpoints)} checkpoints, method={method}, env={env_name}, "
        f"seeds={len(eval_seeds)}, episodes/seed={args.episodes_per_seed}"
    )

    results: List[Dict[str, Any]] = []

    try:
        for ckpt_idx, ckpt in enumerate(checkpoints):
            ckpt_path = ckpt["path"]
            step = ckpt["step"]
            print(
                f"\n[{ckpt_idx + 1}/{len(checkpoints)}] step={step} "
                f"file={os.path.basename(ckpt_path)}"
            )

            row: Dict[str, Any] = {
                "checkpoint_path": ckpt_path,
                "checkpoint_file": os.path.basename(ckpt_path),
                "step": step,
            }

            try:
                policy = PolicyClass(obs_dim, act_dim, **policy_kwargs).to(device)
                state = torch.load(ckpt_path, map_location=device)
                policy.load_state_dict(state)
                policy.eval()

                actor = RLEvalActorAdapter(policy, env.action_space, device) if _needs_rl_adapter(method) else policy

                per_seed = []
                per_seed_full = []
                for s_idx, seed in enumerate(eval_seeds):
                    metrics = evaluate_actor(actor, env, episodes=args.episodes_per_seed, base_seed=seed)
                    row_seed = {
                        "success_rate": metrics["success_rate"],
                        "collision_rate": metrics["collision_rate"],
                        "timeout_rate": metrics["timeout_rate"],
                        "avg_return": metrics["avg_return"],
                        "avg_ep_len": metrics["avg_ep_len"],
                    }
                    per_seed.append(row_seed)
                    per_seed_full.append(metrics)
                    print(
                        f"  [{s_idx + 1}/{len(eval_seeds)}] seed={seed} "
                        f"succ={metrics['success_rate']:.3f} coll={metrics['collision_rate']:.3f} "
                        f"ret={metrics['avg_return']:.3f} len={metrics['avg_ep_len']:.2f}"
                    )

                agg = _aggregate_per_seed(per_seed_full)
                row["per_seed"] = per_seed
                row["totals"] = agg["totals"]
                row["aggregate"] = agg["aggregate"]

                print(
                    "  aggregate: "
                    f"succ={row['aggregate']['success_rate']['mean']:.3f} "
                    f"coll={row['aggregate']['collision_rate']['mean']:.3f} "
                    f"ret={row['aggregate']['avg_return']['mean']:.3f}"
                )
            except Exception as exc:
                row["error"] = str(exc)
                print(f"  failed: {exc}")

            results.append(row)
    finally:
        env.close()

    best = _choose_best(results)

    summary = {
        "run_dir": run_dir,
        "method": method,
        "env_name": env_name,
        "device": str(device),
        "eval_seeds": eval_seeds,
        "episodes_per_seed": int(args.episodes_per_seed),
        "num_checkpoints": len(checkpoints),
        "best_checkpoint": best,
        "checkpoints": results,
    }

    out_path = os.path.join(run_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved: {out_path}")
    if best is not None:
        s = best["aggregate"]["success_rate"]["mean"]
        c = best["aggregate"]["collision_rate"]["mean"]
        r = best["aggregate"]["avg_return"]["mean"]
        print(f"Best checkpoint: {best['checkpoint_file']} (succ={s:.3f}, coll={c:.3f}, ret={r:.3f})")
    else:
        print("No valid checkpoint result (all failed).")


if __name__ == "__main__":
    main()
