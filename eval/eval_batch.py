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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import Config
from crowd_sim.utils import (
    build_env,
    relative_obs_dim_from_env_dim,
    dump_test_config,
)
from eval.eval_util import RLEvalActorAdapter, run_one_episode, resolve_episode_seed

FIXED_EVAL_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
EVAL_SUMMARY_FILENAME = "checkpoint_eval_all_multiseed.json"


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

        is_best = "best" in lower
        if "sac" in lower:
            algo = "sac"
        elif "ppo" in lower:
            algo = "ppo"
        else:
            algo = "actor"

        entries.append(
            {
                "path": path,
                "step": _extract_step(path),
                "algo": algo,
                "is_best": is_best,
            }
        )

    def _sort_key(item):
        step = item["step"] if item["step"] >= 0 else 10**18
        return (0 if item["is_best"] else 1, step, item["path"])

    entries.sort(key=_sort_key)
    return entries


def evaluate_actor(actor, env, episodes: int, base_seed: Optional[int]) -> Dict[str, Any]:
    returns = []
    lens = []
    success_hits = 0
    collision_hits = 0
    timeout_hits = 0
    infeasible_hits = 0

    for ep in range(episodes):
        seed = resolve_episode_seed(base_seed, ep)
        result = run_one_episode(
            actor=actor,
            env=env,
            seed=seed,
            track_signals=False,
            unom_holder=None,
            collect_frames=False,
        )

        returns.append(float(result["ep_ret"]))
        lens.append(int(result["ep_len"]))
        if bool(result["ep_success"]) and (not bool(result["ep_collision"])):
            success_hits += 1
        if bool(result["ep_collision"]):
            collision_hits += 1
        if bool(result["ep_timeout"]):
            timeout_hits += 1
        if bool(result.get("ep_infeasible", False)):
            infeasible_hits += 1

    total = max(episodes, 1)
    return {
        "total_episodes": episodes,
        "success_rate": success_hits / total,
        "collision_rate": collision_hits / total,
        "timeout_rate": timeout_hits / total,
        "infeasible_rate": infeasible_hits / total,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_ep_len": float(np.mean(lens)) if lens else 0.0,
    }


def _evaluate_over_seeds(actor, env, eval_seeds, episodes_per_seed: int):
    per_seed = []
    per_seed_full = []
    for s_idx, seed in enumerate(eval_seeds):
        metrics = evaluate_actor(actor, env, episodes=episodes_per_seed, base_seed=seed)
        row_seed = {
            "success_rate": metrics["success_rate"],
            "collision_rate": metrics["collision_rate"],
            "timeout_rate": metrics["timeout_rate"],
            "infeasible_rate": metrics["infeasible_rate"],
            "avg_return": metrics["avg_return"],
            "avg_ep_len": metrics["avg_ep_len"],
        }
        per_seed.append(row_seed)
        per_seed_full.append(metrics)
        print(
            f"  [{s_idx + 1}/{len(eval_seeds)}] seed={seed} "
            f"succ={metrics['success_rate']:.3f} coll={metrics['collision_rate']:.3f} infeas={metrics['infeasible_rate']:.3f} "
            f"ret={metrics['avg_return']:.3f} len={metrics['avg_ep_len']:.2f}"
        )

    agg = _aggregate_per_seed(per_seed_full)
    return per_seed, agg


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


def _print_eval_context(
    cfg: Config,
    method: str,
    env_name: str,
    is_rl_method: bool,
    num_checkpoints: int,
    num_seeds: int,
    episodes_per_seed: int,
):
    robot_type = getattr(cfg.robot, "type", "unknown")
    num_humans = getattr(cfg.human, "num_humans", "unknown")
    max_obs = getattr(cfg.env, "max_obstacles_obs", "unknown")
    mode = "rl" if is_rl_method else "controller"
    target = f"{num_checkpoints} checkpoints" if is_rl_method else "controller"
    print(
        f"[EvalConfig] mode={mode}, target={target}, method={method}, env={env_name}, "
        f"robot_type={robot_type}, obstacles={num_humans}, max_obstacles_obs={max_obs}, "
        f"seeds={num_seeds}, episodes/seed={episodes_per_seed}",
        flush=True,
    )


def _aggregate_per_seed(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_episodes = int(sum(r["total_episodes"] for r in per_seed))

    denom = max(total_episodes, 1)
    success_rate = (
        sum(float(r["success_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
    )
    collision_rate = (
        sum(float(r["collision_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
    )
    timeout_rate = (
        sum(float(r["timeout_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
    )
    infeasible_rate = (
        sum(float(r["infeasible_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
    )

    totals = {
        "total_episodes": total_episodes,
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "timeout_rate": float(timeout_rate),
        "infeasible_rate": float(infeasible_rate),
    }

    aggregate = {
        "success_rate": _summarize_metric([r["success_rate"] for r in per_seed]),
        "collision_rate": _summarize_metric([r["collision_rate"] for r in per_seed]),
        "timeout_rate": _summarize_metric([r["timeout_rate"] for r in per_seed]),
        "infeasible_rate": _summarize_metric([r["infeasible_rate"] for r in per_seed]),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_model",
        required=True,
        type=str,
        help="Run folder name under trained_models (e.g. 20260221_234406_unicycle_rl)",
    )
    parser.add_argument(
        "--episodes_per_seed",
        type=int,
        default=50,
        help="Episodes evaluated for each fixed seed (50..1000).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rl",
        help="Policy/controller method (e.g. rl, cbfqp, nominal).",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default=None,
        help="Optional robot type override: single_integrator / unicycle.",
    )
    parser.add_argument(
        "--num_humans",
        type=int,
        default=None,
        help="Optional override for config.human.num_humans.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(os.path.join("trained_models", args.actor_model))
    if _needs_rl_adapter(args.method) and not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes_per_seed must be > 0")

    eval_seeds = FIXED_EVAL_SEEDS

    device = torch.device("cpu")

    cfg = Config()
    if args.robot_type is not None:
        cfg.robot.type = args.robot_type
    if cfg.robot.type == "unicycle":
        cfg.robot.ini_goal_dist = 6.0
    elif cfg.robot.type == "single_integrator":
        cfg.robot.ini_goal_dist = 8.0
    if args.num_humans is not None:
        if args.num_humans <= 0:
            raise ValueError("--num_humans must be > 0")
        human_count = int(args.num_humans)
        cfg.human.num_humans = human_count

    method = args.method
    # cfg.env.rl_xy_to_unicycle = bool(method == "rl" and cfg.robot.type == "unicycle")
    env_name = cfg.env.get("name", "social_nav_var_num")

    is_rl_method = _needs_rl_adapter(method)
    checkpoints: List[Dict[str, Any]]
    if is_rl_method:
        checkpoints = discover_checkpoints(run_dir)
        if not checkpoints:
            raise RuntimeError(f"No actor checkpoints found in {run_dir}")
    else:
        checkpoints = [
            {
                "path": None,
                "step": -1,
                "algo": "controller",
                "is_best": True,
            }
        ]
    _print_eval_context(
        cfg=cfg,
        method=method,
        env_name=env_name,
        is_rl_method=is_rl_method,
        num_checkpoints=len(checkpoints),
        num_seeds=len(eval_seeds),
        episodes_per_seed=int(args.episodes_per_seed),
    )
    env = build_env(env_name, render_mode=None, config=cfg)

    dump_test_config(
        run_dir,
        cfg,
        hyperparameters={
            "env_name": env_name,
            "method": method,
            "eval_seeds": eval_seeds,
            "episodes_per_seed": int(args.episodes_per_seed),
            "device": str(device),
        },
        extra={
            "script": "eval/eval_batch.py",
            "config_source": "config.py",
        },
    )

    results: List[Dict[str, Any]] = []

    try:
        if is_rl_method:
            from crowd_nav.rl_policy_factory import get_rl_policy_class

            PolicyClass = get_rl_policy_class(method)
            obs_dim = relative_obs_dim_from_env_dim(env.observation_space.shape[0])
            act_dim = env.action_space.shape[0]
            policy_kwargs = _policy_kwargs_from_config(cfg)

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

                policy = PolicyClass(obs_dim, act_dim, **policy_kwargs).to(device)
                state = torch.load(ckpt_path, map_location=device)
                policy.load_state_dict(state)
                policy.eval()
                actor = RLEvalActorAdapter(policy, env.action_space, device)

                per_seed, agg = _evaluate_over_seeds(
                    actor=actor,
                    env=env,
                    eval_seeds=eval_seeds,
                    episodes_per_seed=args.episodes_per_seed,
                )
                row["totals"] = agg["totals"]
                row["aggregate"] = agg["aggregate"]
                row["per_seed"] = per_seed

                print(
                    "  aggregate: "
                    f"succ={row['aggregate']['success_rate']['mean']:.3f} "
                    f"coll={row['aggregate']['collision_rate']['mean']:.3f} "
                    f"ret={row['aggregate']['avg_return']['mean']:.3f}"
                )


                results.append(row)
        else:
            row = {
                "checkpoint_path": None,
                "checkpoint_file": f"controller:{method}",
                "step": -1,
            }
            from controller.robot_controller_factory import build_robot_controller

            actor = build_robot_controller(method, cfg, env)
            per_seed, agg = _evaluate_over_seeds(
                actor=actor,
                env=env,
                eval_seeds=eval_seeds,
                episodes_per_seed=args.episodes_per_seed,
            )
            row["totals"] = agg["totals"]
            row["aggregate"] = agg["aggregate"]
            row["per_seed"] = per_seed
            print(
                "  aggregate: "
                f"succ={row['aggregate']['success_rate']['mean']:.3f} "
                f"coll={row['aggregate']['collision_rate']['mean']:.3f} "
                f"ret={row['aggregate']['avg_return']['mean']:.3f}"
            )

            results.append(row)
    finally:
        env.close()

    best = _choose_best(results)

    summary = {
        "run_dir": run_dir,
        "method": method,
        "env_name": env_name,
        "config_source": "config.py",
        "device": str(device),
        "eval_seeds": eval_seeds,
        "episodes_per_seed": int(args.episodes_per_seed),
        "num_checkpoints": len(checkpoints),
        "best_checkpoint": best,
        "checkpoints": results,
    }

    out_path = os.path.join(run_dir, EVAL_SUMMARY_FILENAME)
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
