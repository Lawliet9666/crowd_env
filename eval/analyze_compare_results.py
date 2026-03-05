#!/usr/bin/env python3
"""
Analyze compare results from trained_models/compare.

Outputs:
1) Summary CSV + Markdown report
2) Success-rate curves (mean +/- std)
3) Radar charts for selected metrics (6/1/3/4/5 only)
   - 6: Robustness (seed stability)
   - 1: Success rate
   - 3: Feasibility (= 1 - infeasible rate)
   - 4: Efficiency (inverse of trajectory time)
   - 5: Compute time (inverse)

Missing metrics are patched to neutral score 50 to keep plotting robust.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


POLICY_DISPLAY = {
    "orca": "ORCA",
    "cbfqp": "CBF-QP",
    "cvarqp": "CVaR-BF-QP",
    "adapcvarqp": "Adaptive-CVaR-BF-QP",
    "drcvarqp": "DR-CVaR-BF-QP",
    "rl": "RL",
    "rlcbfgamma": "RL-CBF-QP",
    "rlcvarbetaradius": "RL-CVaR-BF-QP",
}

TYPE_DISPLAY = {
    "orca": "Optimization",
    "cbfqp": "Optimization",
    "cvarqp": "Optimization",
    "adapcvarqp": "Optimization",
    "drcvarqp": "Optimization",
    "rl": "Learning",
    "rlcbfgamma": "Learning",
    "rlcvarbetaradius": "Learning",
}

METHOD_ORDER = [
    "orca",
    "cbfqp",
    "cvarqp",
    "adapcvarqp",
    "drcvarqp",
    "rl",
    "rlcbfgamma",
    "rlcvarbetaradius",
]
FRIENDLY_SEED_METHOD = "rlcvarbetaradius"
METHOD_MARKERS = {
    "orca": "o",
    "cbfqp": "s",
    "cvarqp": "^",
    "adapcvarqp": "D",
    "drcvarqp": "P",
    "rl": "X",
    "rlcbfgamma": "v",
    "rlcvarbetaradius": "*",
}

# Radar uses ONLY user-selected metrics: 6, 1, 3, 4, 5.
RADAR_AXES = [
    ("robustness", "Robustness (6)"),
    ("success", "Success Rate (1)"),
    ("feasibility", "Feasibility (3)"),
    ("efficiency", "Efficiency / Traj Time (4)"),
    ("compute", "Compute Time (5)"),
]

COMMON_SETTING_PATHS = [
    "env.name",
    "env.dt",
    "env.max_steps",
    "env.max_obstacles_obs",
    "env.sensing_radius",
    "env.normalize_obs",
    "human.policy",
    "human.num_humans",
    "human.vmax",
    "human.radius",
    "human.use_gmm",
    "robot.type",
    "robot.vmax",
    "robot.radius",
    "robot.omega_max",
    "robot.ini_goal_dist",
    "controller.cbf_alpha",
    "controller.cvar_beta",
    "controller.safety_margin",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_root", type=str, default="trained_models/compare")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def parse_scenario_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    if "_obs_" not in name:
        return None, None
    robot_type, obs_str = name.rsplit("_obs_", 1)
    try:
        obstacle_count = int(obs_str)
    except Exception:
        obstacle_count = None
    return robot_type, obstacle_count


def robot_display(robot_type: Optional[str]) -> str:
    if robot_type == "single_integrator":
        return "Single Integrator"
    if robot_type == "unicycle":
        return "Unicycle"
    return robot_type or "Unknown"


def method_sort_key(method: str) -> Tuple[int, str]:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), method
    return 999, method


def metric_tuple(aggregate: Dict, key_candidates: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    for key in key_candidates:
        metric = aggregate.get(key)
        if isinstance(metric, dict):
            return to_float(metric.get("mean")), to_float(metric.get("std")), to_float(metric.get("ci95"))
    return None, None, None


def get_by_path(obj: Dict, dotted_path: str):
    cur = obj
    for k in dotted_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def fmt_common_value(v) -> str:
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _friendly_seed_rank_key(row: Dict) -> Tuple[float, float, float, float, float, float]:
    sr = row.get("success_rate_mean")
    fr = row.get("infeasible_rate_mean")
    cr = row.get("collision_rate_mean")
    tr = row.get("timeout_rate_mean")
    ret = row.get("avg_return_mean")
    ep = row.get("avg_ep_len_mean")
    # Better rank: high SR, then low FR/CR/TR, then high return, then short ep length.
    return (
        -(float(sr) if sr is not None else -1.0),
        float(fr) if fr is not None else 1.0,
        float(cr) if cr is not None else 1.0,
        float(tr) if tr is not None else 1.0,
        -(float(ret) if ret is not None else -1e9),
        float(ep) if ep is not None else 1e9,
    )


def _summarize_seed_entries(entries: List[Dict]) -> Dict:
    return {
        "support": len(entries),
        "success_rate_mean": _safe_mean([e.get("success_rate") for e in entries]),
        "collision_rate_mean": _safe_mean([e.get("collision_rate") for e in entries]),
        "timeout_rate_mean": _safe_mean([e.get("timeout_rate") for e in entries]),
        "infeasible_rate_mean": _safe_mean([e.get("infeasible_rate") for e in entries]),
        "avg_return_mean": _safe_mean([e.get("avg_return") for e in entries]),
        "avg_ep_len_mean": _safe_mean([e.get("avg_ep_len") for e in entries]),
    }


def _top_seeds_for_group(entries: List[Dict], top_k: int = 2) -> List[Dict]:
    per_seed = defaultdict(list)
    for e in entries:
        seed = e.get("seed")
        if seed is None:
            continue
        per_seed[int(seed)].append(e)

    if not per_seed:
        return []

    candidates = []
    for seed, seed_entries in per_seed.items():
        summary = _summarize_seed_entries(seed_entries)
        summary["seed"] = seed
        candidates.append(summary)

    candidates.sort(key=_friendly_seed_rank_key)
    return candidates[: max(1, int(top_k))]


def collect_friendly_seed_summary(compare_root: Path, method: str = FRIENDLY_SEED_METHOD, top_k: int = 2) -> Dict:
    records: List[Dict] = []

    for sdir in sorted([p for p in compare_root.iterdir() if p.is_dir()]):
        scenario = sdir.name
        robot_type, obstacle_count = parse_scenario_name(scenario)
        if robot_type is None:
            continue

        summary_path = sdir / method / "checkpoint_eval_all_multiseed.json"
        if not summary_path.exists():
            continue

        payload = load_json(summary_path)
        best = payload.get("best_checkpoint") if isinstance(payload, dict) else None
        if not isinstance(best, dict):
            continue

        per_seed = best.get("per_seed")
        eval_seeds = payload.get("eval_seeds")
        if not isinstance(per_seed, list):
            continue
        if not isinstance(eval_seeds, list):
            eval_seeds = []

        for idx, seed_metrics in enumerate(per_seed):
            if not isinstance(seed_metrics, dict):
                continue
            seed_val = eval_seeds[idx] if idx < len(eval_seeds) else idx
            records.append(
                {
                    "scenario": scenario,
                    "robot_type": robot_type,
                    "obstacle_count": obstacle_count,
                    "seed": int(seed_val),
                    "success_rate": to_float(seed_metrics.get("success_rate")),
                    "collision_rate": to_float(seed_metrics.get("collision_rate")),
                    "timeout_rate": to_float(seed_metrics.get("timeout_rate")),
                    "infeasible_rate": to_float(seed_metrics.get("infeasible_rate")),
                    "avg_return": to_float(seed_metrics.get("avg_return")),
                    "avg_ep_len": to_float(seed_metrics.get("avg_ep_len")),
                }
            )

    by_robot_obstacle = []

    grouped_robot_obs = defaultdict(list)
    for r in records:
        robot_type = r.get("robot_type")
        obstacle_count = r.get("obstacle_count")
        if robot_type is None:
            continue
        if obstacle_count is None:
            continue
        grouped_robot_obs[(robot_type, int(obstacle_count))].append(r)

    for robot_type, obstacle_count in sorted(grouped_robot_obs.keys(), key=lambda x: (x[0], x[1])):
        top_seeds = _top_seeds_for_group(grouped_robot_obs[(robot_type, obstacle_count)], top_k=top_k)
        if not top_seeds:
            continue
        row = {
            "robot_type": robot_type,
            "obstacle_count": obstacle_count,
            "top_seeds": top_seeds,
        }
        by_robot_obstacle.append(row)

    return {
        "method": method,
        "top_k": int(top_k),
        "records_count": len(records),
        "by_robot_obstacle": by_robot_obstacle,
    }


# -----------------------------------------------------------------------------
# 1) Collection / Scan
# -----------------------------------------------------------------------------

def collect_common_settings(compare_root: Path) -> Dict:
    config_files = sorted(compare_root.glob("**/test_config.json"))
    configs: List[Dict] = []
    for path in config_files:
        payload = load_json(path)
        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        if isinstance(cfg, dict):
            configs.append(cfg)

    common = {}
    for dotted in COMMON_SETTING_PATHS:
        values = [get_by_path(cfg, dotted) for cfg in configs]
        if not values or any(v is None for v in values):
            continue
        first = values[0]
        if all(v == first for v in values[1:]):
            common[dotted] = first

    return {
        "num_test_configs": len(configs),
        "common": common,
    }


def collect_compare_rows(compare_root: Path) -> Tuple[Dict[str, Dict], List[Dict]]:
    scenario_info: Dict[str, Dict] = {}
    rows: List[Dict] = []

    scenario_dirs = sorted([p for p in compare_root.iterdir() if p.is_dir()])
    for sdir in scenario_dirs:
        scenario = sdir.name
        robot_type, obstacle_count = parse_scenario_name(scenario)
        if robot_type is None:
            continue

        info = {
            "scenario": scenario,
            "robot_type": robot_type,
            "obstacle_count": obstacle_count,
            "env_name": None,
            "episodes_per_seed": None,
            "num_eval_seeds": None,
            "max_obstacles_obs": None,
            "num_humans": None,
            "ini_goal_dist": None,
            "env_dt": None,
        }

        test_cfg_candidates = sorted(sdir.glob("*/test_config.json"))
        if test_cfg_candidates:
            test_cfg = load_json(test_cfg_candidates[0])
            hyp = test_cfg.get("hyperparameters", {}) if isinstance(test_cfg, dict) else {}
            cfg = test_cfg.get("config", {}) if isinstance(test_cfg, dict) else {}
            env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
            human_cfg = cfg.get("human", {}) if isinstance(cfg, dict) else {}
            robot_cfg = cfg.get("robot", {}) if isinstance(cfg, dict) else {}

            eval_seeds = hyp.get("eval_seeds")
            info["env_name"] = hyp.get("env_name")
            info["episodes_per_seed"] = hyp.get("episodes_per_seed")
            info["num_eval_seeds"] = len(eval_seeds) if isinstance(eval_seeds, list) else None
            info["max_obstacles_obs"] = env_cfg.get("max_obstacles_obs")
            info["num_humans"] = human_cfg.get("num_humans")
            info["ini_goal_dist"] = robot_cfg.get("ini_goal_dist")
            info["env_dt"] = to_float(env_cfg.get("dt"))

        scenario_info[scenario] = info

        method_dirs = sorted([p for p in sdir.iterdir() if p.is_dir()])
        for mdir in method_dirs:
            method = mdir.name
            row = {
                "scenario": scenario,
                "robot_type": robot_type,
                "obstacle_count": obstacle_count,
                "method": method,
                "type": TYPE_DISPLAY.get(method, "Unknown"),
                "policy": POLICY_DISPLAY.get(method, method),
                "status": "no_summary",
                "num_checkpoints": None,
                "valid_checkpoints": None,
                "best_checkpoint_file": None,
                "error": None,
                # Main eval metrics.
                "sr_mean": None,
                "sr_std": None,
                "sr_ci95": None,
                "cr_mean": None,
                "cr_std": None,
                "cr_ci95": None,
                "tr_mean": None,
                "tr_std": None,
                "tr_ci95": None,
                "fr_mean": None,
                "fr_std": None,
                "fr_ci95": None,
                "avg_return_mean": None,
                "avg_return_std": None,
                "avg_return_ci95": None,
                "avg_ep_len_mean": None,
                "avg_ep_len_std": None,
                "avg_ep_len_ci95": None,
                # Optional future metrics (patched to handle missing now).
                "traj_time_mean": None,
                "traj_time_std": None,
                "traj_time_ci95": None,
                "compute_time_mean": None,
                "compute_time_std": None,
                "compute_time_ci95": None,
                "robustness_mean": None,
                "robustness_std": None,
                "robustness_ci95": None,
            }

            summary_path = mdir / "checkpoint_eval_all_multiseed.json"
            if not summary_path.exists():
                rows.append(row)
                continue

            payload = load_json(summary_path)
            checkpoints = payload.get("checkpoints", []) if isinstance(payload, dict) else []
            valid_ckpts = [c for c in checkpoints if isinstance(c, dict) and "error" not in c]
            row["num_checkpoints"] = len(checkpoints)
            row["valid_checkpoints"] = len(valid_ckpts)

            best = payload.get("best_checkpoint") if isinstance(payload, dict) else None
            if not isinstance(best, dict):
                row["status"] = "all_failed"
                for ck in checkpoints:
                    if isinstance(ck, dict) and "error" in ck:
                        row["error"] = str(ck.get("error", "")).replace("\n", " ")
                        break
                rows.append(row)
                continue

            row["status"] = "ok"
            row["best_checkpoint_file"] = best.get("checkpoint_file")
            aggregate = best.get("aggregate", {}) if isinstance(best, dict) else {}

            row["sr_mean"], row["sr_std"], row["sr_ci95"] = metric_tuple(aggregate, ["success_rate", "sr"])
            row["cr_mean"], row["cr_std"], row["cr_ci95"] = metric_tuple(aggregate, ["collision_rate", "cr"])
            row["tr_mean"], row["tr_std"], row["tr_ci95"] = metric_tuple(aggregate, ["timeout_rate", "tr"])
            row["fr_mean"], row["fr_std"], row["fr_ci95"] = metric_tuple(aggregate, ["infeasible_rate", "fr"])
            row["avg_return_mean"], row["avg_return_std"], row["avg_return_ci95"] = metric_tuple(
                aggregate,
                ["avg_return", "return"],
            )
            row["avg_ep_len_mean"], row["avg_ep_len_std"], row["avg_ep_len_ci95"] = metric_tuple(
                aggregate,
                ["avg_ep_len", "ep_len", "episode_len"],
            )

            row["traj_time_mean"], row["traj_time_std"], row["traj_time_ci95"] = metric_tuple(
                aggregate,
                ["avg_traj_time", "traj_time", "trajectory_time", "avg_travel_time"],
            )
            row["compute_time_mean"], row["compute_time_std"], row["compute_time_ci95"] = metric_tuple(
                aggregate,
                [
                    "avg_compute_time",
                    "compute_time",
                    "avg_step_compute_time",
                    "avg_inference_time",
                    "avg_solver_time",
                ],
            )
            row["robustness_mean"], row["robustness_std"], row["robustness_ci95"] = metric_tuple(
                aggregate,
                ["robustness", "robustness_seed", "seed_robustness", "stability"],
            )

            rows.append(row)

    return scenario_info, rows


# -----------------------------------------------------------------------------
# 2) Export summary (CSV / Markdown)
# -----------------------------------------------------------------------------

def fmt_pm(mean: Optional[float], ci95: Optional[float], digits: int = 3) -> str:
    if mean is None:
        return "N/A"
    m = float(mean)
    c = 0.0 if ci95 is None else float(ci95)
    if abs(m) < 1e-12 and abs(c) < 1e-12:
        return "0"
    return f"{m:.{digits}f} +/- {c:.{digits}f}"


def fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "N/A"
    x = float(v)
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.{digits}f}"


def write_summary_csv(rows: List[Dict], path: Path):
    fieldnames = [
        "scenario",
        "obstacle_count",
        "robot_type",
        "method",
        "type",
        "policy",
        "status",
        "num_checkpoints",
        "valid_checkpoints",
        "best_checkpoint_file",
        "sr_mean",
        "sr_std",
        "sr_ci95",
        "cr_mean",
        "cr_std",
        "cr_ci95",
        "tr_mean",
        "tr_std",
        "tr_ci95",
        "fr_mean",
        "fr_std",
        "fr_ci95",
        "avg_return_mean",
        "avg_return_std",
        "avg_return_ci95",
        "avg_ep_len_mean",
        "avg_ep_len_std",
        "avg_ep_len_ci95",
        "traj_time_mean",
        "traj_time_std",
        "traj_time_ci95",
        "compute_time_mean",
        "compute_time_std",
        "compute_time_ci95",
        "robustness_mean",
        "robustness_std",
        "robustness_ci95",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_markdown(
    scenario_info: Dict[str, Dict],
    rows: List[Dict],
    common_settings: Dict,
    friendly_seed_summary: Dict,
    path: Path,
):
    rows_by_obs = defaultdict(list)
    for row in rows:
        rows_by_obs[row["obstacle_count"]].append(row)

    for obs in rows_by_obs:
        rows_by_obs[obs].sort(
            key=lambda r: (
                0 if r["type"] == "Optimization" else 1 if r["type"] == "Learning" else 2,
                method_sort_key(r["method"]),
                0 if r["robot_type"] == "unicycle" else 1,
            )
        )

    with path.open("w", encoding="utf-8") as f:
        f.write("# Compare Summary\n\n")

        f.write("## Common Settings (Shared Across All Scenarios)\n\n")
        cfg_count = int(common_settings.get("num_test_configs", 0))
        common = common_settings.get("common", {})
        f.write(f"- Parsed from `{cfg_count}` test_config files.\n")
        if common:
            for key in sorted(common.keys()):
                f.write(f"- `{key}`: `{fmt_common_value(common[key])}`\n")
        else:
            f.write("- No common fields found in selected basic settings.\n")
        f.write("\n")

        f.write("## Scenario Info\n\n")
        f.write(
            "| Scenario | Robot | Obstacles | Env | Episodes/Seed | #Seeds | "
            "max_obstacles_obs | num_humans | ini_goal_dist |\n"
        )
        f.write("|---|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for scenario in sorted(scenario_info.keys()):
            info = scenario_info[scenario]
            f.write(
                f"| {scenario} | {robot_display(info['robot_type'])} | {info['obstacle_count']} | "
                f"{info['env_name'] or 'N/A'} | {info['episodes_per_seed'] or 'N/A'} | "
                f"{info['num_eval_seeds'] or 'N/A'} | {info['max_obstacles_obs'] or 'N/A'} | "
                f"{info['num_humans'] or 'N/A'} | {info['ini_goal_dist'] or 'N/A'} |\n"
            )
        f.write("\n")

        f.write(f"## Friendly Seed for `{friendly_seed_summary.get('method', FRIENDLY_SEED_METHOD)}`\n\n")
        if int(friendly_seed_summary.get("records_count", 0)) <= 0:
            f.write("- No per-seed data found for this method.\n\n")
        else:
            f.write(
                "Ranking rule: higher `success_rate`, then lower `infeasible/collision/timeout`, "
                "then higher `avg_return`.\n\n"
            )

            by_robot_obstacle = friendly_seed_summary.get("by_robot_obstacle", [])
            if by_robot_obstacle:
                top_k = int(friendly_seed_summary.get("top_k", 2))
                f.write(f"### Top-{top_k} Seeds by Robot Type and Obstacle Count (SR/FR Only)\n\n")
                f.write("| Robot | Obstacle | Rank | Seed | SR | FR | Avg Ep Len |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|\n")
                for row in by_robot_obstacle:
                    for rank_idx, seed_row in enumerate(row.get("top_seeds", []), start=1):
                        f.write(
                            f"| {robot_display(row['robot_type'])} | {row['obstacle_count']} | {rank_idx} | "
                            f"{seed_row.get('seed', 'N/A')} | "
                            f"{fmt_num(seed_row.get('success_rate_mean'))} | "
                            f"{fmt_num(seed_row.get('infeasible_rate_mean'))} | "
                            f"{fmt_num(seed_row.get('avg_ep_len_mean'))} |\n"
                        )
                f.write("\n")

        for obs in sorted(k for k in rows_by_obs.keys() if k is not None):
            f.write(f"## Obstacle = {obs}\n\n")
            grouped_by_robot = defaultdict(list)
            for row in rows_by_obs[obs]:
                grouped_by_robot[row["robot_type"]].append(row)

            preferred_order = ["single_integrator", "unicycle"]
            ordered_robot_types = [rt for rt in preferred_order if rt in grouped_by_robot]
            ordered_robot_types.extend(
                sorted([rt for rt in grouped_by_robot.keys() if rt not in preferred_order])
            )

            for robot_type in ordered_robot_types:
                part_rows = sorted(
                    grouped_by_robot[robot_type],
                    key=lambda r: (
                        0 if r["type"] == "Optimization" else 1 if r["type"] == "Learning" else 2,
                        method_sort_key(r["method"]),
                    ),
                )
                f.write(f"### {robot_display(robot_type)}\n\n")
                f.write("| Type | Policy | SR | CR | TR | FR | Avg Return | Avg Ep Len |\n")
                f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
                for row in part_rows:
                    f.write(
                        f"| {row['type']} | {row['policy']} | "
                        f"{fmt_pm(row['sr_mean'], row['sr_ci95'])} | "
                        f"{fmt_pm(row['cr_mean'], row['cr_ci95'])} | "
                        f"{fmt_pm(row['tr_mean'], row['tr_ci95'])} | "
                        f"{fmt_pm(row['fr_mean'], row['fr_ci95'])} | "
                        f"{fmt_pm(row['avg_return_mean'], row['avg_return_ci95'])} | "
                        f"{fmt_pm(row['avg_ep_len_mean'], row['avg_ep_len_ci95'])} |\n"
                    )
                f.write("\n")


# -----------------------------------------------------------------------------
# 3) Plot success-rate curves
# -----------------------------------------------------------------------------

def build_sr_series(rows: List[Dict], robot_type: str) -> Dict[str, List[Tuple[int, float, float]]]:
    series = defaultdict(list)
    for row in rows:
        if row["robot_type"] != robot_type or row["status"] != "ok":
            continue
        obs = row["obstacle_count"]
        sr = row["sr_mean"]
        if obs is None or sr is None:
            continue
        sr_std = 0.0 if row["sr_std"] is None else float(row["sr_std"])
        series[row["method"]].append((int(obs), float(sr), sr_std))

    for method in list(series.keys()):
        series[method].sort(key=lambda x: x[0])
    return series


def plot_success_rate_curves(rows: List[Dict], out_dir: Path, dpi: int) -> List[Path]:
    robot_types = sorted({r["robot_type"] for r in rows if r["robot_type"]})
    saved: List[Path] = []

    for robot_type in robot_types:
        series = build_sr_series(rows, robot_type)
        if not series:
            continue

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        y_low_all = []
        y_high_all = []

        for method in METHOD_ORDER:
            pts = series.get(method, [])
            if not pts:
                continue

            xs = [x for x, _, _ in pts]
            ys = [y for _, y, _ in pts]
            ss = [s for _, _, s in pts]

            style = "-" if TYPE_DISPLAY.get(method) == "Optimization" else "--"
            label = POLICY_DISPLAY.get(method, method)
            marker = METHOD_MARKERS.get(method, "o")
            line = ax.plot(
                xs,
                ys,
                marker=marker,
                linewidth=2.0,
                linestyle=style,
                label=label,
            )[0]
            color = line.get_color()

            low = [max(0.0, y - s) for y, s in zip(ys, ss)]
            high = [min(1.0, y + s) for y, s in zip(ys, ss)]
            y_low_all.extend(low)
            y_high_all.extend(high)
            ax.fill_between(xs, low, high, color=color, alpha=0.18)

        ax.set_title(f"Success Rate vs Obstacle Count ({robot_display(robot_type)})")
        ax.set_xlabel("Obstacle Count")
        ax.set_ylabel("Success Rate")
        if y_low_all and y_high_all:
            y_min = float(min(y_low_all))
            y_max = float(max(y_high_all))
            span = y_max - y_min
            if span < 1e-6:
                pad = 0.03
            else:
                pad = max(0.02, 0.12 * span)
            lo = max(0.0, y_min - pad)
            hi = min(1.0, y_max + pad)
            # Keep a minimal visible window to avoid over-zoom jitter.
            if hi - lo < 0.08:
                c = 0.5 * (lo + hi)
                lo = max(0.0, c - 0.04)
                hi = min(1.0, c + 0.04)
            ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="best", fontsize=9, frameon=True)

        out_path = out_dir / f"success_rate_vs_obstacles_{robot_type}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        saved.append(out_path)

    return saved


# -----------------------------------------------------------------------------
# 4) Plot radar charts (metrics 6/1/3/4/5 only)
# -----------------------------------------------------------------------------

def _clamp01_to_100(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return 100.0 * float(np.clip(v, 0.0, 1.0))


def _inverse_minmax_scores(raw_by_method: Dict[str, Optional[float]], neutral: float = 50.0) -> Tuple[Dict[str, float], bool]:
    methods = list(raw_by_method.keys())
    valid_vals = [v for v in raw_by_method.values() if v is not None]
    if not valid_vals:
        return {m: neutral for m in methods}, True

    vmin, vmax = min(valid_vals), max(valid_vals)
    if abs(vmax - vmin) < 1e-12:
        return {m: neutral for m in methods}, False

    scores = {}
    for method, raw in raw_by_method.items():
        if raw is None:
            scores[method] = neutral
        else:
            # Lower raw value is better (time), so invert.
            scores[method] = 100.0 * (vmax - float(raw)) / (vmax - vmin)
    return scores, False


def build_radar_raw(rows: List[Dict], scenario_info: Dict[str, Dict], robot_type: str) -> Dict[str, Dict[str, Optional[float]]]:
    per_method = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row["robot_type"] != robot_type or row["status"] != "ok":
            continue

        method = row["method"]
        scenario = row["scenario"]
        dt = scenario_info.get(scenario, {}).get("env_dt")
        dt = 1.0 if dt is None else float(dt)

        # 1) Success
        per_method[method]["success"].append(row["sr_mean"])

        # 3) Feasibility = 1 - infeasible rate
        if row["fr_mean"] is None:
            per_method[method]["feasibility"].append(None)
        else:
            per_method[method]["feasibility"].append(max(0.0, 1.0 - float(row["fr_mean"])))

        # 6) Robustness: prefer explicit robustness metric, fallback to (1 - sr_std)
        if row["robustness_mean"] is not None:
            per_method[method]["robustness"].append(float(row["robustness_mean"]))
        elif row["sr_std"] is not None:
            per_method[method]["robustness"].append(max(0.0, 1.0 - float(row["sr_std"])))
        else:
            per_method[method]["robustness"].append(None)

        # 4) Efficiency raw source: lower trajectory time is better.
        if row["traj_time_mean"] is not None:
            per_method[method]["traj_time"].append(float(row["traj_time_mean"]))
        elif row["avg_ep_len_mean"] is not None:
            per_method[method]["traj_time"].append(float(row["avg_ep_len_mean"]) * dt)
        else:
            per_method[method]["traj_time"].append(None)

        # 5) Compute time raw source: lower is better.
        per_method[method]["compute_time"].append(row["compute_time_mean"])

    aggregated: Dict[str, Dict[str, Optional[float]]] = {}
    for method, d in per_method.items():
        aggregated[method] = {
            "success": _safe_mean(d["success"]),
            "feasibility": _safe_mean(d["feasibility"]),
            "robustness": _safe_mean(d["robustness"]),
            "traj_time": _safe_mean(d["traj_time"]),
            "compute_time": _safe_mean(d["compute_time"]),
        }
    return aggregated


def build_radar_scores(raw_metrics: Dict[str, Dict[str, Optional[float]]]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    methods = list(raw_metrics.keys())
    if not methods:
        return {}, []

    notes: List[str] = []

    # direct [0,1] metrics -> [0,100]
    success_scores = {m: (_clamp01_to_100(raw_metrics[m]["success"]) or 50.0) for m in methods}
    feasibility_scores = {m: (_clamp01_to_100(raw_metrics[m]["feasibility"]) or 50.0) for m in methods}
    robustness_scores = {m: (_clamp01_to_100(raw_metrics[m]["robustness"]) or 50.0) for m in methods}

    # time metrics -> inverse min-max -> [0,100]
    traj_raw = {m: raw_metrics[m]["traj_time"] for m in methods}
    efficiency_scores, traj_missing = _inverse_minmax_scores(traj_raw, neutral=50.0)

    comp_raw = {m: raw_metrics[m]["compute_time"] for m in methods}
    compute_scores, comp_missing = _inverse_minmax_scores(comp_raw, neutral=50.0)

    # Missing notes per axis.
    if all(raw_metrics[m]["success"] is None for m in methods):
        notes.append("success missing -> 50")
    if all(raw_metrics[m]["feasibility"] is None for m in methods):
        notes.append("feasibility missing -> 50")
    if all(raw_metrics[m]["robustness"] is None for m in methods):
        notes.append("robustness missing -> 50")
    if traj_missing:
        notes.append("traj_time missing -> 50")
    if comp_missing:
        notes.append("compute_time missing -> 50")

    scores: Dict[str, Dict[str, float]] = {}
    for method in methods:
        scores[method] = {
            "robustness": float(robustness_scores[method]),
            "success": float(success_scores[method]),
            "feasibility": float(feasibility_scores[method]),
            "efficiency": float(efficiency_scores[method]),
            "compute": float(compute_scores[method]),
        }

    return scores, notes


def write_radar_metrics_csv(
    out_path: Path,
    robot_type: str,
    raw_metrics: Dict[str, Dict[str, Optional[float]]],
    scores: Dict[str, Dict[str, float]],
):
    fieldnames = [
        "robot_type",
        "method",
        "policy",
        "type",
        "success_raw",
        "feasibility_raw",
        "robustness_raw",
        "traj_time_raw",
        "compute_time_raw",
        "success_score",
        "feasibility_score",
        "robustness_score",
        "efficiency_score",
        "compute_score",
    ]

    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for method in sorted(scores.keys(), key=method_sort_key):
            raw = raw_metrics[method]
            sc = scores[method]
            writer.writerow(
                {
                    "robot_type": robot_type,
                    "method": method,
                    "policy": POLICY_DISPLAY.get(method, method),
                    "type": TYPE_DISPLAY.get(method, "Unknown"),
                    "success_raw": raw["success"],
                    "feasibility_raw": raw["feasibility"],
                    "robustness_raw": raw["robustness"],
                    "traj_time_raw": raw["traj_time"],
                    "compute_time_raw": raw["compute_time"],
                    "success_score": sc["success"],
                    "feasibility_score": sc["feasibility"],
                    "robustness_score": sc["robustness"],
                    "efficiency_score": sc["efficiency"],
                    "compute_score": sc["compute"],
                }
            )


def _plot_single_radar(scores: Dict[str, Dict[str, float]], title: str, out_path: Path, dpi: int):
    axis_keys = [k for k, _ in RADAR_AXES]
    axis_labels = [label for _, label in RADAR_AXES]

    n = len(axis_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9.2, 6.8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)
    ax.grid(True, alpha=0.25)

    for method in METHOD_ORDER:
        if method not in scores:
            continue
        vals = [scores[method][k] for k in axis_keys]
        vals += vals[:1]
        style = "-" if TYPE_DISPLAY.get(method) == "Optimization" else "--"
        label = POLICY_DISPLAY.get(method, method)
        line = ax.plot(angles, vals, linewidth=2.0, linestyle=style, label=label)[0]
        ax.fill(angles, vals, color=line.get_color(), alpha=0.07)

    ax.set_title(title, pad=20)
    ax.legend(loc="center left", bbox_to_anchor=(1.10, 0.5), fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_radar_charts(rows: List[Dict], scenario_info: Dict[str, Dict], out_dir: Path, dpi: int) -> List[Path]:
    robot_types = sorted({r["robot_type"] for r in rows if r["robot_type"]})
    saved: List[Path] = []

    radar_csv = out_dir / "radar_metrics_summary.csv"
    if radar_csv.exists():
        radar_csv.unlink()

    for robot_type in robot_types:
        raw = build_radar_raw(rows, scenario_info, robot_type)
        if not raw:
            continue

        scores, notes = build_radar_scores(raw)
        if not scores:
            continue

        write_radar_metrics_csv(radar_csv, robot_type, raw, scores)

        note_suffix = ""
        if notes:
            note_suffix = " | patch: " + ", ".join(notes)
        title = f"Radar Evaluation ({robot_display(robot_type)}){note_suffix}"

        out_path = out_dir / f"radar_metrics_{robot_type}.png"
        _plot_single_radar(scores, title, out_path, dpi=dpi)
        saved.append(out_path)

    return saved


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    compare_root = Path(args.compare_root).resolve()
    if not compare_root.exists():
        raise FileNotFoundError(f"compare_root not found: {compare_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else compare_root
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_info, rows = collect_compare_rows(compare_root)
    common_settings = collect_common_settings(compare_root)
    friendly_seed_summary = collect_friendly_seed_summary(compare_root, method=FRIENDLY_SEED_METHOD)

    csv_path = out_dir / "summary_compare_metrics.csv"
    md_path = out_dir / "summary_compare_report.md"

    write_summary_csv(rows, csv_path)
    write_summary_markdown(scenario_info, rows, common_settings, friendly_seed_summary, md_path)

    sr_figs = plot_success_rate_curves(rows, out_dir, dpi=args.dpi)
    radar_figs = plot_radar_charts(rows, scenario_info, out_dir, dpi=args.dpi)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    for p in sr_figs:
        print(f"Saved Figure (SR): {p}")
    for p in radar_figs:
        print(f"Saved Figure (Radar): {p}")
    if not sr_figs:
        print("No SR curve figure generated.")
    if not radar_figs:
        print("No radar figure generated.")


if __name__ == "__main__":
    main()
