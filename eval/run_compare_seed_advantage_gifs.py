#!/usr/bin/env python3
"""
Run one-episode evaluation for a single (robot_type, obstacle_number) scenario
across a 100-seed window: [seed, seed + 99].

Selection rule ("target_method works best"):
  target method succeeds AND every other method fails.

For matched seeds, this script renders GIFs for all methods.
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import imageio
import numpy as np
import torch
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MAIN_DIR = str(ROOT_DIR)

from config.config import Config
from controller.robot_controller_factory import build_robot_controller
from crowd_nav.rl_policy_factory import get_rl_policy_class
from crowd_sim.utils import build_env, polar_obs_dim_from_env_dim, relative_obs_dim_from_env_dim
from eval.policy_kwargs import build_eval_policy_kwargs, filter_policy_kwargs
from eval.eval_util import RLEvalActorAdapter, build_obs_preprocess_fn, run_one_episode


DEFAULT_METHOD_ORDER = [
    "orca",
    "cbfqp",
    "cvarqp",
    "adapcvarqp",
    "drcvarqp",
    "rl",
    "rlcbfgamma",
    "rlcvarbetaradius",
]
DEFAULT_FPS = 10
SEED_WINDOW = 100

METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcbfgamma_2nets": True,
    "rlcbfgamma_2nets_risk": True,
    "rlcvarbetaradius": True,
    "rlcvarbetaradius_2nets": True,
    "rlcvarbetaradius_2nets_risk": True,
}


def _resolve_needs_qp_relative(method: str) -> bool:
    return bool(METHOD_NEEDS_QP_RELATIVE.get(str(method).strip().lower(), False))


def parse_scenario_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    if "_obs_" not in name:
        return None, None
    robot_type, obs = name.rsplit("_obs_", 1)
    try:
        obs_count = int(obs)
    except Exception:
        obs_count = None
    return robot_type, obs_count


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _method_sort_key(name: str) -> Tuple[int, str]:
    if name in DEFAULT_METHOD_ORDER:
        return DEFAULT_METHOD_ORDER.index(name), name
    return 999, name


def _apply_config_snapshot(cfg: Config, cfg_dict: Dict[str, Any]) -> None:
    for section_name in ["env", "human", "robot", "controller", "reward"]:
        section_data = cfg_dict.get(section_name)
        if not isinstance(section_data, dict):
            continue
        section_obj = getattr(cfg, section_name, None)
        if section_obj is None:
            continue
        for key, value in section_data.items():
            section_obj[key] = value


def build_scenario_config(scenario_dir: Path) -> Tuple[Config, str]:
    cfg = Config()
    env_name = str(cfg.env.get("name", "social_nav_var_num"))
    has_snapshot = False

    snapshot_paths = sorted(scenario_dir.glob("*/test_config.json"))
    if snapshot_paths:
        has_snapshot = True
        snapshot = _load_json(snapshot_paths[0])
        cfg_dict = snapshot.get("config")
        if isinstance(cfg_dict, dict):
            _apply_config_snapshot(cfg, cfg_dict)
        hyp = snapshot.get("hyperparameters")
        if isinstance(hyp, dict) and isinstance(hyp.get("env_name"), str):
            env_name = str(hyp["env_name"])

    robot_type, obs_count = parse_scenario_name(scenario_dir.name)
    if robot_type is not None:
        cfg.robot.type = robot_type
    if obs_count is not None:
        cfg.human.num_humans = int(obs_count)
        if not has_snapshot:
            cfg.env.max_obstacles_obs = int(obs_count)

    return cfg, env_name


def discover_methods(scenario_dir: Path) -> List[str]:
    methods: List[str] = []
    for d in scenario_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("__"):
            continue
        # Exclude this script's own output folders inside scenario dir.
        if re.fullmatch(r"seed_\d+_\d+", name):
            continue
        # Keep known method dirs; for unknown dirs, only keep if they look like eval method folders.
        if name not in DEFAULT_METHOD_ORDER and not (d / "checkpoint_eval_all_multiseed.json").exists():
            continue
        methods.append(name)
    methods.sort(key=_method_sort_key)
    return methods


def _is_rl_method(method: str) -> bool:
    return str(method).lower().startswith("rl")


def _extract_step(name: str) -> int:
    m = re.search(r"(?:^|[_-])step[_-]?(\d+)", name.lower())
    return int(m.group(1)) if m else -1


def resolve_actor_checkpoint(method_dir: Path) -> Optional[Path]:
    summary = _load_json(method_dir / "checkpoint_eval_all_multiseed.json")
    best = summary.get("best_checkpoint")
    if isinstance(best, dict):
        cp = best.get("checkpoint_path")
        if isinstance(cp, str) and cp:
            cp_path = Path(cp)
            if not cp_path.is_absolute():
                cp_path = method_dir / cp_path
            if cp_path.exists():
                return cp_path

        cp_file = best.get("checkpoint_file")
        if isinstance(cp_file, str) and cp_file and ":" not in cp_file:
            candidate = method_dir / cp_file
            if candidate.exists():
                return candidate

    best_ckpt = method_dir / "ppo_actor_best.pth"
    if best_ckpt.exists():
        return best_ckpt

    actor_ckpts: List[Path] = []
    for p in sorted(method_dir.glob("*.pth")):
        lower = p.name.lower()
        if "actor" not in lower:
            continue
        if "critic" in lower:
            continue
        if any(x in lower for x in ("optim", "optimizer", "sched", "scaler")):
            continue
        actor_ckpts.append(p)

    if not actor_ckpts:
        return None
    actor_ckpts.sort(key=lambda x: (_extract_step(x.name), x.name))
    return actor_ckpts[-1]


def build_actor_for_method(
    method: str,
    method_dir: Path,
    cfg: Config,
    env,
    device: torch.device,
    *,
    obs_topk: int,
    obs_farest_dist: float,
    nHidden1: int,
    nHidden21: int,
    nHidden22: int,
    alpha_hidden1: int,
    alpha_hidden2: int,
) -> Tuple[Any, Optional[Path], Any]:
    if _is_rl_method(method):
        PolicyClass = get_rl_policy_class(method)
        ckpt_path = resolve_actor_checkpoint(method_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No actor checkpoint found under {method_dir}")

        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        env_obs_dim = int(env.observation_space.shape[0])
        actor_obs_dim = int(polar_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk))
        needs_qp_relative = _resolve_needs_qp_relative(method)
        qp_obs_dim = int(relative_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk)) if needs_qp_relative else None
        act_dim = env.action_space.shape[0]
        policy_kwargs = build_eval_policy_kwargs(
            cfg,
            method,
            qp_obs_dim=qp_obs_dim,
            nHidden1=int(nHidden1),
            nHidden21=int(nHidden21),
            nHidden22=int(nHidden22),
            alpha_hidden1=int(alpha_hidden1),
            alpha_hidden2=int(alpha_hidden2),
        )
        policy = PolicyClass(actor_obs_dim, act_dim, **filter_policy_kwargs(PolicyClass, policy_kwargs)).to(device)
        policy.load_state_dict(state, strict=True)
        policy.eval()
        actor = RLEvalActorAdapter(policy, env.action_space, device)
        obs_preprocess_fn = build_obs_preprocess_fn(
            obs_topk=int(obs_topk),
            obs_farest_dist=float(obs_farest_dist),
            needs_qp_relative=bool(needs_qp_relative),
        )
        return actor, ckpt_path, obs_preprocess_fn

    actor = build_robot_controller(method, cfg, env)
    return actor, None, None


def evaluate_one_method(
    scenario_cfg: Config,
    env_name: str,
    scenario_dir: Path,
    method: str,
    seed: int,
    device: torch.device,
    collect_frames: bool,
    *,
    obs_topk: int,
    obs_farest_dist: float,
    nHidden1: int,
    nHidden21: int,
    nHidden22: int,
    alpha_hidden1: int,
    alpha_hidden2: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(scenario_cfg)
    cfg.env.rl_xy_to_unicycle = bool(method == "rl" and cfg.robot.type == "unicycle")

    render_mode = "rgb_array" if collect_frames else None
    env = build_env(env_name, render_mode=render_mode, config=cfg)

    checkpoint_path: Optional[Path] = None
    try:
        actor, checkpoint_path, obs_preprocess_fn = build_actor_for_method(
            method=method,
            method_dir=scenario_dir / method,
            cfg=cfg,
            env=env,
            device=device,
            obs_topk=int(obs_topk),
            obs_farest_dist=float(obs_farest_dist),
            nHidden1=int(nHidden1),
            nHidden21=int(nHidden21),
            nHidden22=int(nHidden22),
            alpha_hidden1=int(alpha_hidden1),
            alpha_hidden2=int(alpha_hidden2),
        )
        ep = run_one_episode(
            actor=actor,
            env=env,
            seed=seed,
            reset_options=None,
            track_signals=False,
            unom_holder=None,
            collect_frames=collect_frames,
            obs_preprocess_fn=obs_preprocess_fn,
        )

        success = bool(ep["ep_success"]) and (not bool(ep["ep_collision"]))
        result = {
            "status": "ok",
            "success": success,
            "ep_collision": bool(ep["ep_collision"]),
            "ep_success": bool(ep["ep_success"]),
            "ep_timeout": bool(ep["ep_timeout"]),
            "ep_infeasible": bool(ep.get("ep_infeasible", False)),
            "ep_len": int(ep["ep_len"]),
            "ep_ret": float(ep["ep_ret"]),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "error": None,
        }
        if collect_frames:
            result["frames"] = ep.get("frames", [])
        return result
    except Exception as exc:
        return {
            "status": "error",
            "success": False,
            "ep_collision": None,
            "ep_success": None,
            "ep_timeout": None,
            "ep_infeasible": None,
            "ep_len": None,
            "ep_ret": None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
            "frames": [] if collect_frames else None,
        }
    finally:
        env.close()


def save_gif(frames: List[np.ndarray], out_path: Path, fps: int) -> bool:
    if not frames:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=int(fps))
    return True


def short_result(res: Dict[str, Any]) -> str:
    if res.get("status") != "ok":
        return f"error ({res.get('error')})"
    return (
        f"succ={int(bool(res.get('success')))} "
        f"coll={int(bool(res.get('ep_collision')))} "
        f"timeout={int(bool(res.get('ep_timeout')))} "
        f"infeas={int(bool(res.get('ep_infeasible')))} "
        f"len={res.get('ep_len')}"
    )


def main(cfg: DictConfig) -> None:
    seed_start = int(cfg.seed)
    compare_root = Path(str(cfg.compare_root)).resolve()
    target_method = str(cfg.target_method)
    robot_type = str(cfg.robot_type)
    obstacle_number = int(cfg.obstacle_number)
    out_dir_cfg = str(cfg.out_dir)
    obs_topk = int(cfg.obs_topk)
    obs_farest_dist = float(cfg.obs_farest_dist)
    nHidden1 = int(cfg.nHidden1)
    nHidden21 = int(cfg.nHidden21)
    nHidden22 = int(cfg.nHidden22)
    alpha_hidden1 = int(cfg.alpha_hidden1)
    alpha_hidden2 = int(cfg.alpha_hidden2)

    if obs_topk <= 0:
        raise ValueError("--obs_topk must be > 0")
    if not compare_root.exists():
        raise FileNotFoundError(f"compare_root not found: {compare_root}")

    if obstacle_number <= 0:
        raise ValueError("--obstacle_number must be > 0")

    scenario_name = f"{robot_type}_obs_{obstacle_number}"
    scenario_dir = compare_root / scenario_name
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    seed_end = seed_start + SEED_WINDOW - 1

    out_dir = (
        Path(out_dir_cfg).resolve()
        if out_dir_cfg
        else (compare_root / scenario_name / f"seed_{seed_start}_{seed_end}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    np.random.seed(seed_start)
    random.seed(seed_start)
    torch.manual_seed(seed_start)

    methods = discover_methods(scenario_dir)
    if not methods:
        raise RuntimeError(f"No method directories found in: {scenario_dir}")

    scenario_cfg, env_name = build_scenario_config(scenario_dir)

    print(f"[Config] compare_root={compare_root}")
    print(f"[Config] scenario={scenario_name}, env={env_name}, methods={methods}")
    print(f"[Config] seed_range=[{seed_start}, {seed_end}], device={device}")
    print(f"[Config] target_method={target_method}")

    summary: Dict[str, Any] = {
        "seed_start": seed_start,
        "seed_end": seed_end,
        "seed_window": SEED_WINDOW,
        "target_method": target_method,
        "robot_type": robot_type,
        "obstacle_number": obstacle_number,
        "scenario": scenario_name,
        "compare_root": str(compare_root),
        "out_dir": str(out_dir),
        "device": str(device),
        "methods": methods,
        "matched_seeds": [],
        "seed_records": [],
    }

    if target_method not in methods:
        summary["error"] = (
            f"target_method '{target_method}' not found in {scenario_dir}. "
            f"Available methods: {methods}"
        )
        summary_path = out_dir / f"{scenario_name}_seed_{seed_start}_{seed_end}_advantage_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(summary["error"])
        print(f"Summary saved: {summary_path}")
        return

    for seed in range(seed_start, seed_end + 1):
        print(f"\n[seed={seed}] first pass", flush=True)
        first_pass: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            print(f"  - eval {method}", flush=True)
            res = evaluate_one_method(
                scenario_cfg=scenario_cfg,
                env_name=env_name,
                scenario_dir=scenario_dir,
                method=method,
                seed=seed,
                device=device,
                collect_frames=False,
                obs_topk=obs_topk,
                obs_farest_dist=obs_farest_dist,
                nHidden1=nHidden1,
                nHidden21=nHidden21,
                nHidden22=nHidden22,
                alpha_hidden1=alpha_hidden1,
                alpha_hidden2=alpha_hidden2,
            )
            first_pass[method] = res
            print(f"    {short_result(res)}", flush=True)

        target_res = first_pass.get(target_method)
        others = [m for m in methods if m != target_method]
        target_success = bool(target_res is not None and target_res.get("success"))
        others_all_fail = bool(others) and all(not bool(first_pass[m].get("success")) for m in others)
        all_methods_ok = all(first_pass[m].get("status") == "ok" for m in methods)
        is_match = bool(target_success and others_all_fail and all_methods_ok)

        seed_record: Dict[str, Any] = {
            "seed": seed,
            "first_pass": first_pass,
            "match": is_match,
            "match_rule": {
                "target_success": target_success,
                "others_all_fail": others_all_fail,
                "all_methods_ok": all_methods_ok,
            },
            "gif_dir": None,
            "render_pass": None,
        }

        if is_match:
            case_dir = out_dir / f"seed_{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"  -> matched. rendering gifs into {case_dir}", flush=True)

            second_pass: Dict[str, Dict[str, Any]] = {}
            for method in methods:
                render_res = evaluate_one_method(
                    scenario_cfg=scenario_cfg,
                    env_name=env_name,
                    scenario_dir=scenario_dir,
                    method=method,
                    seed=seed,
                    device=device,
                    collect_frames=True,
                    obs_topk=obs_topk,
                    obs_farest_dist=obs_farest_dist,
                    nHidden1=nHidden1,
                    nHidden21=nHidden21,
                    nHidden22=nHidden22,
                    alpha_hidden1=alpha_hidden1,
                    alpha_hidden2=alpha_hidden2,
                )

                gif_path = None
                frames = render_res.get("frames", []) or []
                if render_res.get("status") == "ok" and len(frames) > 0:
                    succ_bit = 1 if bool(render_res.get("success")) else 0
                    coll_bit = 1 if bool(render_res.get("ep_collision")) else 0
                    gif_name = f"{method}_seed_{seed}_succ_{succ_bit}_coll_{coll_bit}.gif"
                    out_path = case_dir / gif_name
                    if save_gif(frames, out_path, fps=DEFAULT_FPS):
                        gif_path = str(out_path)
                        print(f"    saved {method}: {out_path}", flush=True)

                render_res.pop("frames", None)
                render_res["gif_path"] = gif_path
                second_pass[method] = render_res

            seed_record["gif_dir"] = str(case_dir)
            seed_record["render_pass"] = second_pass
            summary["matched_seeds"].append(seed)
        else:
            print(
                "  -> not matched "
                f"(target_success={target_success}, others_all_fail={others_all_fail}, all_methods_ok={all_methods_ok})",
                flush=True,
            )

        summary["seed_records"].append(seed_record)

    summary_path = out_dir / f"{scenario_name}_seed_{seed_start}_{seed_end}_advantage_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Summary saved: {summary_path}")
    print(f"Matched seeds: {summary['matched_seeds']}")


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="eval_compare_seed_advantage_gifs",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
