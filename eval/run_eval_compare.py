#!/usr/bin/env python3
"""Batch compare runner: all robot types x obstacle counts."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def run_eval(
    root_dir: Path,
    actor_model_path: str,
    method: str,
    robot_type: str,
    obstacle_count: int,
    episodes_per_seed: int,
):
    cmd = [
        sys.executable,
        str(root_dir / "eval" / "eval_batch.py"),
        f"actor_model={actor_model_path}",
        f"method={method}",
        f"episodes_per_seed={int(episodes_per_seed)}",
        f"robot_type={robot_type}",
        f"num_humans={int(obstacle_count)}",
    ]
    subprocess.run(cmd, cwd=str(root_dir), check=True)


def evaluate_one_scenario(
    root_dir: Path,
    trained_models_dir: Path,
    source_model_folder: str,
    robot_type: str,
    obstacle_count: int,
    episodes_per_seed: int,
    ctrl_methods: list[str],
    rl_methods: list[str],
    rl_source_by_method: dict,
):
    scenario = f"{robot_type}_obs_{obstacle_count}"
    base_dir = trained_models_dir / "compare" / scenario
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Scenario: {scenario} ===")
    print(f"save_dir={base_dir}")

    for method in ctrl_methods:
        method_dir = base_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"[controller] method={method}")
        run_eval(
            root_dir=root_dir,
            actor_model_path=str(method_dir),
            method=method,
            robot_type=robot_type,
            obstacle_count=obstacle_count,
            episodes_per_seed=episodes_per_seed,
        )

    for method in rl_methods:
        src_dir = trained_models_dir / source_model_folder / rl_source_by_method[method]
        src_ckpts = sorted(src_dir.glob("*.pth"))
        dst_dir = base_dir / method
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_ckpts:
            print(f"[rl] method={method} skipped: no checkpoints in {src_dir}")
            continue

        for ckpt in src_ckpts:
            shutil.copy2(ckpt, dst_dir / ckpt.name)

        print(f"[rl] method={method} source={rl_source_by_method[method]}")
        run_eval(
            root_dir=root_dir,
            actor_model_path=str(dst_dir),
            method=method,
            robot_type=robot_type,
            obstacle_count=obstacle_count,
            episodes_per_seed=episodes_per_seed,
        )


def main():
    root_dir = ROOT_DIR
    trained_models_dir = root_dir / "trained_models"
    # Set where RL checkpoints are copied from under trained_models/.
    source_model_folder = "default2"  # e.g. "default", "default2"

    # Global rollout settings.
    episodes_per_seed = 50

    # Rollout all robot types and all obstacle counts from 5 to 25.
    robot_types = ["unicycle", "single_integrator"]
    # robot_types = ["unicycle"]
    obstacle_counts = [5, 10, 15, 20, 25]
    # obstacle_counts = [20, 15, 10, 5]
    # obstacle_counts = [15, 10, 5]
    # obstacle_counts = [25]
    # obstacle_counts = [5]

    # Methods configured in one place.
    # ctrl_methods: list[str] = ["orca", "cbfqp", "cvarqp", "adapcvarqp"]
    # ctrl_methods: list[str] = ["orca", "cbfqp", "cvarqp"]
    ctrl_methods: list[str] = ["orca"]
    # ctrl_methods: list[str] = ["adapcvarqp"]
    # ctrl_methods = ["orca", "cbfqp"]
    # ctrl_methods = []
    # rl_methods: list[str] = ["rl", "rlcbfgamma", "rlcvarbetaradius"]
    # rl_methods: list[str] = ["rlcbfgamma"]
    # rl_methods: list[str] = ["rlcvarbetaradius"]
    # rl_methods: list[str] = ["rl"]
    rl_methods = []

    # Set source model per RL method (edit these names as needed).
    rl_sources = {
        "unicycle": {
            "rl": "20260301_160245_unicycle_rl",
            "rlcbfgamma": "20260301_072939_unicycle_rlcbfgamma", # 20260301_072939_unicycle_rlcbfgamma
            "rlcvarbetaradius": "20260301_173615_unicycle_rlcvarbetaradius", # 20260301_122725_unicycle_rlcvarbetaradius
        },
        "single_integrator": {
            "rl": "20260301_110617_single_integrator_rl",
            "rlcbfgamma": "20260301_085210_single_integrator_rlcbfgamma",
            "rlcvarbetaradius": "20260301_160625_single_integrator_rlcvarbetaradius",
        },
    }

    print("Batch compare start")
    print(f"robot_types={robot_types}")
    print(f"obstacle_counts={obstacle_counts}")
    print(f"episodes_per_seed={episodes_per_seed}")
    print(f"source_model_folder={source_model_folder}")

    for robot_type in robot_types:
        rl_source_by_method = rl_sources[robot_type]
        print(f"\nRobot type: {robot_type}")
        print(f"controller_methods={ctrl_methods}")
        print(f"rl_methods={rl_methods}")
        print("RL source models:")
        for method in rl_methods:
            print(f"  {method}: {rl_source_by_method[method]}")

        for obstacle_count in obstacle_counts:
            evaluate_one_scenario(
                root_dir=root_dir,
                trained_models_dir=trained_models_dir,
                source_model_folder=source_model_folder,
                robot_type=robot_type,
                obstacle_count=obstacle_count,
                episodes_per_seed=episodes_per_seed,
                ctrl_methods=ctrl_methods,
                rl_methods=rl_methods,
                rl_source_by_method=rl_source_by_method,
            )

    print("\nAll scenarios completed.")


if __name__ == "__main__":
    main()
