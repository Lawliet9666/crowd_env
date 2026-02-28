import argparse
import imageio
import os
import numpy as np
import json
from datetime import datetime

from config.config import Config
from crowd_sim.utils import build_env
import time

def run_env_test(
    env_name= None,
    max_steps=None,
    save_dir=None,
    seed=None,
    human_policy=None,
    human_num=20,
):
    config = Config()

    # parameter overrides for testing
    config.human.num_humans = human_num  # Test with multiple humans to verify dynamics and rendering
    config.env.max_obstacles_obs = 5  # how many humans are included in the observation  
    config.robot.vmax = 1.0  # Set robot max speed to a reasonable value for testing
    config.robot.radius = 0.3  # Set robot radius to a reasonable value for testing
    config.robot.wmax = np.pi / 2  # Set max angular velocity for unicycle (if applicable)
    config.robot.ini_goal_dist = 6.0  #  initial robot-goal distance  
    # config.reward
    
    if human_policy is not None:
        config.human.policy = human_policy
    render_mode = "rgb_array"
    env = build_env(env_name, render_mode, config)

    obs, info = env.reset(seed=seed)

    frames = []
    done = False
    step = 0
    total_reward = 0.0

    # No robot policy: keep action zero for testing env dynamics.
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1 
    time_start = time.time()

    while step < max_steps:
        time_each_step_start = time.time()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        step += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        time_each_step_end = time.time()
        print(f"Step {step} | Reward: {reward:.2f} | Time: {time_each_step_end - time_each_step_start:.3f}s", flush=True)
    time_end = time.time()
    print(f"Total time for {step} steps: {time_end - time_start:.3f}s", flush=True)

    if frames:
        os.makedirs(save_dir, exist_ok=True)
        gif_path = os.path.join(save_dir, f"seed_{seed}.gif")
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved GIF to {gif_path}")

    print(f"Steps: {step} | Total Reward: {total_reward:.2f}")
    result = "running"
    if info.get("is_collision"):
        print("Result: COLLISION")
        result = "collision"
    elif info.get("is_success"):
        print("Result: SUCCESS")
        result = "success"
    elif info.get("is_timeout"):
        print("Result: TIMEOUT")
        result = "timeout"

    env.close()
    return {
        "seed": int(seed) if seed is not None else None,
        "steps": int(step),
        "total_reward": float(total_reward),
        "result": result,
        "gif_path": gif_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="social_nav_var_num",
        choices=["social_nav", "social_nav_var_num"],
        help="Environment to test",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=200, help="Max steps")
    parser.add_argument("--human_policy", type=str, default=None, choices=["orca", "social_force", "potential_field", "nominal"], help="Human policy to use")
    parser.add_argument("--human_num", type=int, default=20, help="Number of humans in the environment")
    parser.add_argument("--num_seeds", type=int, default=8, help="How many consecutive seeds to run")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    cfg_preview = Config()
    policy_name = args.human_policy if args.human_policy is not None else str(cfg_preview.human.get("policy", "nominal"))
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("trained_models", "env_test", f"{run_ts}_{args.env_name}_{policy_name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving test results to: {run_dir}", flush=True)

    summaries = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        print(f"\n=== Running env test with seed {seed} ===")
        summary = run_env_test(
            env_name=args.env_name,
            max_steps=args.steps,
            save_dir=run_dir,
            seed=seed,
            human_policy=args.human_policy,
            human_num=args.human_num,
        )
        summaries.append(summary)

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)

    # run_env_test(
    #     env_name=args.env_name,
    #     max_steps=args.steps,
    #     seed=args.seed,
    #     human_policy=args.human_policy,
    # )
