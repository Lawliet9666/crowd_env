import argparse
import imageio
import os
import numpy as np

from config.config import Config
from crowd_sim.utils import build_env
import time

def run_env_test(
    env_name= None,
    max_steps=None,
    gif_path=None,
    seed=None,
    human_policy=None,
    human_num=20,
):
    config = Config()
    config.human.num_humans = human_num  # Test with multiple humans to verify dynamics and rendering
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
        if gif_path is None:
            human_policy_name = config.human.get('policy', 'nominal') if isinstance(config.human, dict) else getattr(config.human, 'policy', 'nominal')
            save_dir = "trained_models/env_test"
            os.makedirs(save_dir, exist_ok=True)
            gif_path = f"{save_dir}/{env_name}_{human_policy_name}_seed{seed}.gif"
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved GIF to {gif_path}")

    print(f"Steps: {step} | Total Reward: {total_reward:.2f}")
    if info.get("is_collision"):
        print("Result: COLLISION")
    elif info.get("is_success"):
        print("Result: SUCCESS")
    elif info.get("is_timeout"):
        print("Result: TIMEOUT")

    env.close()


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
    parser.add_argument("--save_path", type=str, default=None, help="Output GIF path")
    parser.add_argument("--human_policy", type=str, default=None, choices=["orca", "social_force", "potential_field", "nominal"], help="Human policy to use")
    parser.add_argument("--human_num", type=int, default=20, help="Number of humans in the environment")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    for seed in range(args.seed, args.seed + 5):  # Run multiple seeds for robustness
        print(f"\n=== Running env test with seed {seed} ===")
        run_env_test(
            env_name=args.env_name,
            max_steps=args.steps,
            gif_path=args.save_path,
            seed=seed,
            human_policy=args.human_policy,
        )

    # run_env_test(
    #     env_name=args.env_name,
    #     max_steps=args.steps,
    #     gif_path=args.save_path,
    #     seed=args.seed,
    #     human_policy=args.human_policy,
    # )
