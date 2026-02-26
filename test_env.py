import argparse
import imageio
import numpy as np

from config.config import Config
from crowd_sim.utils import build_env
import time

def run_env_test(
    env_name="social_nav",
    max_steps=200,
    gif_path=None,
    seed=0,
    obs_num=20,
):
    config = Config()
    config.human.num_humans = obs_num  # Test with multiple humans to verify dynamics and rendering
    render_mode = "rgb_array"
    env = build_env(env_name, render_mode, config)

    obs, info = env.reset(seed=seed)

    frames = []
    done = False
    step = 0
    total_reward = 0.0

    # No robot policy: keep action zero for testing env dynamics.
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    while step < max_steps:
        time_start = time.time()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        step += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        time_end = time.time()
        print(f"Step {step} | Reward: {reward:.2f} | Time: {time_end - time_start:.3f}s", flush=True)

    if frames:
        if gif_path is None:
            gif_path = f"trained_models/{env_name}_env_test.gif"
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
    parser.add_argument("--obs_num", type=int, default=20, help="Number of obstacles to observe (for social_nav_var_num)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    run_env_test(
        env_name=args.env_name,
        max_steps=args.steps,
        gif_path=args.save_path,
        obs_num=args.obs_num,
    )
