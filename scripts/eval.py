from omegaconf import OmegaConf
from hydra.utils import instantiate

# Register math resolver (needed for configs with e.g. obs_dim: ${math:3+${env.topk}*4})
OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)
import torch
import os
from new_rl.utils import map_action_to_env
import numpy as np
from functools import partial
import glob
import json
from tqdm import tqdm
import gymnasium as gym
import imageio
from PIL import Image


class Evaluator:
    """Evaluator takes only config. Use Evaluator.from_run_dir(save_dir) for loading from a trained run."""

    def __init__(self, save_dir: str, use_all_obs: bool = False, visualize: bool = False):
        self.use_all_obs = use_all_obs
        self.visualize = visualize
        self.save_dir = save_dir
        self.config = OmegaConf.load(os.path.join(save_dir, "config.yaml"))
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = instantiate(self.config.model, _convert_="none")
        self.prepare_env()

        
    def prepare_env(self):
        if self.config.env.type == "mujoco":
            def make_env_fn(seed_offset: int = 0):
                def _thunk():
                    env = gym.make(self.config.env.env_id)
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    env.reset(seed=self.config.seed + seed_offset)
                    return env

                return _thunk
            self.make_env_fn = make_env_fn
        else:
            from new_rl.env_wrapper.crowdsim import CrowdSimConfig, build_env
            self.crowd_sim_config = CrowdSimConfig()
            if self.use_all_obs: # performance can be slightly worse sometimes
                self.crowd_sim_config.env.max_obstacles_obs = 20
            def make_env_fn(crowd_config: CrowdSimConfig, env_name: str):
                def _init():
                    env = build_env(env_name, render_mode="rgb_array" if self.visualize else None, config=crowd_config, 
                                    topk=getattr(self.config.env, "topk", 5), 
                                    farest_dist=getattr(self.config.env, "farest_dist", 5))
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    env.reset(seed=self.config.seed)
                    return env
                return _init
    
            self.make_env_fn = partial(make_env_fn, self.crowd_sim_config, self.config.env.env_id)
       

    def evaluate_one_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()
        self.model.to(self.device)

        if "obs_normalizer" in ckpt and self.model.obs_normalizer is not None:
            self.model.obs_normalizer.load_state_dict(
                {k: torch.as_tensor(v) for k, v in ckpt["obs_normalizer"].items()},
                strict=False,
            )
        if "return_normalizer" in ckpt and self.model.return_normalizer is not None:
            self.model.return_normalizer.load_state_dict(
                {k: torch.as_tensor(v) for k, v in ckpt["return_normalizer"].items()},
                strict=False,
            )
        if self.visualize:
            ckpt_name = os.path.basename(ckpt_path).split(".")[0]
            self.video_save_dir = os.path.join(self.save_dir, f"visualize_{ckpt_name}")
            if self.use_all_obs:
                self.video_save_dir += "_allobs"
            os.makedirs(self.video_save_dir, exist_ok=True)
        mean_return, std_return, success_rate, collision_rate, timeout_rate = self.evaluate()
        return mean_return, std_return, success_rate, collision_rate, timeout_rate
    
    def eval_all_ckpts(self):
        ckpt_paths = glob.glob(os.path.join(self.save_dir, "ckpt_*.pt")) # sorted by step
        ckpt_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        results = {}
        for ckpt_path in ckpt_paths[::-1]:
            print(f"Evaluating {ckpt_path}...")
            mean_return, std_return, success_rate, collision_rate, timeout_rate = self.evaluate_one_ckpt(ckpt_path)
            print(f"\tMean return: {mean_return:.2f}, std return: {std_return:.2f}, " 
                  f"success rate: {success_rate:.2f}, collision rate: {collision_rate:.2f}, timeout rate: {timeout_rate:.2f}")
            step = int(ckpt_path.split("_")[-1].split(".")[0])
            results[step] = {
                "mean_return": mean_return,
                "std_return": std_return,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
            }
            
        # calculate min, max, mean of all the ckpts
        min_return = min(results.values(), key=lambda x: x["mean_return"])["mean_return"]
        max_return = max(results.values(), key=lambda x: x["mean_return"])["mean_return"]
        mean_return = np.mean([x["mean_return"] for x in results.values()])
        std_return = np.std([x["mean_return"] for x in results.values()])
        mean_success_rate = np.mean([x["success_rate"] for x in results.values()])
        mean_collision_rate = np.mean([x["collision_rate"] for x in results.values()])
        mean_timeout_rate = np.mean([x["timeout_rate"] for x in results.values()])
        max_success_rate = max(results.values(), key=lambda x: x["success_rate"])["success_rate"]
        max_collision_rate = max(results.values(), key=lambda x: x["collision_rate"])["collision_rate"]
        max_timeout_rate = max(results.values(), key=lambda x: x["timeout_rate"])["timeout_rate"]
        min_success_rate = min(results.values(), key=lambda x: x["success_rate"])["success_rate"]
        min_collision_rate = min(results.values(), key=lambda x: x["collision_rate"])["collision_rate"]
        min_timeout_rate = min(results.values(), key=lambda x: x["timeout_rate"])["timeout_rate"]
        
        results["aggregate"] = {
            "min_return": min_return,
            "max_return": max_return,
            "mean_return": mean_return,
            "std_return": std_return,
            "mean_success_rate": mean_success_rate,
            "mean_collision_rate": mean_collision_rate,
            "mean_timeout_rate": mean_timeout_rate,
            "max_success_rate": max_success_rate,
            "max_collision_rate": max_collision_rate,
            "max_timeout_rate": max_timeout_rate,
            "min_success_rate": min_success_rate,
            "min_collision_rate": min_collision_rate,
            "min_timeout_rate": min_timeout_rate,
        }
        if not self.use_all_obs:
            with open(os.path.join(self.save_dir, "eval_results.json"), "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(os.path.join(self.save_dir, "eval_results_allobs.json"), "w") as f:
                json.dump(results, f, indent=4)
        return results

    def evaluate(self):
        """Evaluate policy deterministically (mean action) in a single env."""

        env = self.make_env_fn()()        
        action_space = env.action_space
        action_low = np.asarray(action_space.low, dtype=np.float32)
        action_high = np.asarray(action_space.high, dtype=np.float32)
        seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        episodes = 50
        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_episodes = 0
        returns = []
        for seed in tqdm(seeds, desc="Evaluating"):
            for ep in range(episodes):
                obs_np, _ = env.reset(seed=seed + ep)
                done = False
                total = 0.0
                if self.visualize:
                    frames = []
                while not done:
                    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = self.model.get_action_deterministic(obs_t).squeeze(0)

                    action_np = map_action_to_env(
                        action.cpu().numpy(),
                        action_low,
                        action_high,
                        self.config.trainer.action_bound_method,
                    )

                    obs_np, r, term, trunc, info = env.step(action_np)
                    done = term or trunc
                    total += float(r)
                    
                    if self.visualize:
                        frames.append(Image.fromarray(env.render()).resize((256, 256)))
                        
                if self.visualize:
                    success = info.get("is_success", False)
                    imageio.mimsave(os.path.join(self.video_save_dir, f"seed_{seed + ep}_{success}.mp4"), frames, fps=30)
                        
                returns.append(total)
                total_episodes += 1
                success_count += int(info.get("is_success", False))
                collision_count += int(info.get("is_collision", False))
                timeout_count += int(info.get("is_timeout", False))
        
        env.close()
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        return mean_return, std_return, success_count/total_episodes, collision_count/total_episodes, timeout_count/total_episodes


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--save-dir", type=str, required=True, help="Run directory containing config.yaml and ckpt_*.pt")
    args.add_argument("--use-all-obs", action="store_true", help="Use multiple obstacles information in the environment")
    args.add_argument("--visualize", action="store_true", help="Visualize the evaluation")
    args = args.parse_args()
    evaluator = Evaluator(save_dir=args.save_dir, use_all_obs=args.use_all_obs, visualize=args.visualize)
    evaluator.eval_all_ckpts()