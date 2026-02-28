from omegaconf import DictConfig, OmegaConf
import torch
import os
from new_rl.utils import RunningMeanStd
from new_rl.utils import map_action_to_env
import numpy as np
from functools import partial
import glob
import json
from tqdm import tqdm


class Evaluator:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.config = OmegaConf.load(os.path.join(save_dir, "config.yaml"))
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.prepare_model()
        self.prepare_env()
        
    def prepare_env(self):
        if self.config.env.type == "mujoco":
            import gymnasium as gym
            def make_env_fn(seed_offset: int = 0):
                def _thunk():
                    env = gym.make(self.config.env.env_id)
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    env.reset(seed=self.config.seed + seed_offset)
                    return env

                return _thunk
            self.make_env_fn = make_env_fn
        else:
            from config.config import Config as CrowdSimConfig
            from crowd_sim.utils import build_env
            self.crowd_sim_config = CrowdSimConfig()
            # TODO: change config here later
            def make_env_fn(config: CrowdSimConfig, env_name: str):
                def _init():
                    env = build_env(env_name, render_mode=None, config=config)
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    env.reset(seed=self.config.seed)
                    return env
                return _init
    
            self.make_env_fn = partial(make_env_fn, self.crowd_sim_config, self.config.env.env_id)
       
        
    def prepare_model(self):
        if self.config.model.type == "ppo_base":
            from new_rl.model.ppo_base import ActorCritic
            self.model = ActorCritic(
                obs_dim=self.config.env.obs_dim,
                act_dim=self.config.env.act_dim,
                actor_mlp_config=self.config.model.actor.mlp,
                critic_mlp_config=self.config.model.critic.mlp)
        else:
            raise ValueError(f"Model type {self.config.model.type} not supported")

    def evaluate_one_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(self.device)
        
        obs_normalizer = ckpt.get("obs_normalizer", None)
        return_normalizer = ckpt.get("return_normalizer", None)
        if obs_normalizer is not None:
            self.obs_normalizer = RunningMeanStd(shape=(self.config.env.obs_dim,))
            self.obs_normalizer.mean = obs_normalizer["mean"]
            self.obs_normalizer.var = obs_normalizer["var"]
            self.obs_normalizer.count = obs_normalizer["count"]
        else:
            self.obs_normalizer = None
        if return_normalizer is not None:
            self.return_normalizer = RunningMeanStd(shape=())
            self.return_normalizer.mean = return_normalizer["mean"]
            self.return_normalizer.var = return_normalizer["var"]
            self.return_normalizer.count = return_normalizer["count"]
        else:
            self.return_normalizer = None
        mean_return, std_return = self.evaluate()
        return mean_return, std_return
    
    def eval_all_ckpts(self):
        ckpt_paths = glob.glob(os.path.join(self.save_dir, "ckpt_*.pt")) # sorted by step
        ckpt_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        results = {}
        for ckpt_path in ckpt_paths:
            print(f"Evaluating {ckpt_path}...")
            mean_return, std_return = self.evaluate_one_ckpt(ckpt_path)
            print(f"\tMean return: {mean_return:.2f}, std return: {std_return:.2f}")
            step = int(ckpt_path.split("_")[-1].split(".")[0])
            results[step] = {
                "mean_return": mean_return,
                "std_return": std_return,
            }
            
        # calculate min, max, mean of all the ckpts
        min_return = min(results.values(), key=lambda x: x["mean_return"])["mean_return"]
        max_return = max(results.values(), key=lambda x: x["mean_return"])["mean_return"]
        mean_return = np.mean([x["mean_return"] for x in results.values()])
        std_return = np.std([x["mean_return"] for x in results.values()])
        
        results["aggregate"] = {
            "min_return": min_return,
            "max_return": max_return,
            "mean_return": mean_return,
            "std_return": std_return,
        }
        
        with open(os.path.join(self.save_dir, "eval_results.json"), "w") as f:
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
        returns = []
        for seed in tqdm(seeds, desc="Evaluating"):
            for ep in range(episodes):
                obs_np, _ = env.reset(seed=seed + ep)
                done = False
                total = 0.0
                while not done:
                    if self.obs_normalizer is not None:
                        obs_norm = self.obs_normalizer.normalize(obs_np)
                        if isinstance(obs_norm, torch.Tensor):
                            obs_t = obs_norm.float().to(self.device).unsqueeze(0)
                        else:
                            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
                    else:
                        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        action = self.model.actor.net(obs_t).squeeze(0)  # deterministic (mean)

                    action_np = map_action_to_env(
                        action.cpu().numpy(),
                        action_low,
                        action_high,
                        self.config.trainer.action_bound_method,
                    )

                    obs_np, r, term, trunc, _ = env.step(action_np)
                    done = term or trunc
                    total += float(r)
                returns.append(total)
        
        env.close()
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        return mean_return, std_return


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--save-dir", type=str, required=True)
    args = args.parse_args()
    evaluator = Evaluator(args.save_dir)
    evaluator.eval_all_ckpts()