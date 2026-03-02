# save ckpt
# training during eval
# final eval

import json
from omegaconf import DictConfig, OmegaConf
import torch
import os
import glob
import re
from new_rl.utils import init_weights
from new_rl.utils import set_seed
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import wandb
from new_rl.utils import map_action_to_env
import numpy as np
import gymnasium as gym
import shutil



class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config
        set_seed(config.seed)        
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        self.steps_per_env = self.config.trainer.batch_size // self.config.trainer.num_envs
        self.batch_size = self.config.trainer.batch_size
        self.minibatch_size = self.config.trainer.minibatch_size
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.num_updates = int(self.config.trainer.total_steps) // self.batch_size

        assert self.batch_size % self.config.trainer.num_envs == 0, "batch_size must be divisible by num_envs"
        assert self.batch_size % self.config.trainer.minibatch_size == 0, "batch_size must be divisible by minibatch_size"
        
        self.setup_wandb()
        self.setup_env()
        self.setup_model_and_optimizer() 

    def _get_curriculum_enabled(self) -> bool:
        # Keep backward compatibility with the existing misspelled config key.
        if "use_curriculum" in self.config.trainer:
            return bool(self.config.trainer.use_curriculum)
        return bool(self.config.trainer.use_cirriculum)
    
    def setup_wandb(self):
        run_name = self.config.run_name + "-" + self.config.model.type +  \
            f"-bs{self.batch_size}-ep{self.config.trainer.update_epochs}-lr{self.config.trainer.lr:.1e}-{self.config.trainer.lr_schedule[:4]}-vf{self.config.trainer.vf_coef}-{self.config.trainer.action_bound_method}"
        
        self.human_num = self.config.trainer.max_human_num
        if self._get_curriculum_enabled():
            run_name = "CL-" + run_name
            self.use_cirriculum = True
            self.initial_human_num = self.config.trainer.initial_human_num
            self.human_num = self.initial_human_num
            self.increase_human_num = self.config.trainer.increase_human_num
            self.max_human_num = self.config.trainer.max_human_num
        else:
            self.use_cirriculum = False
        if self.config.trainer.ent_coef > 0:
            run_name += f"-ent{self.config.trainer.ent_coef}"
            if self.config.trainer.ent_coef_decay:
                run_name += "decay"
        if not self.config.trainer.use_adv_norm:
            run_name += "-no-advnorm"
        if not self.config.trainer.use_obs_norm:
            run_name += "-no-obsnorm"
        if not self.config.trainer.use_return_norm:
            run_name += "-no-retnorm"
        if not self.config.trainer.return_scale_only:
            run_name += "-rss"
        if not self.config.trainer.recompute_adv:
            run_name += "-no-recomp"
        
        if self.config.env.type == "crowdsim":
            if self.config.env.success_reward != 20:
                run_name += f"-suc{self.config.env.success_reward}"
            if self.config.env.collision_penalty != -20:
                run_name += f"-col{self.config.env.collision_penalty}"
            if self.config.env.potential_factor != 3.0:
                run_name += f"-pot{self.config.env.potential_factor}"
            if self.config.env.constant_penalty != -0.025:
                run_name += f"-pen{self.config.env.constant_penalty}"
        
        self.run_name = run_name
        self.save_dir = os.path.join(self.config.save_dir, self.run_name)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self.config.overwrite:
                raise ValueError(f"Save directory {self.save_dir} already exists, please use a different run_name or delete the existing directory")
            else:
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
        
        with open(os.path.join(self.save_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.config, f)
        
        print("config: ", self.config)
            
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.run_name,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
  
    def setup_model_and_optimizer(self):
        if self.config.model.type == "ppo_base":
            from new_rl.model.ppo_base import ActorCritic
            self.model = ActorCritic(
                obs_dim=self.config.env.obs_dim,
                act_dim=self.config.env.act_dim,
                actor_mlp_config=self.config.model.actor.mlp,
                critic_mlp_config=self.config.model.critic.mlp,
                use_obs_norm=self.config.trainer.use_obs_norm,
                use_return_norm=self.config.trainer.use_return_norm,
            )
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.trainer.lr)

        elif self.config.model.type == "sac_base":
            from new_rl.model.sac_base import ActorCritic
            self.model = ActorCritic(
                obs_dim=self.config.env.obs_dim,
                act_dim=self.config.env.act_dim,
                act_low=self.action_low,
                act_high=self.action_high,
                actor_mlp_config=self.config.model.actor.mlp,
                critic_mlp_config=self.config.model.critic.mlp,
            )
            self.model.to(self.device)
        else:
            raise ValueError(f"Model type {self.config.model.type} not supported")
        
        if getattr(self.config.trainer, "use_init_weights", True):
            init_weights(self.model)

        
    def setup_env(self):
        num_envs = self.config.trainer.num_envs
        if self.config.env.type == "mujoco":
            def make_env_fn(seed_offset: int = 0):
                def _thunk():
                    env = gym.make(self.config.env.env_id)
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    env.reset(seed=self.config.seed + seed_offset)
                    return env

                return _thunk
            self.train_envs = SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
            self.make_env_fn = make_env_fn
        else:            
            from new_rl.env_wrapper.crowdsim import CrowdSimConfig, build_env
            self.crowd_sim_config = CrowdSimConfig()
            self.crowd_sim_config.env.max_obstacles_obs = 20
            success_reward = self.config.env.success_reward
            collision_penalty = self.config.env.collision_penalty
            potential_factor = self.config.env.potential_factor
            constant_penalty = self.config.env.constant_penalty
            
            if success_reward != self.crowd_sim_config.reward.success_reward:
                print(f"Adjusting success reward from default {self.crowd_sim_config.reward.success_reward} to {success_reward}")
                self.crowd_sim_config.reward.success_reward = success_reward
            if collision_penalty != self.crowd_sim_config.reward.collision_penalty:
                print(f"Adjusting collision penalty from default {self.crowd_sim_config.reward.collision_penalty} to {collision_penalty}")
                self.crowd_sim_config.reward.collision_penalty = collision_penalty
            if potential_factor != self.crowd_sim_config.reward.potential_factor:
                print(f"Adjusting potential factor from default {self.crowd_sim_config.reward.potential_factor} to {potential_factor}")
                self.crowd_sim_config.reward.potential_factor = potential_factor
            if constant_penalty != self.crowd_sim_config.reward.constant_penalty:
                print(f"Adjusting constant penalty from default {self.crowd_sim_config.reward.constant_penalty} to {constant_penalty}")
                self.crowd_sim_config.reward.constant_penalty = constant_penalty
                
            if self._get_curriculum_enabled():
                self.crowd_sim_config.human.num_humans = self.human_num
            
            def make_env_fn(config: CrowdSimConfig, env_name: str, seed_offset: int = 0):
                def _init():
                    env = build_env(env_name, render_mode=None, config=config, topk=self.config.env.topk, farest_dist=self.config.env.farest_dist)
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    env.reset(seed=self.config.seed + seed_offset)
                    return env
                return _init
            print(f"Initializing env with {self.human_num} humans")
            self.train_envs = AsyncVectorEnv(
                [make_env_fn(self.crowd_sim_config, self.config.env.env_id, i) for i in range(num_envs)]
            )
            self.make_env_fn = make_env_fn
        

        self.action_space = self.train_envs.single_action_space
        self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32)
        
        self.observation_space = self.train_envs.single_observation_space
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        
        print(f"Train on environment {self.config.env.env_id}")
        print(f"\tObservation space: {self.observation_space}")
        print(f"\tObs dim: {self.obs_dim}")
        print(f"\tAction space: {self.action_space}")
        print(f"\tAct dim: {self.act_dim}")
        assert self.obs_dim == self.config.env.obs_dim, "obs_dim mismatch"
        assert self.act_dim == self.config.env.act_dim, "act_dim mismatch"

        assert np.allclose(self.action_low, self.config.env.act_low), "action_low mismatch"
        assert np.allclose(self.action_high, self.config.env.act_high), "action_high mismatch"
        
    
    def reset_env(self):
        # for cirriculum learning, reset the env with different human numbers
        if not self.use_cirriculum:
            return
        if self.human_num >= self.max_human_num:
            return
        self.train_envs.close()
        self.human_num = min(self.human_num + self.increase_human_num, self.max_human_num)
        print(f"Resetting env with {self.human_num} humans")
        self.crowd_sim_config.human.num_humans = self.human_num
        self.train_envs = AsyncVectorEnv(
            [
                self.make_env_fn(self.crowd_sim_config, self.config.env.env_id, i)
                for i in range(self.config.trainer.num_envs)
            ]
        )

    def train(self):
        raise NotImplementedError("Training is not implemented yet")
    
    
    def eval(self, episodes: int = 10):
        """Evaluate policy deterministically (mean action) in a single env."""
        self.model.eval()
        if self.config.env.type == "crowdsim":
            env = self.make_env_fn(self.crowd_sim_config, self.config.env.env_id, 0)()
        else:
            env = self.make_env_fn(0)()
        returns = []
        success_count = 0
        collision_count = 0
        timeout_count = 0
        for seed in range(100, 1000 + 1, 100):
            for ep in range(episodes):
                obs_np, _ = env.reset(seed=seed + ep)
                done = False
                total = 0.0
                while not done:
                    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = self.model.get_action_deterministic(obs_t).squeeze(0)

                    action_np = map_action_to_env(
                        action.cpu().numpy(),
                        self.action_low,
                        self.action_high,
                        self.config.trainer.action_bound_method,
                    )

                    obs_np, r, term, trunc, info = env.step(action_np)
                    done = term or trunc
                    total += float(r)
                returns.append(total)
                success_count += int(info.get("is_success", False))
                collision_count += int(info.get("is_collision", False))
                timeout_count += int(info.get("is_timeout", False))
        env.close()
        total_count = success_count + collision_count + timeout_count
        if total_count == 0:
            return float(np.mean(returns)), float(np.std(returns)), 0.0, 0.0, 0.0
        else:
            return float(np.mean(returns)), float(np.std(returns)), float(success_count)/total_count, float(collision_count)/total_count, float(timeout_count)/total_count

    
    def save_ckpt(self, save_dir: str, step: int, performance: float, max_keep: int = 10):
        """Save checkpoint and keep only the top `max_keep` checkpoints by performance."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"ckpt_{step:08d}.pt")
        ckpt = {
            "model": self.model.state_dict(),
            "step": step,
        }
        if self.use_cirriculum:
            if self.human_num < self.max_human_num:
                return 
        torch.save(ckpt, path)

        # Maintain manifest: {filename -> {step, performance}}
        manifest_path = os.path.join(save_dir, "ckpt_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {}

        pattern = os.path.join(save_dir, "ckpt_*.pt")
        for p in glob.glob(pattern):
            name = os.path.basename(p)
            if name not in manifest:
                m = re.search(r"ckpt_(\d+)\.pt$", name)
                step_val = int(m.group(1)) if m else 0
                manifest[name] = {"step": step_val, "performance": float("-inf")}

        manifest[os.path.basename(path)] = {"step": step, "performance": performance}

        # Sort by performance descending, keep top max_keep
        entries = [(name, info["performance"]) for name, info in manifest.items()]
        entries_sorted = sorted(entries, key=lambda x: x[1], reverse=True)
        to_keep = {name for name, _ in entries_sorted[:max_keep]}
        to_remove = [name for name in manifest if name not in to_keep]

        for name in to_remove:
            old_path = os.path.join(save_dir, name)
            if os.path.exists(old_path):
                os.remove(old_path)
            del manifest[name]

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
