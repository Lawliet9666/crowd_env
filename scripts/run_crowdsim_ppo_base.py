"""Run PPO training with Hydra config.

WANDB_API_KEY=xxx python scripts/run_mujoco_ppo_base.py run_name=testest wandb_entity=daiyp_umich wandb_project=crowdsim_ppo
"""
import hydra
import os

from omegaconf import OmegaConf
from new_rl.trainer.ppo_base_trainer import PPOBaseTrainer

# Register resolver for arithmetic in config (e.g. obs_dim: ${math:3+${env.topk}*5})
OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(
    config_path=os.path.join(main_dir, "new_rl", "config"),
    config_name="crowdsim_ppo_base.yaml",
    version_base=None,
)
def main(config):
    assert config.run_name is not None, "run_name must be set"
    assert config.wandb_entity is not None and config.wandb_project is not None, "wandb_entity and wandb_project must be set"

    trainer = PPOBaseTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
