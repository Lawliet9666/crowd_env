import hydra
import os
from pprint import pprint
from omegaconf import OmegaConf



main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(main_dir)

@hydra.main(config_path=os.path.join(main_dir, "new_rl", "config"), config_name="mujoco_ppo_base.yaml", version_base=None)
def main(config):
    pprint(config, indent=4)
    
    # save config to yaml file
    with open("sandbox/config.yaml", "w") as f:
        OmegaConf.save(config, f)


if __name__ == "__main__":
    main()
