import os
from datetime import datetime

import hydra
from omegaconf import DictConfig

from config.config import Config
from controller.robot_controller_factory import build_robot_controller
from crowd_sim.utils import build_env, dump_test_config
from eval.eval_util import eval_policy, run_crossing_scenario


MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_METHODS = {
    "orca",
    "social_force",
    "nominal",
    "cbfqp",
    "cvarqp",
    "adapcvarqp",
    "drcvarqp",
}

VALID_TEST_MODES = {"eval", "crossing", "both"}


def _prepare_save_dirs(cfg: DictConfig, robot_type: str):
    base_dir = os.path.join(
        "trained_models",
        str(cfg.model_folder),
        f"{robot_type}_{cfg.method}",
    )
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return base_dir, run_dir


def main(cfg: DictConfig):
    method = str(cfg.method).strip().lower()
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unsupported method '{cfg.method}'. Expected one of: {sorted(VALID_METHODS)}."
        )

    test_mode = str(cfg.test_mode).strip().lower()
    if test_mode not in VALID_TEST_MODES:
        raise ValueError(
            f"Unsupported test_mode '{cfg.test_mode}'. Expected one of: {sorted(VALID_TEST_MODES)}."
        )

    config = Config()
    env_name = config.env.get("name", "social_nav_var_num")

    render_mode = "human" if bool(cfg.render) else "rgb_array"
    env = build_env(env_name, render_mode=render_mode, config=config)
    controller = build_robot_controller(method, config, env)
    base_save_dir, save_dir = _prepare_save_dirs(cfg, config.robot.type)
    dump_test_config(save_dir, config, extra={"eval_seed": cfg.eval_seed, "method": method})
    print(f"Evaluation base dir: {base_save_dir}", flush=True)
    print(f"Evaluation outputs: {save_dir}", flush=True)

    eval_seed = int(cfg.seed) if cfg.eval_seed is None else int(cfg.eval_seed)

    try:
        if test_mode in ("eval", "both"):
            eval_policy(
                policy=controller,
                env=env,
                max_episodes=int(cfg.test_ep),
                save_path=save_dir,
                base_seed=eval_seed,
                visualize_episodes=int(cfg.test_viz_ep),
            )
        if test_mode in ("crossing", "both"):
            run_crossing_scenario(policy=controller, env=env, save_path=save_dir)
    finally:
        env.close()


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="main_opt",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
