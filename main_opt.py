import os
import shutil
from datetime import datetime
from config.arguments import get_args
from config.config import Config
from controller.robot_controller_factory import build_robot_controller
from eval.eval_util import eval_policy, run_crossing_scenario
from crowd_sim.utils import build_env, dump_test_config

def _prepare_save_dirs(args, robot_type):
    base_dir = os.path.join(
        "trained_models",
        args.model_folder,
        f"{robot_type}_{args.method}",
    )
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return base_dir, run_dir



def main(args):

    config = Config()
    env_name = config.env.get("name", "social_nav_var_num")

    render_mode = "human" if args.render else "rgb_array"
    env = build_env(env_name, render_mode=render_mode, config=config)
    controller = build_robot_controller(args.method, config, env)
    base_save_dir, save_dir = _prepare_save_dirs(args, config.robot.type)
    dump_test_config(save_dir, config,extra={"eval_seed": args.eval_seed, "method": args.method})
    # dump_train_config(save_dir, args, config)
    print(f"Evaluation base dir: {base_save_dir}", flush=True)
    print(f"Evaluation outputs: {save_dir}", flush=True)

    eval_seed = args.seed if getattr(args, "eval_seed", None) is None else args.eval_seed

    try:
        if args.test_mode in ("eval", "both"):
            eval_policy(
                policy=controller,
                env=env,
                max_episodes=args.test_ep,
                save_path=save_dir,
                base_seed=eval_seed,
                visualize_episodes=args.test_viz_ep,
            )
        if args.test_mode in ("crossing", "both"):
            run_crossing_scenario(policy=controller, env=env, save_path=save_dir)
    finally:
        env.close()



if __name__ == "__main__":
    arg = get_args()
    main(arg)
