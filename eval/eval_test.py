"""Hydra entrypoint for single-checkpoint test/evaluation."""

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import Config
from crowd_nav.rl_policy_factory import get_rl_policy_class
from crowd_sim.utils import build_env
from crowd_sim.utils import polar_obs_dim_from_env_dim, relative_obs_dim_from_env_dim
from eval.policy_kwargs import build_eval_policy_kwargs, filter_policy_kwargs
from eval.eval_util import (
    RLEvalActorAdapter,
    build_obs_preprocess_fn,
    eval_policy,
    run_crossing_scenario,
)

METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcbfgamma_2nets": True,
    "rlcbfgamma_2nets_risk": True,
    "rlcvarbetaradius": True,
    "rlcvarbetaradius_2nets": True,
    "rlcvarbetaradius_2nets_risk": True,
    "rlcvarbetaradiusalpha_2nets": True,
    "rlcvarbetaradiusalpha_2nets_risk": True,
}


MAIN_DIR = str(ROOT_DIR)

ANNEAL_PARAM_KEYS = (
    "annealing_learning_alpha",
    "annealing_learning_beta",
    "annealing_learning_radius",
    "anneal_end_timesteps",
    "alpha_anneal_range",
    "beta_anneal_range",
    "radius_scale_anneal_range",
)


def _resolve_eval_save_path(actor_model: str, save_path: str | None, cfg: Config) -> str:
    if save_path is not None and str(save_path).strip():
        return str(save_path)
    actor_dir = os.path.dirname(os.path.abspath(actor_model))
    robot_type = str(cfg.robot.get("type", "robot"))
    obstacle_count = int(cfg.human.get("num_humans", 0))
    vmax = float(cfg.robot.get("vmax", 0.0))
    omega_max = float(cfg.robot.get("omega_max", 0.0))
    folder = f"{robot_type}_obs_{obstacle_count}_vmax_{vmax:g}_omegamax_{omega_max:g}"
    return os.path.join(actor_dir, folder)


def run_policy_test(
    env,
    actor_model,
    cfg,
    device,
    method,
    policy_kwargs=None,
    obs_topk=5,
    obs_farest_dist=5.0,
    needs_qp_relative=False,
    test_episodes=50,
    test_viz_episodes=20,
    base_seed=None,
    save_path=None,
    test_mode="eval",
    algo=None,
):
    print(f"Testing {actor_model}", flush=True)
    if actor_model == "":
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    policy_class = get_rl_policy_class(method)
    if algo is not None:
        print(f"Algorithm: {str(algo).upper()}, Policy: {policy_class.__name__}", flush=True)
    else:
        print(f"Policy: {policy_class.__name__}", flush=True)

    state = torch.load(actor_model, map_location=device)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    filtered_policy_kwargs = filter_policy_kwargs(policy_class, policy_kwargs)
    env_obs_dim = int(env.observation_space.shape[0])
    obs_dim = int(polar_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk))
    qp_obs_dim = int(relative_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk))
    act_dim = int(env.action_space.shape[0])

    if bool(needs_qp_relative):
        filtered_policy_kwargs["qp_obs_dim"] = qp_obs_dim

    print(f"Policy Args: {filtered_policy_kwargs}", flush=True)
    policy = policy_class(obs_dim, act_dim, **filtered_policy_kwargs).to(device)
    policy.load_state_dict(state)
    policy.eval()

    actor = RLEvalActorAdapter(policy, env.action_space, device)
    save_path = _resolve_eval_save_path(actor_model, save_path, cfg=cfg)
    obs_preprocess_fn = build_obs_preprocess_fn(
        obs_topk=int(obs_topk),
        obs_farest_dist=float(obs_farest_dist),
        needs_qp_relative=bool(needs_qp_relative),
    )

    mode = str(test_mode).lower().strip()
    if mode in ("eval", "both"):
        eval_policy(
            policy=actor,
            env=env,
            max_episodes=int(test_episodes),
            save_path=save_path,
            base_seed=base_seed,
            method=method,
            obs_preprocess_fn=obs_preprocess_fn,
            visualize_episodes=int(test_viz_episodes),
        )

    if mode in ("crossing", "both"):
        run_crossing_scenario(
            actor,
            env,
            save_path=save_path,
            obs_preprocess_fn=obs_preprocess_fn,
        )


def main(cfg_args: DictConfig):
    sim_cfg = Config()
    for key in ANNEAL_PARAM_KEYS:
        if key in cfg_args:
            setattr(sim_cfg, key, cfg_args[key])
    env_name = sim_cfg.env.get("name", "social_nav_var_num")
    render_mode = "human" if bool(cfg_args.render) else "rgb_array"
    env = build_env(env_name, render_mode=render_mode, config=sim_cfg)

    try:
        method_key = str(cfg_args.method).strip().lower()
        if method_key not in (
            "rl",
            "rlcbfgamma",
            "rlcbfgamma_2nets",
            "rlcbfgamma_2nets_risk",
            "rlcvarbetaradius",
            "rlcvarbetaradius_2nets",
            "rlcvarbetaradius_2nets_risk",
            "rlcvarbetaradiusalpha_2nets",
            "rlcvarbetaradiusalpha_2nets_risk",
        ):
            raise ValueError(
                f"Unsupported method '{cfg_args.method}'. "
                "Expected one of: rl, rlcbfgamma, rlcbfgamma_2nets, rlcbfgamma_2nets_risk, "
                "rlcvarbetaradius, rlcvarbetaradius_2nets, rlcvarbetaradius_2nets_risk, "
                "rlcvarbetaradiusalpha_2nets, rlcvarbetaradiusalpha_2nets_risk."
            )
        needs_qp_relative = bool(METHOD_NEEDS_QP_RELATIVE.get(method_key, False))
        test_mode = str(cfg_args.test_mode).strip().lower()
        if test_mode not in ("eval", "crossing", "both"):
            raise ValueError(
                f"Unsupported test_mode '{cfg_args.test_mode}'. Expected one of: eval, crossing, both."
            )
        policy_kwargs = build_eval_policy_kwargs(
            sim_cfg,
            cfg_args.method,
            nHidden1=int(cfg_args.nHidden1),
            nHidden21=int(cfg_args.nHidden21),
            nHidden22=int(cfg_args.nHidden22),
            alpha_hidden1=int(cfg_args.alpha_hidden1),
            alpha_hidden2=int(cfg_args.alpha_hidden2),
            qp_obs_dim=None,
        )
        device = torch.device("cpu")
        save_path_value = str(cfg_args.save_path).strip() if cfg_args.save_path is not None else ""
        algo_value = str(cfg_args.algo).strip() if cfg_args.algo is not None else ""
        run_policy_test(
            env=env,
            actor_model=str(cfg_args.actor_model),
            cfg=sim_cfg,
            device=device,
            method=str(cfg_args.method),
            policy_kwargs=policy_kwargs,
            obs_topk=int(cfg_args.obs_topk),
            obs_farest_dist=float(cfg_args.obs_farest_dist),
            needs_qp_relative=needs_qp_relative,
            test_episodes=int(cfg_args.test_ep),
            test_viz_episodes=int(cfg_args.test_viz_ep),
            base_seed=(None if cfg_args.eval_seed is None else int(cfg_args.eval_seed)),
            save_path=(save_path_value or None),
            test_mode=test_mode,
            algo=(algo_value or None),
        )
    finally:
        if hasattr(env, "close"):
            env.close()


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="eval_test",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
