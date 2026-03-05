"""Hydra entrypoint for single-checkpoint test/evaluation."""

import inspect
import os
import sys
from argparse import Namespace
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import Config
from crowd_nav.rl_policy_factory import get_rl_policy_class
from crowd_sim.utils import build_env
from crowd_sim.utils import polar_obs_dim_from_env_dim, relative_obs_dim_from_env_dim
from eval.eval_util import RLEvalActorAdapter, eval_policy, run_crossing_scenario

METHOD_NEEDS_QP_RELATIVE = {
    "rlcbfgamma": True,
    "rlcbfgamma_2nets": True,
    "rlcvarbetaradius": True,
    "rlcvarbetaradius_2nets": True,
}


MAIN_DIR = str(ROOT_DIR)


def _filter_policy_kwargs(policy_class, policy_kwargs):
    kwargs = dict(policy_kwargs or {})
    try:
        sig = inspect.signature(policy_class.__init__)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        return {}


def run_policy_test(
    env,
    actor_model,
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

    filtered_policy_kwargs = _filter_policy_kwargs(policy_class, policy_kwargs)
    env_obs_dim = int(env.observation_space.shape[0])
    obs_dim = int(polar_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk))
    qp_obs_dim = int(relative_obs_dim_from_env_dim(env_obs_dim, topk=obs_topk))
    act_dim = int(env.action_space.shape[0])

    if bool(needs_qp_relative):
        filtered_policy_kwargs["qp_obs_dim"] = qp_obs_dim

    print(f"Policy Args: {filtered_policy_kwargs}", flush=True)
    policy = policy_class(obs_dim, act_dim, **filtered_policy_kwargs).to(device)
    policy.load_state_dict(torch.load(actor_model, map_location=device))
    policy.eval()

    actor = RLEvalActorAdapter(policy, env.action_space, device)
    save_path = save_path or os.path.dirname(actor_model)

    mode = str(test_mode).lower().strip()
    if mode in ("eval", "both"):
        eval_policy(
            policy=actor,
            env=env,
            max_episodes=int(test_episodes),
            save_path=save_path,
            base_seed=base_seed,
            method=method,
            obs_topk=int(obs_topk),
            obs_farest_dist=float(obs_farest_dist),
            needs_qp_relative=bool(needs_qp_relative),
            visualize_episodes=int(test_viz_episodes),
        )

    if mode in ("crossing", "both"):
        run_crossing_scenario(
            actor,
            env,
            save_path=save_path,
            obs_topk=int(obs_topk),
            obs_farest_dist=float(obs_farest_dist),
            needs_qp_relative=bool(needs_qp_relative),
        )


def _resolve_bool_or_auto(text, *, default):
    val = str(text).strip().lower()
    if val == "auto":
        return bool(default)
    if val in ("1", "true", "t", "yes", "y", "on"):
        return True
    if val in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean option '{text}'. Use one of: auto, true, false")


def _build_policy_kwargs_from_config(cfg, method):
    gmm_cfg = dict(cfg.human.get("gmm", {}))
    kwargs = {
        "robot_type": cfg.robot["type"],
        "safe_dist": cfg.controller["safety_margin"] + cfg.human["radius"] + cfg.robot["radius"],
        "alpha": cfg.controller["cbf_alpha"],
        "beta": cfg.controller["cvar_beta"],
        "vmax": cfg.robot["vmax"],
        "amax": cfg.robot["amax"],
        "omega_max": cfg.robot["omega_max"],
    }
    if str(method).strip().lower() in ("rlcvarbetaradius", "rlcvarbetaradius_2nets"):
        kwargs["gmm_weights"] = gmm_cfg.get("weights")
        kwargs["gmm_stds"] = gmm_cfg.get("stds")
        kwargs["gmm_lateral_ratio"] = gmm_cfg.get("lateral_ratio", 0.3)
    return kwargs


def main(args):
    cfg = Config()
    if str(args.robot_type).strip():
        robot_type = str(args.robot_type).strip()
        if robot_type not in ("single_integrator", "unicycle"):
            raise ValueError(
                f"Unsupported robot_type '{robot_type}'. Expected one of: '', single_integrator, unicycle."
            )
        cfg.robot.type = robot_type
    env_name = args.env_name if args.env_name else cfg.env.get("name", "social_nav_var_num")
    render_mode = "human" if args.render else "rgb_array"
    env = build_env(env_name, render_mode=render_mode, config=cfg)

    try:
        method_key = str(args.method).strip().lower()
        if method_key not in (
            "rl",
            "rlcbfgamma",
            "rlcbfgamma_2nets",
            "rlcvarbetaradius",
            "rlcvarbetaradius_2nets",
        ):
            raise ValueError(
                f"Unsupported method '{args.method}'. "
                "Expected one of: rl, rlcbfgamma, rlcbfgamma_2nets, rlcvarbetaradius, rlcvarbetaradius_2nets."
            )
        needs_default = METHOD_NEEDS_QP_RELATIVE.get(method_key, False)
        needs_qp_relative = _resolve_bool_or_auto(args.needs_qp_relative, default=needs_default)
        test_mode = str(args.test_mode).strip().lower()
        if test_mode not in ("eval", "crossing", "both"):
            raise ValueError(
                f"Unsupported test_mode '{args.test_mode}'. Expected one of: eval, crossing, both."
            )
        policy_kwargs = _build_policy_kwargs_from_config(cfg, args.method)
        device = torch.device("cpu")
        run_policy_test(
            env=env,
            actor_model=args.actor_model,
            device=device,
            method=args.method,
            policy_kwargs=policy_kwargs,
            obs_topk=args.obs_topk,
            obs_farest_dist=args.obs_farest_dist,
            needs_qp_relative=needs_qp_relative,
            test_episodes=args.test_ep,
            test_viz_episodes=args.test_viz_ep,
            base_seed=args.eval_seed,
            save_path=(args.save_path or None),
            test_mode=test_mode,
            algo=(args.algo or None),
        )
    finally:
        if hasattr(env, "close"):
            env.close()


def _to_eval_test_args(cfg: DictConfig) -> Namespace:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Hydra config must resolve to a flat dict for eval_test.")
    cfg_dict.pop("hydra", None)
    return Namespace(**cfg_dict)


@hydra.main(
    config_path=os.path.join(MAIN_DIR, "config"),
    config_name="eval_test",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    args = _to_eval_test_args(cfg)
    main(args)


if __name__ == "__main__":
    hydra_main()
