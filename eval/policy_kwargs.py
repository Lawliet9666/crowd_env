"""Shared policy-kwargs builders for eval scripts."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import numpy as np


_CVAR_METHODS = {"rlcvarbetaradius", "rlcvarbetaradius_2nets", "rlcvarbetaradius_2nets_risk"}


def _radius_scalar(v: Any) -> float:
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(np.max(arr))
    return float(v)


def build_eval_policy_kwargs(
    cfg,
    method: str,
    *,
    nHidden1: int,
    nHidden21: int,
    nHidden22: int,
    alpha_hidden1: int,
    alpha_hidden2: int,
    qp_obs_dim: Optional[int] = None,
) -> Dict[str, Any]:
    safe_dist = (
        float(cfg.controller.safety_margin)
        + _radius_scalar(cfg.human.radius)
        + _radius_scalar(cfg.robot.radius)
    )
    kwargs: Dict[str, Any] = {
        "robot_type": cfg.robot.type,
        "safe_dist": safe_dist,
        "alpha": float(cfg.controller.cbf_alpha),
        "beta": float(cfg.controller.cvar_beta),
        "vmax": float(cfg.robot.vmax),
        "amax": float(cfg.robot.amax),
        "omega_max": float(cfg.robot.omega_max),
        "nHidden1": int(nHidden1),
        "nHidden21": int(nHidden21),
        "nHidden22": int(nHidden22),
        "alpha_hidden1": int(alpha_hidden1),
        "alpha_hidden2": int(alpha_hidden2),
    }

    method_key = str(method).strip().lower()
    if method_key in _CVAR_METHODS:
        gmm_cfg = dict(cfg.human.get("gmm", {}))
        kwargs["gmm_weights"] = gmm_cfg.get("weights")
        kwargs["gmm_stds"] = gmm_cfg.get("stds")
        kwargs["gmm_lateral_ratio"] = gmm_cfg.get("lateral_ratio", 0.3)

    if qp_obs_dim is not None:
        kwargs["qp_obs_dim"] = int(qp_obs_dim)

    return kwargs


def filter_policy_kwargs(policy_class, policy_kwargs):
    kwargs = dict(policy_kwargs or {})
    try:
        sig = inspect.signature(policy_class.__init__)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        return {}
