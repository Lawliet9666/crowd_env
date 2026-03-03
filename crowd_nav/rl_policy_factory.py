"""Factory helpers for RL policy classes."""


def get_rl_policy_class(method: str):
    method = (method or "").strip()
    if method == "rl":
        from rl.network import FCNet
        return FCNet
    if method == "rlcbf":
        import rl.network_qpth_v0 as cbf_u
        return cbf_u.BarrierNet
    if method == "rlcbfgamma":
        import rl.network_qpth_v1 as cbf_u_gamma
        return cbf_u_gamma.BarrierNet
    if method == "rlcbfgamma_v2":
        import rl.network_qpth_v2 as cbf_u_gamma_v2
        return cbf_u_gamma_v2.BarrierNet
    if method == "rlcvar":
        import rl.network_qpth_cvar_v0 as cvar_cbf
        return cvar_cbf.BarrierNet
    if method == "rlcvarbeta":
        import rl.network_qpth_cvar_v1 as cvar_cbf_u_beta
        return cvar_cbf_u_beta.BarrierNet
    if method == "rlcvarbetaradius":
        import rl.network_qpth_cvar_v2 as cvar_cbf_u_beta_r
        return cvar_cbf_u_beta_r.BarrierNet
    if method in ("orca", "social_force"):
        raise ValueError(f"method '{method}' is supported in main_opt.py, not main_vec.py")
    raise ValueError(f"Unknown method {method}")


def get_policy_class(method: str):
    """Backward-compatible alias."""
    return get_rl_policy_class(method)
