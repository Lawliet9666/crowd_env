"""Shared policy class registry for RL actors."""

def get_policy_class(method: str):
    if method == "rl":
        from rl.network import FCNet
        return FCNet
    if method == "rlcbfgamma":
        import rl.network_qpth_v1 as cbf_u_gamma
        return cbf_u_gamma.BarrierNet
    if method == "rlcvarbetaradius":
        import rl.network_qpth_cvar_v2 as cvar_cbf_u_beta_r
        return cvar_cbf_u_beta_r.BarrierNet
    if method in ("orca", "social_force"):
        raise ValueError(f"method '{method}' is supported in main_opt.py, not main_vec.py")
    raise ValueError(f"Unknown method {method}")
