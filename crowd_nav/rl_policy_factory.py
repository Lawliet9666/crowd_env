"""Factory helpers for RL policy classes."""


def get_rl_policy_class(method: str):
    method = (method or "").strip()
    if method == "rl":
        from rl.network import FCNet
        return FCNet
    if method == "rlcbfgamma":
        import rl.network_qpth as cbf_u_gamma
        return cbf_u_gamma.BarrierNet
    if method == "rlcbfgamma_2nets":
        import rl.network_qpth_2nets as cbf_u_gamma_2nets
        return cbf_u_gamma_2nets.BarrierNet
    if method == "rlcbfgamma_2nets_risk":
        import rl.network_qpth_2nets_risk as cbf_u_gamma_2nets_risk
        return cbf_u_gamma_2nets_risk.BarrierNet
    if method == "rlcvar":
        import rl.network_qpth_cvar_v0 as cvar_cbf_u
        return cvar_cbf_u.BarrierNet
    if method == "rlcvarbetaradius":
        import rl.network_qpth_cvar as cvar_cbf_u_beta_r
        return cvar_cbf_u_beta_r.BarrierNet
    if method == "rlcvarbetaradius_2nets":
        import rl.network_qpth_cvar_2nets as cvar_cbf_u_beta_r_2nets
        return cvar_cbf_u_beta_r_2nets.BarrierNet
    if method == "rlcvarbetaradius_2nets_risk":
        import rl.network_qpth_cvar_2nets_risk as cvar_cbf_u_beta_r_2nets_risk
        return cvar_cbf_u_beta_r_2nets_risk.BarrierNet
    if method in ("orca", "social_force"):
        raise ValueError(f"method '{method}' is supported in main_opt.py, not main_vec.py")
    raise ValueError(f"Unknown method {method}")
