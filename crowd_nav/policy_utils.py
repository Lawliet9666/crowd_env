"""Shared policy class registry for RL actors."""

def get_policy_class(method: str):
    if method == "rl":
        from rl.network import FCNet
        return FCNet
    if method == "rlatt":
        from rl.network_att import FCNet as FCNetAtt
        return FCNetAtt
    if method == "rldeepsets":
        from rl.network_deepsets import DeepSetsPolicy
        return DeepSetsPolicy
    if method == "rldeepsetscbfgamma":
        import rl.network_deepsets_qpth_v1 as deepsets_cbf_u_gamma
        return deepsets_cbf_u_gamma.BarrierNet
    if method == "rldeepsetscvarbetaradius":
        import rl.network_deepsets_qpth_cvar_v2 as deepsets_cvar_cbf_u_beta_r
        return deepsets_cvar_cbf_u_beta_r.BarrierNet
    if method == "rlcbf":
        import rl.network_qpth_v0 as cbf_u
        return cbf_u.BarrierNet
    if method == "rlcbfgamma":
        import rl.network_qpth_v1 as cbf_u_gamma
        return cbf_u_gamma.BarrierNet
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
