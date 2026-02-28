"""Shared policy class registry for RL actors."""

from rl.network import FCNet
from rl.network_att import FCNet as FCNetAtt
from rl.network_deepsets import DeepSetsPolicy
import rl.network_deepsets_qpth_v1 as deepsets_cbf_u_gamma
import rl.network_deepsets_qpth_cvar_v2 as deepsets_cvar_cbf_u_beta_r
import rl.network_qpth_v0 as cbf_u
import rl.network_qpth_v1 as cbf_u_gamma
import rl.network_qpth_cvar_v0 as cvar_cbf
import rl.network_qpth_cvar_v1 as cvar_cbf_u_beta
import rl.network_qpth_cvar_v2 as cvar_cbf_u_beta_r


def get_policy_class(method: str):
    if method == "rl":
        return FCNet
    if method == "rlatt":
        return FCNetAtt
    if method == "rldeepsets":
        return DeepSetsPolicy
    if method == "rldeepsetscbfgamma":
        return deepsets_cbf_u_gamma.BarrierNet
    if method == "rldeepsetscvarbetaradius":
        return deepsets_cvar_cbf_u_beta_r.BarrierNet
    if method == "rlcbf":
        return cbf_u.BarrierNet
    if method == "rlcbfgamma":
        return cbf_u_gamma.BarrierNet
    if method == "rlcvar":
        return cvar_cbf.BarrierNet
    if method == "rlcvarbeta":
        return cvar_cbf_u_beta.BarrierNet
    if method == "rlcvarbetaradius":
        return cvar_cbf_u_beta_r.BarrierNet
    if method in ("orca", "social_force"):
        raise ValueError(f"method '{method}' is supported in main_opt.py, not main_vec.py")
    raise ValueError(f"Unknown method {method}")
