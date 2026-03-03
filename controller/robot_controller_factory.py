"""Factory for non-RL robot controllers."""

from controller.adap_cvarbf_qp import AdaptiveCVaRBFQPController
from controller.cbf_qp import CBFQPController
from controller.cvarbf_qp import CVaRBFQPController
from controller.drcvarbf_qp import CVaRBFQPController as DRCVaRBFQPController
from controller.nominal_controller import NominalController
from crowd_nav.policy.social_force_helper import SocialForceController


def build_robot_controller(method, config, env):
    if method == "orca":
        try:
            from crowd_nav.policy.orca_helper import ORCAController
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Method 'orca' requested but optional dependency is missing. "
                "Install Python-RVO2 (rvo2) to use ORCA."
            ) from exc
        return ORCAController(config_file=config, env=env)

    if method == "social_force":
        return SocialForceController(config_file=config, env=env)

    if method == "nominal":
        return NominalController(config_file=config, env=env)

    if method == "cbfqp":
        return CBFQPController(config_file=config, env=env)

    if method == "cvarqp":
        return CVaRBFQPController(config_file=config, env=env)

    if method == "adapcvarqp":
        return AdaptiveCVaRBFQPController(config_file=config, env=env)

    if method == "drcvarqp":
        return DRCVaRBFQPController(config_file=config, env=env)

    raise ValueError(f"Unknown controller method: {method}")

