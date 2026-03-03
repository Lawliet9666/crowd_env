"""Factory map for human policies used by the environment."""

from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.potential_field import POTENTIAL_FIELD


def none_policy():
    return None


try:
    from crowd_nav.policy.orca import ORCA
except ModuleNotFoundError:
    ORCA = None


def _orca_unavailable(*args, **kwargs):
    raise ModuleNotFoundError(
        "ORCA policy requested but optional dependency is missing. "
        "Install Python-RVO2 (rvo2) to use ORCA."
    )


human_policy_factory = {
    "orca": ORCA if ORCA is not None else _orca_unavailable,
    "none": none_policy,
    "social_force": SOCIAL_FORCE,
    "potential_field": POTENTIAL_FIELD,
}


def get_human_policy_factory():
    return human_policy_factory
