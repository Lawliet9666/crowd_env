policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.potential_field import POTENTIAL_FIELD

try:
    from crowd_nav.policy.orca import ORCA
except ModuleNotFoundError:
    ORCA = None


def _orca_unavailable(*args, **kwargs):
    raise ModuleNotFoundError(
        "ORCA policy requested but optional dependency is missing. "
        "Install Python-RVO2 (rvo2) to use ORCA."
    )

policy_factory['orca'] = ORCA if ORCA is not None else _orca_unavailable
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['potential_field'] = POTENTIAL_FIELD
