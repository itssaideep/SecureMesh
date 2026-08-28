# securemesh_testbed/experiments/scenario_zeroday.py
"""Zero-day / unseen-attack scenario runner.

Uses the ZERO_DAY scenario configuration (anomaly-based IDS only,
higher vulnerability levels) to test the defender's ability to
handle novel attack patterns.
"""

from ..config.scenarios import ZERO_DAY, ScenarioConfig


def get_scenario(seed: int = 123) -> ScenarioConfig:
    """Return a copy of the zero-day scenario with the given seed."""
    import copy
    sc = copy.deepcopy(ZERO_DAY)
    sc.seed = seed
    return sc
