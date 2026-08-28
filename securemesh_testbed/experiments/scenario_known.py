# securemesh_testbed/experiments/scenario_known.py
"""Known-attack scenario runner.

Uses the KNOWN_ATTACKS scenario configuration (signature-based IDS,
standard vulnerability levels) and runs the specified attacker/defender
pair for N episodes.
"""

from ..config.scenarios import KNOWN_ATTACKS, ScenarioConfig


def get_scenario(seed: int = 42) -> ScenarioConfig:
    """Return a copy of the known-attacks scenario with the given seed."""
    import copy
    sc = copy.deepcopy(KNOWN_ATTACKS)
    sc.seed = seed
    return sc
