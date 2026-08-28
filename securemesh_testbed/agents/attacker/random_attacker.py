# securemesh_testbed/agents/attacker/random_attacker.py
"""Baseline attacker that selects actions uniformly at random.

Used as a lower-bound reference — any learned policy should
outperform random action selection.
"""

import numpy as np
from ..base_agent import BaseAgent


class RandomAttacker(BaseAgent):
    """Uniform-random attacker (baseline)."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)

    def select_action(self, observation) -> int:
        return int(self.rng.randint(self.n_actions))
