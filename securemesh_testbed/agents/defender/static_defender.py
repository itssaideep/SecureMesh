# securemesh_testbed/agents/defender/static_defender.py
"""Static / non-adaptive defender baseline.

Uses fixed heuristic rules:
  * Block the attacker IP after *N* consecutive IDS alerts.
  * Isolate any service that appears compromised.
  * Otherwise NOOP.

This is the simplest baseline — a defender that never learns.
"""

from ..base_agent import BaseAgent
from ...game.actions import DefenderAction


class StaticDefender(BaseAgent):
    """Rule-based defender with a fixed policy.

    Parameters
    ----------
    block_after : int
        Number of cumulative alerts before issuing BLOCK_IP.
    """

    def __init__(self, block_after: int = 5):
        self.block_after = block_after
        self._alert_count = 0
        self._blocked = False

    def select_action(self, observation) -> int:
        # The observation is a flat array; we inspect specific indices
        # to decide what to do.  By convention the first values encode
        # host/service state (see state.py).
        import numpy as np
        obs = np.asarray(observation, dtype=np.float32).flatten()

        # Simple heuristic: if any service shows as compromised (value 1.0
        # at even indices after the first) → isolate.
        compromised_detected = False
        if len(obs) >= 3:
            # service compromised flags are at indices 1, 4, 7, … in the
            # current flattening scheme (isolated, compromised, sessions)
            for i in range(1, len(obs), 3):
                if i < len(obs) and obs[i] >= 1.0:
                    compromised_detected = True
                    break

        if compromised_detected:
            self._alert_count += 1

        if self._alert_count >= self.block_after and not self._blocked:
            self._blocked = True
            return DefenderAction.BLOCK_IP.value

        if compromised_detected:
            return DefenderAction.ISOLATE_SERVICE.value

        return DefenderAction.NOOP.value
