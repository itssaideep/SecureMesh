# securemesh_sce/agents/defender/static.py
"""Defender 1: Static rule-based defender.

Deterministic rules that respond to observed network state.
Serves as the non-adaptive baseline.
"""

from __future__ import annotations

import numpy as np
from ..attacker.scripted import BaseAgent
from ...game.actions import DefenderAction, N_DEFENDER_ACTIONS


class StaticDefender(BaseAgent):
    """Defender 1: Rule-based defender.

    Applies threshold-based rules:
    - High scan rate → increase monitoring
    - Auth attempt detected → rate limit
    - Exploit detected → isolate service
    - Compromised host → block IP
    - Otherwise → monitor
    """

    def __init__(
        self,
        scan_threshold: float = 0.5,
        auth_threshold: float = 0.3,
        exploit_threshold: float = 0.2,
        compromise_threshold: float = 0.1,
    ):
        self.scan_threshold = scan_threshold
        self.auth_threshold = auth_threshold
        self.exploit_threshold = exploit_threshold
        self.compromise_threshold = compromise_threshold

    def select_action(self, observation: np.ndarray) -> int:
        obs = np.asarray(observation, dtype=np.float32)

        # Heuristic: look at observable signal levels in the state vector
        # The observation layout is [system_state, belief, history, risk]
        # We use the history component (if available) to make decisions

        if len(obs) > 20:
            # Extract history features (these start after system_state + belief)
            # Rough heuristic based on observation magnitude
            avg = obs.mean()
            mx = obs.max()

            if mx > self.compromise_threshold * 10:
                return list(DefenderAction).index(DefenderAction.BLOCK_IP)
            elif mx > self.exploit_threshold * 10:
                return list(DefenderAction).index(DefenderAction.ISOLATE_SERVICE)
            elif avg > self.auth_threshold:
                return list(DefenderAction).index(DefenderAction.RATE_LIMIT)
            elif avg > self.scan_threshold * 0.5:
                return list(DefenderAction).index(DefenderAction.INCREASE_MONITORING)

        return list(DefenderAction).index(DefenderAction.MONITOR)
