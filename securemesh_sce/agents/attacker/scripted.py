# securemesh_sce/agents/attacker/scripted.py
"""Level 0: Deterministic scripted attacker.

Follows a fixed kill-chain script. Used as the reproducible baseline
and as an expert demonstrator for the BC attacker.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from abc import ABC, abstractmethod
from ...game.actions import AttackerAction, AttackerType, N_ATTACKER_ACTIONS


class BaseAgent(ABC):
    """Abstract base for all SCE agents."""

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        pass

    def update(self, transition: tuple):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def reset(self):
        pass


# =====================================================================
# Kill-chain scripts per attacker type
# =====================================================================

KILL_CHAIN_SCRIPTS = {
    AttackerType.OPPORTUNISTIC: [
        AttackerAction.RECON_SCAN,
        AttackerAction.RECON_SCAN,
        AttackerAction.AUTH_BRUTEFORCE,
        AttackerAction.AUTH_BRUTEFORCE,
        AttackerAction.AUTH_BRUTEFORCE,
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.MALWARE_DROP,
        AttackerAction.PERSIST_BACKDOOR,
        AttackerAction.AUTH_BRUTEFORCE,
        AttackerAction.EXPLOIT_IOT,
    ],
    AttackerType.STEALTH: [
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.NOOP,
        AttackerAction.EVADE_OBFUSCATE,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.NOOP,
        AttackerAction.AUTH_CREDENTIAL_STUFF,
        AttackerAction.EVADE_SLOWDOWN,
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.EVADE_OBFUSCATE,
        AttackerAction.PERSIST_C2,
    ],
    AttackerType.ADAPTIVE: [
        AttackerAction.RECON_SCAN,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.EXPLOIT_IOT,
        AttackerAction.LATERAL_MOVE,
        AttackerAction.MALWARE_DROP,
        AttackerAction.PERSIST_BACKDOOR,
        AttackerAction.PERSIST_C2,
        AttackerAction.EVADE_OBFUSCATE,
        AttackerAction.EXPLOIT_SERVICE,
    ],
    AttackerType.RESOURCE_AWARE: [
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.AUTH_CREDENTIAL_STUFF,
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.EVADE_SLOWDOWN,
        AttackerAction.PERSIST_C2,
        AttackerAction.NOOP,
        AttackerAction.NOOP,
        AttackerAction.EXPLOIT_IOT,
        AttackerAction.LATERAL_MOVE,
    ],
}


class ScriptedAttacker(BaseAgent):
    """Level 0: Deterministic scripted attacker.

    Follows a pre-defined kill chain that repeats cyclically.
    Each attacker type has a distinct script.
    """

    def __init__(
        self,
        attacker_type: AttackerType = AttackerType.OPPORTUNISTIC,
        noise: float = 0.0,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        attacker_type
            Which script to follow.
        noise
            Probability of random action override (ε-greedy noise).
        seed
            Random seed for noise.
        """
        self.attacker_type = attacker_type
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.script = KILL_CHAIN_SCRIPTS[attacker_type]
        self._step = 0

    def select_action(self, observation: np.ndarray) -> int:
        if self.noise > 0 and self.rng.random() < self.noise:
            return int(self.rng.randint(0, N_ATTACKER_ACTIONS))

        action = self.script[self._step % len(self.script)]
        self._step += 1
        return list(AttackerAction).index(action)

    def reset(self):
        self._step = 0

    def get_action_sequence(self, length: int = 100) -> List[int]:
        """Return the first `length` action indices from this script."""
        self.reset()
        obs_dummy = np.zeros(1)
        return [self.select_action(obs_dummy) for _ in range(length)]
