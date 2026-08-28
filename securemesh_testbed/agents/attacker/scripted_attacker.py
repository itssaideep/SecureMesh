# securemesh_testbed/agents/attacker/scripted_attacker.py
"""Rule-based multi-phase attacker.

Implements the canonical kill-chain:
  1. Reconnaissance  (RECON_SCAN, RECON_FINGERPRINT)
  2. Authentication  (AUTH_BRUTEFORCE, AUTH_CREDENTIAL_STUFF)
  3. Exploitation    (EXPLOIT_SERVICE, EXPLOIT_IOT)
  4. Persistence     (MALWARE_DROP, PERSIST_BACKDOOR, PERSIST_C2)
  5. Evasion / Idle  (EVADE_OBFUSCATE, LATERAL_MOVE)

The attacker deterministically cycles through these phases,
spending a configurable number of steps in each phase.
"""

from ..base_agent import BaseAgent
from ...game.actions import AttackerAction


# Phase definitions: each phase maps to a list of action enum values
_PHASES = [
    # Phase 0 — Recon
    [AttackerAction.RECON_SCAN, AttackerAction.RECON_FINGERPRINT],
    # Phase 1 — Auth
    [AttackerAction.AUTH_BRUTEFORCE, AttackerAction.AUTH_CREDENTIAL_STUFF],
    # Phase 2 — Exploit
    [AttackerAction.EXPLOIT_SERVICE, AttackerAction.EXPLOIT_IOT],
    # Phase 3 — Persist
    [AttackerAction.MALWARE_DROP, AttackerAction.PERSIST_BACKDOOR,
     AttackerAction.PERSIST_C2],
    # Phase 4 — Evasion / lateral
    [AttackerAction.EVADE_OBFUSCATE, AttackerAction.LATERAL_MOVE],
]


class ScriptedAttacker(BaseAgent):
    """Deterministic kill-chain attacker.

    Parameters
    ----------
    steps_per_phase : int
        Number of environment steps to spend in each kill-chain phase
        before advancing to the next one.  After the last phase the
        attacker loops back to phase 0.
    """

    def __init__(self, steps_per_phase: int = 5):
        self.steps_per_phase = steps_per_phase
        self._step = 0

    def select_action(self, observation) -> int:
        phase_idx = (self._step // self.steps_per_phase) % len(_PHASES)
        within = self._step % self.steps_per_phase
        actions = _PHASES[phase_idx]
        action = actions[within % len(actions)]
        self._step += 1
        return action.value  # integer index expected by the environment
