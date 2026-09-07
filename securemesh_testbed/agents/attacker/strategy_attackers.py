# securemesh_testbed/agents/attacker/strategy_attackers.py
"""Deterministic attacker strategies for controlled experiments.

Three named strategies serve as the scientific control group against which
RL and LLM attackers are compared.  Each strategy encodes a fixed action
sequence with distinct behavioural profiles:

  * **Aggressive** — fast escalation, high detection risk
  * **Stealthy** — slow, evasive, low detection risk
  * **ReconHeavy** — extended reconnaissance, minimal exploitation

These strategies are directly referenced in SCE experiment YAML files and
the experiment matrix runner.
"""

from __future__ import annotations

from ..base_agent import BaseAgent
from ...game.actions import AttackerAction


class AggressiveAttacker(BaseAgent):
    """Fast-cycling kill-chain: recon → brute-force → exploit → persist.

    Short dwell time per phase (2 steps).  Maximises offensive throughput
    at the cost of high detection probability.  Represents an attacker
    who prioritises speed over stealth.
    """

    _SEQUENCE = [
        # Phase 0 — Quick recon
        AttackerAction.RECON_SCAN,
        AttackerAction.RECON_FINGERPRINT,
        # Phase 1 — Immediate brute-force
        AttackerAction.AUTH_BRUTEFORCE,
        AttackerAction.AUTH_BRUTEFORCE,
        # Phase 2 — Exploit
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.EXPLOIT_IOT,
        # Phase 3 — Drop malware, persist
        AttackerAction.MALWARE_DROP,
        AttackerAction.PERSIST_BACKDOOR,
        AttackerAction.PERSIST_C2,
        # Phase 4 — Lateral movement
        AttackerAction.LATERAL_MOVE,
    ]

    def __init__(self):
        self._step = 0

    def select_action(self, observation) -> int:
        action = self._SEQUENCE[self._step % len(self._SEQUENCE)]
        self._step += 1
        return action.value


class StealthyAttacker(BaseAgent):
    """Slow, evasive attacker: recon → wait → fingerprint → credential stuff.

    Long dwell time with frequent NOOP/evasion steps.  Attempts to stay
    below IDS detection thresholds.  Represents an advanced persistent
    threat (APT) profile.
    """

    _SEQUENCE = [
        # Phase 0 — Slow recon with pauses
        AttackerAction.RECON_SCAN,
        AttackerAction.NOOP,
        AttackerAction.NOOP,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.NOOP,
        AttackerAction.EVADE_SLOWDOWN,
        # Phase 1 — Careful credential stuffing
        AttackerAction.AUTH_CREDENTIAL_STUFF,
        AttackerAction.NOOP,
        AttackerAction.EVADE_OBFUSCATE,
        AttackerAction.AUTH_CREDENTIAL_STUFF,
        AttackerAction.NOOP,
        AttackerAction.NOOP,
        # Phase 2 — Single exploit attempt
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.EVADE_OBFUSCATE,
        AttackerAction.NOOP,
        # Phase 3 — Slow persistence
        AttackerAction.PERSIST_BACKDOOR,
        AttackerAction.EVADE_SLOWDOWN,
        AttackerAction.NOOP,
        AttackerAction.NOOP,
    ]

    def __init__(self):
        self._step = 0

    def select_action(self, observation) -> int:
        action = self._SEQUENCE[self._step % len(self._SEQUENCE)]
        self._step += 1
        return action.value


class ReconHeavyAttacker(BaseAgent):
    """Extended reconnaissance with minimal exploitation.

    Spends 70%+ of steps on scanning and fingerprinting.  Only attempts
    a single exploitation path after thorough reconnaissance.  Represents
    an attacker focused on intelligence gathering rather than immediate
    compromise.
    """

    _SEQUENCE = [
        # Phase 0 — Extended reconnaissance
        AttackerAction.RECON_SCAN,
        AttackerAction.RECON_SCAN,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.RECON_SCAN,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.RECON_SCAN,
        # Phase 1 — Targeted credential attempt
        AttackerAction.AUTH_CREDENTIAL_STUFF,
        AttackerAction.RECON_SCAN,
        # Phase 2 — Single exploit
        AttackerAction.EXPLOIT_SERVICE,
        AttackerAction.RECON_FINGERPRINT,
        AttackerAction.RECON_SCAN,
    ]

    def __init__(self):
        self._step = 0

    def select_action(self, observation) -> int:
        action = self._SEQUENCE[self._step % len(self._SEQUENCE)]
        self._step += 1
        return action.value
