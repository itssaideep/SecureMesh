# securemesh_sce/game/transitions.py
"""State transitions, exploit consequences, and service degradation.

Encapsulates the dynamics of the Bayesian Markov game:
- Attacker action resolution (success, detection, impact)
- Defender action resolution (service_down, isolation, restoration)
- Service degradation under sustained attack
- IoT availability/security trade-offs
"""

from __future__ import annotations

import random
from typing import Tuple, Dict, Any, Optional

from .actions import (
    AttackerAction, DefenderAction, AttackerType,
    ATTACK_IMPACT_WEIGHTS,
    SecurityStage,
)
from .rewards import StepOutcome, DEFENDER_ACTION_COSTS


# =====================================================================
# Attacker-type likelihood profiles  P(action | θ)
# =====================================================================
# Rows: attacker types, Cols: attacker actions
# These hand-crafted priors encode behavioural signatures.

import numpy as np

_ATYPES = list(AttackerType)
_AACTIONS = list(AttackerAction)

# P(action | attacker_type) — each row sums to 1.0
ATTACKER_LIKELIHOOD_TABLE = np.array([
    # RECON_SCAN  RECON_FP  AUTH_BF  AUTH_CS  EXPLOIT_SVC  EXPLOIT_IOT
    # MAL_DROP  PERSIST_BD  PERSIST_C2  EVADE_OBF  EVADE_SLOW  LAT_MOVE  NOOP
    # --- opportunistic (noisy, brute-force heavy) ---
    [0.15, 0.05, 0.22, 0.10, 0.15, 0.10, 0.05, 0.03, 0.02, 0.02, 0.01, 0.05, 0.05],
    # --- stealth (slow, evasive, patient) ---
    [0.06, 0.10, 0.03, 0.05, 0.08, 0.05, 0.04, 0.06, 0.05, 0.18, 0.15, 0.08, 0.07],
    # --- adaptive (exploit-heavy, persistent, adjusts) ---
    [0.08, 0.08, 0.06, 0.08, 0.18, 0.12, 0.10, 0.10, 0.08, 0.04, 0.02, 0.04, 0.02],
    # --- resource_aware (efficient, targeted, budget-conscious) ---
    [0.10, 0.12, 0.04, 0.06, 0.14, 0.08, 0.06, 0.08, 0.06, 0.06, 0.06, 0.06, 0.08],
], dtype=np.float32)

assert ATTACKER_LIKELIHOOD_TABLE.shape == (len(_ATYPES), len(_AACTIONS))
# Verify rows sum to 1
for i, row in enumerate(ATTACKER_LIKELIHOOD_TABLE):
    assert abs(row.sum() - 1.0) < 1e-5, f"Row {i} sums to {row.sum()}"


# =====================================================================
# Transition engine
# =====================================================================

class TransitionEngine:
    """Resolves attacker and defender actions into structured outcomes.

    This class encapsulates the stochastic dynamics of the environment,
    including exploit probabilities, IDS detection, and service state
    changes.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def resolve_attacker(
        self,
        action: AttackerAction,
        attacker_type: AttackerType,
        network_state: Dict[str, Any],
        ids_sensitivity: float = 0.5,
    ) -> StepOutcome:
        """Resolve an attacker action against the current network state.

        Parameters
        ----------
        action
            The attacker's chosen action.
        attacker_type
            The true (hidden) attacker type.
        network_state
            Current network snapshot dict.
        ids_sensitivity
            IDS detection sensitivity (0..1). Higher = more likely to detect.

        Returns
        -------
        StepOutcome
            Partially filled outcome (attacker fields only).
        """
        outcome = StepOutcome()

        # Base detection probability depends on action noisiness
        base_detection = self._detection_probability(action, attacker_type)
        detect_roll = self.rng.random()

        if action == AttackerAction.RECON_SCAN:
            outcome.recon_value = True
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity)

        elif action == AttackerAction.RECON_FINGERPRINT:
            outcome.recon_value = True
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity * 0.7)

        elif action in (AttackerAction.AUTH_BRUTEFORCE, AttackerAction.AUTH_CREDENTIAL_STUFF):
            # Check if any service is available for auth attempts
            vuln = self._pick_service_vulnerability(network_state)
            success_prob = 0.15 if action == AttackerAction.AUTH_BRUTEFORCE else 0.10
            if vuln > 0:
                outcome.attack_success = self.rng.random() < (success_prob * vuln)
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity)

        elif action in (AttackerAction.EXPLOIT_SERVICE, AttackerAction.EXPLOIT_IOT):
            vuln = self._pick_service_vulnerability(network_state)
            outcome.attack_success = self.rng.random() < vuln
            if outcome.attack_success:
                outcome.new_capability_gained = self.rng.random() < 0.3
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity)

        elif action == AttackerAction.MALWARE_DROP:
            has_access = self._has_compromised_service(network_state)
            outcome.attack_success = has_access and self.rng.random() < 0.6
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity * 1.2)

        elif action in (AttackerAction.PERSIST_BACKDOOR, AttackerAction.PERSIST_C2):
            has_access = self._has_compromised_service(network_state)
            outcome.attack_success = has_access and self.rng.random() < 0.5
            outcome.new_capability_gained = outcome.attack_success
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity * 0.8)

        elif action in (AttackerAction.EVADE_OBFUSCATE, AttackerAction.EVADE_SLOWDOWN):
            # Evasion reduces future detection probability (handled by caller)
            outcome.attacker_detected = False  # evasion actions are inherently stealthy

        elif action == AttackerAction.LATERAL_MOVE:
            has_access = self._has_compromised_service(network_state)
            outcome.attack_success = has_access and self.rng.random() < 0.35
            outcome.attacker_detected = detect_roll < (base_detection * ids_sensitivity)

        elif action == AttackerAction.NOOP:
            pass  # no effect

        # Set impact score
        outcome.impact_score = ATTACK_IMPACT_WEIGHTS.get(action, 0.0) if outcome.attack_success else 0.0

        return outcome

    def resolve_defender(
        self,
        action: DefenderAction,
        attacker_outcome: StepOutcome,
        network_state: Dict[str, Any],
    ) -> StepOutcome:
        """Resolve a defender action, updating the outcome in-place.

        Parameters
        ----------
        action
            The defender's chosen action.
        attacker_outcome
            The outcome from attacker resolution (will be modified).
        network_state
            Current network snapshot.

        Returns
        -------
        StepOutcome
            The fully resolved outcome.
        """
        outcome = attacker_outcome  # modify in-place
        real_threat = outcome.attack_success or outcome.attacker_detected or self._has_compromised_service(network_state)

        # Set defence cost
        outcome.defence_action_cost = DEFENDER_ACTION_COSTS.get(action, 0.0)

        if action == DefenderAction.BLOCK_IP:
            if real_threat:
                outcome.correct_detection = True
                outcome.attacker_blocked = True
                outcome.attacker_isolated_by_defender = True
            else:
                outcome.false_positive = True
                outcome.intervention_unnecessary = True

        elif action == DefenderAction.RATE_LIMIT:
            if real_threat:
                outcome.correct_detection = True
            else:
                outcome.false_positive = True

        elif action == DefenderAction.ISOLATE_SERVICE:
            outcome.service_down = True
            outcome.service_maintained = False
            if self._has_compromised_service(network_state):
                outcome.correct_detection = True
                outcome.service_restored = True
                outcome.attacker_isolated_by_defender = True
            elif not real_threat:
                outcome.false_positive = True
                outcome.intervention_unnecessary = True

        elif action == DefenderAction.ISOLATE_IOT:
            outcome.service_down = True
            outcome.service_maintained = False
            if real_threat:
                outcome.correct_detection = True
                outcome.attacker_isolated_by_defender = True
            else:
                outcome.false_positive = True
                outcome.intervention_unnecessary = True

        elif action == DefenderAction.TERMINATE_SESSION:
            if self._has_active_sessions(network_state):
                outcome.correct_detection = True
                outcome.service_restored = True
            elif not real_threat:
                outcome.false_positive = True

        elif action == DefenderAction.INCREASE_MONITORING:
            pass  # effect handled by caller (adjust IDS sensitivity)

        elif action == DefenderAction.ADJUST_IDS_THRESHOLD:
            pass  # effect handled by caller

        elif action == DefenderAction.HONEYPOT_DEPLOY:
            # Honeypot may redirect attacker; no immediate detection
            pass

        elif action == DefenderAction.RESET_CREDENTIALS:
            if real_threat:
                outcome.correct_detection = True
            else:
                outcome.intervention_unnecessary = True

        elif action == DefenderAction.PATCH_SERVICE:
            # Temporarily reduces service availability but fixes vulnerability
            outcome.service_down = True
            outcome.service_maintained = False
            outcome.service_restored = True

        elif action == DefenderAction.RESTORE_SERVICE:
            outcome.service_restored = True

        elif action == DefenderAction.NETWORK_SHUTDOWN:
            outcome.service_down = True
            outcome.service_maintained = False
            if real_threat:
                outcome.correct_detection = True
                outcome.attacker_isolated_by_defender = True
                outcome.attacker_blocked = True
            else:
                outcome.intervention_unnecessary = True

        elif action in (DefenderAction.MONITOR, DefenderAction.NOOP):
            if outcome.attack_success and not outcome.attacker_detected:
                outcome.missed_attack = True

        return outcome

    # ----- helpers -----

    def _detection_probability(self, action: AttackerAction, attacker_type: AttackerType) -> float:
        """Base detection probability depends on how noisy the action is."""
        noisy_actions = {
            AttackerAction.RECON_SCAN: 0.4,
            AttackerAction.AUTH_BRUTEFORCE: 0.6,
            AttackerAction.AUTH_CREDENTIAL_STUFF: 0.3,
            AttackerAction.EXPLOIT_SERVICE: 0.5,
            AttackerAction.EXPLOIT_IOT: 0.45,
            AttackerAction.MALWARE_DROP: 0.7,
            AttackerAction.PERSIST_BACKDOOR: 0.4,
            AttackerAction.PERSIST_C2: 0.35,
            AttackerAction.LATERAL_MOVE: 0.3,
        }
        base = noisy_actions.get(action, 0.1)
        # Stealth attackers are harder to detect
        if attacker_type == AttackerType.STEALTH:
            base *= 0.5
        elif attacker_type == AttackerType.RESOURCE_AWARE:
            base *= 0.7
        return min(base, 1.0)

    def _pick_service_vulnerability(self, network_state: Dict[str, Any]) -> float:
        """Pick a random service vulnerability from the network."""
        vulns = []
        for hdata in network_state.get("hosts", {}).values():
            if hdata.get("isolated", False):
                continue
            for svc in hdata.get("services", {}).values():
                vulns.append(svc.get("vulnerability", 0.0))
        for ddata in network_state.get("iot_devices", {}).values():
            if not ddata.get("isolated", False):
                vulns.append(0.3 if ddata.get("compromised", False) else 0.2)
        return self.rng.choice(vulns) if vulns else 0.0

    def _has_compromised_service(self, network_state: Dict[str, Any]) -> bool:
        """Check if any service is currently compromised."""
        for hdata in network_state.get("hosts", {}).values():
            for svc in hdata.get("services", {}).values():
                if svc.get("compromised", False):
                    return True
        for ddata in network_state.get("iot_devices", {}).values():
            if ddata.get("compromised", False):
                return True
        return False

    def _has_active_sessions(self, network_state: Dict[str, Any]) -> bool:
        """Check if any service has active attacker sessions."""
        for hdata in network_state.get("hosts", {}).values():
            for svc in hdata.get("services", {}).values():
                if svc.get("active_sessions", 0) > 0:
                    return True
        return False
