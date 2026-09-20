# securemesh_sce/game/rewards.py
"""Multi-objective reward functions for attacker and defender.

Defender reward:
    R_D = w_s·Security + w_a·Availability + w_r·Recovery
          - w_c·DefenceCost - w_i·UnnecessaryIntervention

The reward penalises the trivial "isolate everything immediately" strategy
by imposing costs for unnecessary disruption and service downtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# =====================================================================
# Reward weight configuration
# =====================================================================

@dataclass
class DefenderRewardWeights:
    """Tunable weights for the multi-objective defender reward."""
    w_security: float = 1.0
    w_availability: float = 1.5
    w_recovery: float = 0.3
    w_defence_cost: float = 0.2
    w_unnecessary_intervention: float = 1.5


@dataclass
class AttackerRewardWeights:
    """Tunable weights for the attacker reward."""
    w_objective: float = 1.0
    w_stealth: float = 0.3
    w_persistence: float = 0.4
    w_step_cost: float = 0.05


# =====================================================================
# Step outcome (passed to reward functions)
# =====================================================================

@dataclass
class StepOutcome:
    """Structured outcome of a single game step.

    Populated by the game engine and consumed by the reward functions.
    """
    # Attacker outcomes
    attack_success: bool = False
    new_capability_gained: bool = False
    recon_value: bool = False
    attacker_detected: bool = False
    attacker_blocked: bool = False
    attacker_isolated: bool = False

    # Defender outcomes
    correct_detection: bool = False
    false_positive: bool = False
    missed_attack: bool = False
    service_down: bool = False
    service_restored: bool = False
    service_maintained: bool = True
    attacker_isolated_by_defender: bool = False

    # Costs
    defence_action_cost: float = 0.0    # action-specific cost
    intervention_unnecessary: bool = False

    # Impact & Availability
    impact_score: float = 0.0           # severity-weighted (0..1)
    service_availability: float = 1.0   # current network availability ratio (0..1)


# =====================================================================
# Attacker reward
# =====================================================================

def attacker_reward(outcome: StepOutcome,
                    weights: AttackerRewardWeights | None = None) -> float:
    """Compute attacker reward for a single step.

    Parameters
    ----------
    outcome
        Structured step outcome.
    weights
        Tunable reward weights. If None, uses defaults.

    Returns
    -------
    float
        Scalar attacker reward.
    """
    w = weights or AttackerRewardWeights()
    reward = -w.w_step_cost  # constant step cost

    # Objective achievement
    if outcome.attack_success:
        reward += w.w_objective * 100.0
    if outcome.new_capability_gained:
        reward += w.w_objective * 20.0
    if outcome.recon_value:
        reward += w.w_objective * 10.0

    # Stealth bonus/penalty
    if outcome.attacker_detected:
        reward -= w.w_stealth * 30.0
    if outcome.attacker_blocked:
        reward -= w.w_stealth * 20.0
    if outcome.attacker_isolated:
        reward -= w.w_stealth * 50.0

    return reward


# =====================================================================
# Defender reward (multi-objective)
# =====================================================================

def defender_reward(outcome: StepOutcome,
                    weights: DefenderRewardWeights | None = None) -> float:
    """Compute multi-objective defender reward.

    R_D = w_s·Security + w_a·Availability + w_r·Recovery
          - w_c·DefenceCost - w_i·UnnecessaryIntervention

    Parameters
    ----------
    outcome
        Structured step outcome.
    weights
        Tunable reward weights. If None, uses defaults.

    Returns
    -------
    float
        Scalar defender reward.
    """
    w = weights or DefenderRewardWeights()
    reward = 0.0

    # --- Security component ---
    security = 0.0
    if outcome.correct_detection:
        security += 50.0
    if outcome.attacker_isolated_by_defender:
        security += 30.0
    if outcome.missed_attack:
        security -= 40.0
    reward += w.w_security * security

    # --- Availability component ---
    availability = 0.0
    if outcome.service_maintained and not outcome.service_down:
        availability += 10.0
    if outcome.service_down:
        availability -= 60.0
    # Continuous availability penalty: penalises remaining in a degraded state every step
    availability += (outcome.service_availability - 1.0) * 50.0
    reward += w.w_availability * availability

    # --- Recovery component ---
    recovery = 0.0
    if outcome.service_restored:
        recovery += 20.0
    reward += w.w_recovery * recovery

    # --- Defence cost ---
    reward -= w.w_defence_cost * outcome.defence_action_cost

    # --- Unnecessary intervention penalty ---
    if outcome.false_positive:
        reward -= w.w_unnecessary_intervention * 20.0
    if outcome.intervention_unnecessary:
        reward -= w.w_unnecessary_intervention * 15.0

    return reward


# =====================================================================
# Defence action costs
# =====================================================================

# Cost of each defender action (higher = more disruptive)
from .actions import DefenderAction

DEFENDER_ACTION_COSTS: Dict[DefenderAction, float] = {
    DefenderAction.MONITOR:              0.0,
    DefenderAction.INCREASE_MONITORING:  1.0,
    DefenderAction.ADJUST_IDS_THRESHOLD: 2.0,
    DefenderAction.RATE_LIMIT:           5.0,
    DefenderAction.TERMINATE_SESSION:    8.0,
    DefenderAction.RESET_CREDENTIALS:    10.0,
    DefenderAction.HONEYPOT_DEPLOY:      3.0,
    DefenderAction.BLOCK_IP:             15.0,
    DefenderAction.ISOLATE_SERVICE:      25.0,
    DefenderAction.ISOLATE_IOT:          20.0,
    DefenderAction.PATCH_SERVICE:        12.0,
    DefenderAction.RESTORE_SERVICE:      8.0,
    DefenderAction.NETWORK_SHUTDOWN:     50.0,
    DefenderAction.NOOP:                 0.0,
}
