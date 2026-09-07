# securemesh_testbed/game/rewards.py
"""Reward functions for attacker and defender.

Both agents receive a scalar reward at each step. Two reward profiles
are provided:

  * **Legacy** — the original small-scale rewards (backward compatible)
  * **SCE** — the larger-scale rewards from the SecureMesh-SCE roadmap

Select the profile via the ``reward_profile`` parameter. The default is
``"sce"`` for new experiments; pass ``"legacy"`` to reproduce old results.
"""

from enum import Enum
from typing import Dict


# =========================================================================
# Legacy reward constants (original SecureMesh)
# =========================================================================

ATTACK_SUCCESS_REWARD = 10.0
ATTACK_DETECTED_PENALTY = -8.0
ATTACK_BLOCKED_PENALTY = -5.0
ATTACK_STEP_COST = -0.1

DEFEND_DETECTION_REWARD = 8.0
DEFEND_FALSE_POSITIVE_PENALTY = -2.0
DEFEND_SERVICE_DOWN_PENALTY = -6.0
DEFEND_STEP_COST = -0.05


# =========================================================================
# SCE reward constants (SecureMesh-SCE roadmap)
# =========================================================================

SCE_ATTACK_REWARDS: Dict[str, float] = {
    "objective_achieved":       100.0,
    "new_capability":            20.0,
    "useful_recon":              10.0,
    "step_cost":                 -0.5,
    "detected":                 -30.0,
    "isolated":                 -50.0,
    "terminated":              -100.0,
}

SCE_DEFEND_REWARDS: Dict[str, float] = {
    "correct_detection":         50.0,
    "successful_isolation":      30.0,
    "service_maintained":        10.0,
    "step_cost":                 -0.2,
    "false_positive":           -20.0,
    "missed_attack":            -40.0,
    "service_downtime":         -60.0,
}


# =========================================================================
# Reward functions
# =========================================================================

def attacker_reward(success: bool, detected: bool, blocked: bool,
                    *, profile: str = "sce",
                    new_capability: bool = False,
                    isolated: bool = False,
                    terminated: bool = False,
                    recon_value: bool = False) -> float:
    """Return attacker reward for a single step.

    Parameters
    ----------
    success : bool
        Attacker achieved a high-level goal (e.g., persisted backdoor).
    detected : bool
        IDS raised an alert for the attacker's action.
    blocked : bool
        Defender blocked the action (e.g., firewall rule).
    profile : str
        ``"legacy"`` for original rewards, ``"sce"`` for SCE scale.
    new_capability : bool
        Attacker discovered a new capability (exploit, credential).
    isolated : bool
        Attacker was isolated from the network.
    terminated : bool
        Experiment was terminated due to attacker detection.
    recon_value : bool
        Reconnaissance produced useful intelligence.
    """
    if profile == "legacy":
        reward = ATTACK_STEP_COST
        if success:
            reward += ATTACK_SUCCESS_REWARD
        if detected:
            reward += ATTACK_DETECTED_PENALTY
        if blocked:
            reward += ATTACK_BLOCKED_PENALTY
        return reward

    # SCE profile
    r = SCE_ATTACK_REWARDS
    reward = r["step_cost"]
    if success:
        reward += r["objective_achieved"]
    if new_capability:
        reward += r["new_capability"]
    if recon_value:
        reward += r["useful_recon"]
    if detected:
        reward += r["detected"]
    if isolated or blocked:
        reward += r["isolated"]
    if terminated:
        reward += r["terminated"]
    return reward


def defender_reward(detected: bool, false_positive: bool,
                    service_down: bool,
                    *, profile: str = "sce",
                    isolated_attacker: bool = False,
                    missed_attack: bool = False,
                    service_maintained: bool = True) -> float:
    """Return defender reward for a single step.

    Parameters
    ----------
    detected : bool
        Defender correctly detected an attack.
    false_positive : bool
        IDS signaled an alarm for benign activity.
    service_down : bool
        A protected service became unavailable.
    profile : str
        ``"legacy"`` for original rewards, ``"sce"`` for SCE scale.
    isolated_attacker : bool
        Defender successfully isolated the attacker.
    missed_attack : bool
        An attack succeeded without detection.
    service_maintained : bool
        Protected services remained operational this step.
    """
    if profile == "legacy":
        reward = DEFEND_STEP_COST
        if detected:
            reward += DEFEND_DETECTION_REWARD
        if false_positive:
            reward += DEFEND_FALSE_POSITIVE_PENALTY
        if service_down:
            reward += DEFEND_SERVICE_DOWN_PENALTY
        return reward

    # SCE profile
    r = SCE_DEFEND_REWARDS
    reward = r["step_cost"]
    if detected:
        reward += r["correct_detection"]
    if isolated_attacker:
        reward += r["successful_isolation"]
    if service_maintained and not service_down:
        reward += r["service_maintained"]
    if false_positive:
        reward += r["false_positive"]
    if missed_attack:
        reward += r["missed_attack"]
    if service_down:
        reward += r["service_downtime"]
    return reward
