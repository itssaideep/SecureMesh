# securemesh_testbed/game/rewards.py
"""Reward functions for attacker and defender.

Both agents receive a scalar reward at each step. The definitions are
simple linear combinations of key events; they can be tuned later via
configuration.
"""

from enum import Enum

# Simple weight constants – can be parameterised later
ATTACK_SUCCESS_REWARD = 10.0
ATTACK_DETECTED_PENALTY = -8.0
ATTACK_BLOCKED_PENALTY = -5.0
ATTACK_STEP_COST = -0.1

DEFEND_DETECTION_REWARD = 8.0
DEFEND_FALSE_POSITIVE_PENALTY = -2.0
DEFEND_SERVICE_DOWN_PENALTY = -6.0
DEFEND_STEP_COST = -0.05

def attacker_reward(success: bool, detected: bool, blocked: bool) -> float:
    """Return attacker reward for a single step.

    - `success`: attacker achieved a high‑level goal (e.g., persisted backdoor)
    - `detected`: IDS raised an alert for the attacker's action
    - `blocked`: defender blocked the action (e.g., firewall rule)
    """
    reward = ATTACK_STEP_COST
    if success:
        reward += ATTACK_SUCCESS_REWARD
    if detected:
        reward += ATTACK_DETECTED_PENALTY
    if blocked:
        reward += ATTACK_BLOCKED_PENALTY
    return reward

def defender_reward(detected: bool, false_positive: bool, service_down: bool) -> float:
    """Return defender reward for a single step.

    - `detected`: defender correctly detected an attack
    - `false_positive`: IDS signaled an alarm for benign activity
    - `service_down`: a protected service became unavailable
    """
    reward = DEFEND_STEP_COST
    if detected:
        reward += DEFEND_DETECTION_REWARD
    if false_positive:
        reward += DEFEND_FALSE_POSITIVE_PENALTY
    if service_down:
        reward += DEFEND_SERVICE_DOWN_PENALTY
    return reward
