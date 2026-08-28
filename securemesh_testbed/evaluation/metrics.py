# securemesh_testbed/evaluation/metrics.py
"""Metric computations for the SecureMesh testbed.

All functions operate on the structured step logs produced by the
experiment runner and return scalar or per-episode metric values.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any


def attack_success_rate(steps: List[Dict[str, Any]]) -> float:
    """Fraction of attacker actions that achieved their goal."""
    if not steps:
        return 0.0
    successes = sum(1 for s in steps if s.get("attacker_success", False))
    return successes / len(steps)


def detection_rate(steps: List[Dict[str, Any]]) -> float:
    """True-positive rate: fraction of attacks detected by IDS/defender."""
    attacks = [s for s in steps if s.get("attacker_success", False)]
    if not attacks:
        return 1.0  # nothing to detect
    detected = sum(1 for s in attacks if s.get("attacker_detected", False)
                   or s.get("defender_detected", False))
    return detected / len(attacks)


def false_positive_rate(steps: List[Dict[str, Any]]) -> float:
    """Fraction of benign steps incorrectly flagged as attacks."""
    benign = [s for s in steps if not s.get("attacker_success", False)]
    if not benign:
        return 0.0
    fp = sum(1 for s in benign if s.get("defender_detected", False))
    return fp / len(benign)


def detection_latency(steps: List[Dict[str, Any]]) -> float:
    """Average number of steps between an attack and its detection."""
    latencies: List[int] = []
    pending_attack_step: int | None = None
    for i, s in enumerate(steps):
        if s.get("attacker_success", False) and pending_attack_step is None:
            pending_attack_step = i
        if pending_attack_step is not None and (
                s.get("attacker_detected") or s.get("defender_detected")):
            latencies.append(i - pending_attack_step)
            pending_attack_step = None
    return float(np.mean(latencies)) if latencies else 0.0


def service_availability(steps: List[Dict[str, Any]]) -> float:
    """Fraction of steps where all services remained operational."""
    if not steps:
        return 1.0
    available = sum(1 for s in steps if not s.get("service_down", False))
    return available / len(steps)


def attack_impact(steps: List[Dict[str, Any]]) -> float:
    """Severity-weighted compromise score (0..1)."""
    if not steps:
        return 0.0
    total = sum(s.get("impact_score", 0.0) for s in steps)
    return min(1.0, total / max(len(steps), 1))


def recovery_time(steps: List[Dict[str, Any]]) -> float:
    """Average steps from compromise to restored service."""
    times: List[int] = []
    compromised_step: int | None = None
    for i, s in enumerate(steps):
        if s.get("attacker_success") and compromised_step is None:
            compromised_step = i
        if compromised_step is not None and s.get("service_restored", False):
            times.append(i - compromised_step)
            compromised_step = None
    return float(np.mean(times)) if times else 0.0


def cumulative_reward(episode_rewards: List[float]) -> float:
    """Sum of per-step rewards over an episode."""
    return float(np.sum(episode_rewards))


def adaptation_kl(prob_history: List[np.ndarray]) -> float:
    """KL divergence between the first and last policy distributions.

    Measures how much the agent's policy has changed over training.
    """
    if len(prob_history) < 2:
        return 0.0
    p = np.asarray(prob_history[0], dtype=np.float64) + 1e-10
    q = np.asarray(prob_history[-1], dtype=np.float64) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def system_resilience(availability: float, impact: float,
                      recovery: float, max_recovery: float = 100.0) -> float:
    """Composite resilience score:  availability × (1 − impact) × recovery_speed."""
    recovery_speed = 1.0 - min(recovery / max_recovery, 1.0)
    return availability * (1.0 - impact) * recovery_speed


def compute_all(steps: List[Dict[str, Any]],
                atk_rewards: List[float],
                def_rewards: List[float]) -> Dict[str, float]:
    """Convenience function that computes every metric at once."""
    asr = attack_success_rate(steps)
    dr = detection_rate(steps)
    fpr = false_positive_rate(steps)
    dl = detection_latency(steps)
    sa = service_availability(steps)
    ai = attack_impact(steps)
    rt = recovery_time(steps)
    return {
        "attack_success_rate": asr,
        "detection_rate": dr,
        "false_positive_rate": fpr,
        "detection_latency": dl,
        "service_availability": sa,
        "attack_impact": ai,
        "recovery_time": rt,
        "attacker_cumulative_reward": cumulative_reward(atk_rewards),
        "defender_cumulative_reward": cumulative_reward(def_rewards),
        "system_resilience": system_resilience(sa, ai, rt),
    }
