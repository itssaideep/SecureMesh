# securemesh_sce/evaluation/metrics.py
"""Comprehensive evaluation metrics for SecureMesh-SCE.

Implements all five research metric groups:
1. Security Metrics (ASR, DR, Precision, Recall, F1, FPR)
2. Response Metrics (MTTD, MTTR, Intervention Cost)
3. Adaptation Metrics (Action diversity, trajectory similarity)
4. Bayesian Uncertainty Metrics (Accuracy, Brier score, Log loss, ECE)
5. Resilience & Autonomy Metrics (Service availability, Cumulative impact, Violations prevented)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from ..game.actions import (
    AttackerAction, DefenderAction, ImpactTier,
    DEFENDER_ACTION_TIERS, ATTACK_IMPACT_WEIGHTS,
)
from ..game.rewards import DEFENDER_ACTION_COSTS, StepOutcome


@dataclass
class EpisodeMetrics:
    # Metadata
    episode_id: int = 0
    total_steps: int = 0
    attacker_type: str = "opportunistic"

    # 1. Security
    attack_success_rate: float = 0.0
    detection_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0

    # 2. Response
    mttd: float = 0.0
    mttr: float = 0.0
    total_intervention_cost: float = 0.0

    # 3. Adaptation
    action_entropy: float = 0.0

    # 4. Bayesian
    bayesian_accuracy: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0

    # 5. Resilience
    mean_service_availability: float = 1.0
    cumulative_impact: float = 0.0
    recovery_debt: float = 0.0

    # 6. Autonomous Gating
    autonomous_action_rate: float = 1.0
    high_impact_rate: float = 0.0
    safety_violations_prevented: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsEngine:
    """Computes all metrics from an episode's step log and outcomes."""

    @staticmethod
    def evaluate_episode(
        episode_id: int,
        step_log: List[Dict[str, Any]],
        outcomes: List[StepOutcome],
        calibration_summary: Optional[Dict[str, float]] = None,
        safety_stats: Optional[Dict[str, Any]] = None,
    ) -> EpisodeMetrics:
        m = EpisodeMetrics(episode_id=episode_id, total_steps=len(step_log))
        if not step_log:
            return m

        m.attacker_type = step_log[0].get("attacker_type", "opportunistic")

        # Counts
        total_attacks = 0
        successful_attacks = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        first_attack_step = None
        first_detect_step = None
        first_mitigate_step = None

        total_cost = 0.0
        availabilities = []
        attacker_actions = []
        defender_tiers = []

        for i, step in enumerate(step_log):
            atk_act_name = step.get("attacker_action", "NOOP")
            def_act_name = step.get("defender_action", "MONITOR")

            try:
                atk_act = AttackerAction[atk_act_name]
            except KeyError:
                atk_act = AttackerAction.NOOP

            try:
                def_act = DefenderAction[def_act_name]
            except KeyError:
                def_act = DefenderAction.MONITOR

            tier = DEFENDER_ACTION_TIERS.get(def_act, ImpactTier.LOW)
            defender_tiers.append(tier)
            total_cost += DEFENDER_ACTION_COSTS.get(def_act, 0.0)

            is_attack = atk_act != AttackerAction.NOOP
            if is_attack:
                total_attacks += 1
                if first_attack_step is None:
                    first_attack_step = i

            success = step.get("attack_success", False)
            if success:
                successful_attacks += 1

            detected = step.get("attacker_detected", False)
            correct = step.get("correct_detection", False)
            is_fp = step.get("false_positive", False)
            is_fn = step.get("missed_attack", False)

            if correct:
                tp += 1
                if first_detect_step is None:
                    first_detect_step = i
            elif is_fp:
                fp += 1
            elif is_fn:
                fn += 1
            else:
                tn += 1

            if tier in (ImpactTier.HIGH, ImpactTier.CRITICAL) and first_mitigate_step is None:
                first_mitigate_step = i

            avail = step.get("service_availability", 1.0)
            availabilities.append(avail)
            attacker_actions.append(atk_act.value)

        # 1. Security
        m.attack_success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        m.detection_rate = tp / total_attacks if total_attacks > 0 else 0.0

        p_denom = tp + fp
        m.precision = tp / p_denom if p_denom > 0 else 1.0

        r_denom = tp + fn
        m.recall = tp / r_denom if r_denom > 0 else 0.0

        if (m.precision + m.recall) > 0:
            m.f1_score = 2 * (m.precision * m.recall) / (m.precision + m.recall)
        else:
            m.f1_score = 0.0

        fpr_denom = fp + tn
        m.false_positive_rate = fp / fpr_denom if fpr_denom > 0 else 0.0

        # 2. Response
        if first_attack_step is not None and first_detect_step is not None:
            m.mttd = float(max(0, first_detect_step - first_attack_step))
        else:
            m.mttd = float(len(step_log))

        if first_detect_step is not None and first_mitigate_step is not None:
            m.mttr = float(max(0, first_mitigate_step - first_detect_step))
        else:
            m.mttr = 0.0

        m.total_intervention_cost = total_cost

        # 3. Adaptation
        if attacker_actions:
            counts = np.bincount(attacker_actions)
            probs = counts[counts > 0] / len(attacker_actions)
            m.action_entropy = float(-np.sum(probs * np.log2(probs)))

        # 4. Bayesian
        if calibration_summary:
            m.bayesian_accuracy = calibration_summary.get("accuracy", 0.0)
            m.brier_score = calibration_summary.get("brier_score", 0.0)
            m.log_loss = calibration_summary.get("log_loss", 0.0)
            m.ece = calibration_summary.get("ece", 0.0)

        # 5. Resilience
        m.mean_service_availability = float(np.mean(availabilities)) if availabilities else 1.0
        if step_log:
            last = step_log[-1]
            m.cumulative_impact = last.get("cumulative_impact", 0.0)

        # 6. Autonomous gating
        if defender_tiers:
            n_auto = sum(1 for t in defender_tiers if t == ImpactTier.LOW)
            n_high = sum(1 for t in defender_tiers if t in (ImpactTier.HIGH, ImpactTier.CRITICAL))
            m.autonomous_action_rate = n_auto / len(defender_tiers)
            m.high_impact_rate = n_high / len(defender_tiers)

        if safety_stats:
            m.safety_violations_prevented = safety_stats.get("safety_violations_prevented", 0)

        return m
