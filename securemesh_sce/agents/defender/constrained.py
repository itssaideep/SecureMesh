# securemesh_sce/agents/defender/constrained.py
"""Defender 5: Constrained Bayesian RL defender with Safety Gate & Bounded Autonomy.

Wraps an underlying Bayesian RL policy with an action safety gate enforcing:
1. Impact Tier bounds (LOW, MEDIUM, HIGH, CRITICAL)
2. Staged workflow gating (NORMAL -> SUSPICIOUS -> INVESTIGATE -> CONFIRMED -> MITIGATE -> RECOVER)
3. Belief uncertainty check (blocks irreversible or high-impact actions under high entropy)
4. Telemetry of gated interventions and safety violations prevented.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List

from ..attacker.scripted import BaseAgent
from .bayesian_rl import BayesianRLDefender
from ...game.actions import (
    DefenderAction, ImpactTier, SecurityStage,
    DEFENDER_ACTION_TIERS, DEFENDER_ACTION_REVERSIBLE,
    N_DEFENDER_ACTIONS, N_ATTACKER_TYPES,
)
from ...game.state import AttackHistory


class SafetyGate:
    """Action safety filter enforcing staged escalation and impact limits."""

    def __init__(
        self,
        max_entropy_for_critical: float = 1.0,
        require_confirmation_for_critical: bool = True,
        require_investigation_for_high: bool = True,
    ):
        self.max_entropy_for_critical = max_entropy_for_critical
        self.require_confirmation_for_critical = require_confirmation_for_critical
        self.require_investigation_for_high = require_investigation_for_high

        # Telemetry metrics
        self.total_decisions: int = 0
        self.gated_actions_count: int = 0
        self.safety_violations_prevented: int = 0
        self.action_history: List[Dict[str, Any]] = []

    def evaluate_and_gate(
        self,
        proposed_action_idx: int,
        observation: np.ndarray,
        stage: SecurityStage = SecurityStage.NORMAL,
        belief_entropy: Optional[float] = None,
    ) -> int:
        """Filter the proposed action through safety constraints.

        Returns approved action index (either original or downgraded fallback).
        """
        self.total_decisions += 1
        action_list = list(DefenderAction)
        proposed_action = action_list[proposed_action_idx % N_DEFENDER_ACTIONS]
        tier = DEFENDER_ACTION_TIERS.get(proposed_action, ImpactTier.LOW)

        approved = True
        violation_reason = None
        fallback_action = proposed_action

        # Rule 1: CRITICAL impact actions require CONFIRMED stage and low belief entropy
        if tier == ImpactTier.CRITICAL:
            if self.require_confirmation_for_critical and stage.value < SecurityStage.CONFIRMED.value:
                approved = False
                violation_reason = f"CRITICAL action {proposed_action.name} blocked in stage {stage.name}"
                fallback_action = DefenderAction.INCREASE_MONITORING
            elif belief_entropy is not None and belief_entropy > self.max_entropy_for_critical:
                approved = False
                violation_reason = f"CRITICAL action {proposed_action.name} blocked due to high belief entropy ({belief_entropy:.2f})"
                fallback_action = DefenderAction.RATE_LIMIT

        # Rule 2: HIGH impact actions require at least SUSPICIOUS / INVESTIGATE stage
        elif tier == ImpactTier.HIGH:
            if self.require_investigation_for_high and stage.value < SecurityStage.SUSPICIOUS.value:
                approved = False
                violation_reason = f"HIGH action {proposed_action.name} blocked in stage {stage.name}"
                fallback_action = DefenderAction.MONITOR

        # Rule 3: Irreversible actions under NORMAL stage are prevented
        if approved and not DEFENDER_ACTION_REVERSIBLE.get(proposed_action, True):
            if stage == SecurityStage.NORMAL:
                approved = False
                violation_reason = f"Irreversible action {proposed_action.name} prohibited in NORMAL stage"
                fallback_action = DefenderAction.MONITOR

        if not approved:
            self.gated_actions_count += 1
            self.safety_violations_prevented += 1
            final_action = fallback_action
        else:
            final_action = proposed_action

        final_idx = action_list.index(final_action)
        self.action_history.append({
            "proposed": proposed_action.name,
            "final": final_action.name,
            "approved": approved,
            "reason": violation_reason,
            "tier": tier.value,
            "stage": stage.name,
        })
        return final_idx

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "gated_actions_count": self.gated_actions_count,
            "safety_violations_prevented": self.safety_violations_prevented,
            "gated_ratio": (
                self.gated_actions_count / self.total_decisions
                if self.total_decisions > 0 else 0.0
            ),
        }

    def reset(self):
        self.total_decisions = 0
        self.gated_actions_count = 0
        self.safety_violations_prevented = 0
        self.action_history.clear()


class ConstrainedDefender(BaseAgent):
    """Defender 5: Bayesian RL agent with Bounded Autonomy & Safety Gate.

    Combines the adaptability of Bayesian reinforcement learning with
    deterministic safety contracts, preventing premature or destructive
    autonomous responses.
    """

    def __init__(
        self,
        obs_dim: int,
        base_policy: Optional[BaseAgent] = None,
        max_entropy_for_critical: float = 1.0,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.policy = base_policy or BayesianRLDefender(obs_dim=obs_dim, seed=seed)
        self.safety_gate = SafetyGate(max_entropy_for_critical=max_entropy_for_critical)

    def _parse_stage_and_entropy(self, observation: np.ndarray) -> tuple[SecurityStage, Optional[float]]:
        """Extract security stage and belief entropy from the composite observation vector."""
        obs = np.asarray(observation, dtype=np.float32).ravel()
        # In DefenderState: flat is [system_obs, belief, history, risk]
        # risk vector is [cum_impact, availability, debt, security_stage]
        stage = SecurityStage.NORMAL
        entropy = None

        if len(obs) >= 4:
            stage_val = int(round(obs[-1]))
            stage_list = list(SecurityStage)
            if 0 <= stage_val < len(stage_list):
                stage = stage_list[stage_val]

        # Extract belief vector if dimension matches
        extra_dim = N_ATTACKER_TYPES + len(AttackHistory.FEATURE_NAMES) + 4
        if len(obs) >= extra_dim:
            belief_start = len(obs) - extra_dim
            belief = obs[belief_start:belief_start + N_ATTACKER_TYPES]
            belief = np.clip(belief, 1e-12, 1.0)
            belief = belief / belief.sum()
            entropy = float(-np.sum(belief * np.log2(belief)))

        return stage, entropy

    def select_action(self, observation: np.ndarray) -> int:
        raw_action = self.policy.select_action(observation)
        stage, entropy = self._parse_stage_and_entropy(observation)
        gated_action = self.safety_gate.evaluate_and_gate(
            proposed_action_idx=raw_action,
            observation=observation,
            stage=stage,
            belief_entropy=entropy,
        )
        return gated_action

    def action_probs(self, observation: np.ndarray) -> np.ndarray:
        return self.policy.action_probs(observation)

    def update(self, transition: tuple):
        self.policy.update(transition)

    def save(self, path: str):
        self.policy.save(path)

    def load(self, path: str):
        self.policy.load(path)

    def get_safety_stats(self) -> Dict[str, Any]:
        return self.safety_gate.get_stats()

    def reset_safety_gate(self):
        self.safety_gate.reset()
