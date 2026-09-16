# tests/test_agents.py
"""Unit tests for Attacker (L0-L4) and Defender (D1-D5) agent hierarchies."""

import pytest
import numpy as np

from securemesh_sce.agents.attacker import (
    ScriptedAttacker, BCAttacker, GAILAttacker, PPOAttacker, LLMAttacker,
)
from securemesh_sce.agents.defender import (
    StaticDefender, RandomForestDefender, RLDefender,
    BayesianRLDefender, ConstrainedDefender, SafetyGate,
)
from securemesh_sce.game.actions import (
    AttackerType, N_ATTACKER_ACTIONS, N_DEFENDER_ACTIONS,
    DefenderAction, SecurityStage,
)


def test_attacker_hierarchy():
    obs_dim = 20
    dummy_obs = np.ones(obs_dim, dtype=np.float32)

    # L0 Scripted
    l0 = ScriptedAttacker(attacker_type=AttackerType.OPPORTUNISTIC)
    a0 = l0.select_action(dummy_obs)
    assert 0 <= a0 < N_ATTACKER_ACTIONS

    # L1 BC
    l1 = BCAttacker(obs_dim=obs_dim, seed=42)
    a1 = l1.select_action(dummy_obs)
    assert 0 <= a1 < N_ATTACKER_ACTIONS

    # L2 GAIL
    l2 = GAILAttacker(obs_dim=obs_dim, seed=42)
    a2 = l2.select_action(dummy_obs)
    assert 0 <= a2 < N_ATTACKER_ACTIONS

    # L3 PPO
    l3 = PPOAttacker(obs_dim=obs_dim, seed=42)
    a3 = l3.select_action(dummy_obs)
    assert 0 <= a3 < N_ATTACKER_ACTIONS

    # L4 LLM
    l4 = LLMAttacker(seed=42)
    a4 = l4.select_action(dummy_obs)
    assert 0 <= a4 < N_ATTACKER_ACTIONS


def test_defender_hierarchy():
    full_obs_dim = 30
    dummy_obs = np.ones(full_obs_dim, dtype=np.float32)

    # D1 Static
    d1 = StaticDefender()
    ad1 = d1.select_action(dummy_obs)
    assert 0 <= ad1 < N_DEFENDER_ACTIONS

    # D2 Random Forest
    d2 = RandomForestDefender(seed=42)
    # Train with synthetic observations
    X_train = np.random.randn(20, full_obs_dim).astype(np.float32)
    y_train = np.random.randint(0, N_DEFENDER_ACTIONS, size=20)
    d2.train(X_train, y_train)
    ad2 = d2.select_action(dummy_obs)
    assert 0 <= ad2 < N_DEFENDER_ACTIONS

    # D3 RL (strips augmented belief)
    d3 = RLDefender(obs_dim=full_obs_dim, seed=42)
    ad3 = d3.select_action(dummy_obs)
    assert 0 <= ad3 < N_DEFENDER_ACTIONS

    # D4 Bayesian RL
    d4 = BayesianRLDefender(obs_dim=full_obs_dim, seed=42)
    ad4 = d4.select_action(dummy_obs)
    assert 0 <= ad4 < N_DEFENDER_ACTIONS

    # D5 Constrained
    d5 = ConstrainedDefender(obs_dim=full_obs_dim, seed=42)
    ad5 = d5.select_action(dummy_obs)
    assert 0 <= ad5 < N_DEFENDER_ACTIONS


def test_safety_gate_blocking():
    gate = SafetyGate()
    action_list = list(DefenderAction)

    shutdown_idx = action_list.index(DefenderAction.NETWORK_SHUTDOWN)
    dummy_obs = np.zeros(10, dtype=np.float32)

    # In NORMAL stage, critical action NETWORK_SHUTDOWN should be blocked
    gated_idx = gate.evaluate_and_gate(
        proposed_action_idx=shutdown_idx,
        observation=dummy_obs,
        stage=SecurityStage.NORMAL,
    )
    assert gated_idx != shutdown_idx
    assert gate.gated_actions_count == 1
    assert gate.safety_violations_prevented == 1
