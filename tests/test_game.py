# tests/test_game.py
"""Unit tests for Bayesian Markov Game core, actions, and state transitions."""

import pytest
import numpy as np

from securemesh_sce.game.actions import (
    AttackerAction, DefenderAction, AttackerType, ImpactTier,
    DEFENDER_ACTION_TIERS, N_ATTACKER_ACTIONS, N_DEFENDER_ACTIONS,
)
from securemesh_sce.game.state import (
    flatten_network_state, AttackHistory, RiskState, DefenderState,
)
from securemesh_sce.game.rewards import (
    attacker_reward, defender_reward, StepOutcome,
    AttackerRewardWeights, DefenderRewardWeights,
)
from securemesh_sce.game.bayesian_game import BayesianGameEnv, SCEScenario


def test_action_spaces_and_tiers():
    assert N_ATTACKER_ACTIONS == 13
    assert N_DEFENDER_ACTIONS == 14

    for act in DefenderAction:
        assert act in DEFENDER_ACTION_TIERS
        assert isinstance(DEFENDER_ACTION_TIERS[act], ImpactTier)


def test_network_state_flattening():
    snapshot = {
        "hosts": {
            "host_0": {
                "isolated": False,
                "services": {
                    "ssh": {"compromised": False, "active_sessions": 1, "vulnerability": 0.3},
                    "http": {"compromised": True, "active_sessions": 0, "vulnerability": 0.5},
                },
            }
        },
        "iot_devices": {
            "iot_0": {"isolated": False, "compromised": False, "firmware_integrity": 1.0}
        },
    }
    flat = flatten_network_state(snapshot)
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.float32
    assert len(flat) > 0


def test_bayesian_game_step_and_reset():
    scenario = SCEScenario(duration=10, seed=42)
    env = BayesianGameEnv(scenario=scenario)

    (atk_obs, def_obs), info = env.reset(seed=42)
    assert isinstance(atk_obs, np.ndarray)
    assert isinstance(def_obs, np.ndarray)
    assert "attacker_type" in info

    (next_atk_obs, next_def_obs), (r_atk, r_def), term, trunc, step_info = env.step((0, 0))
    assert isinstance(r_atk, float)
    assert isinstance(r_def, float)
    assert "attacker_action" in step_info
    assert "defender_action" in step_info
    assert "service_availability" in step_info


def test_reward_penalizes_blind_shutdown():
    # If defender does full shutdown with no actual attack, penalty should trigger
    weights = DefenderRewardWeights(w_unnecessary_intervention=1.0)
    outcome = StepOutcome(
        attack_success=False,
        attacker_detected=False,
        correct_detection=False,
        false_positive=True,
        service_down=True,
        intervention_unnecessary=True,
    )
    r_def = defender_reward(outcome, weights)
    assert r_def < 0.0, "Unnecessary network shutdown must be penalised"
