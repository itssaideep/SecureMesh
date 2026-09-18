# tests/test_agents.py
"""Unit tests for Attacker (Scripted, PPO) and Defender (Static, RL) agents."""

import pytest
import numpy as np

from securemesh_sce.agents.attacker import (
    ScriptedAttacker, PPOAttacker,
)
from securemesh_sce.agents.defender import (
    StaticDefender, RLDefender,
)
from securemesh_sce.game.actions import (
    AttackerType, N_ATTACKER_ACTIONS, N_DEFENDER_ACTIONS,
)


def test_attacker_agents():
    obs_dim = 20
    dummy_obs = np.ones(obs_dim, dtype=np.float32)

    # Scripted Attacker
    scripted = ScriptedAttacker(attacker_type=AttackerType.OPPORTUNISTIC)
    a_scripted = scripted.select_action(dummy_obs)
    assert 0 <= a_scripted < N_ATTACKER_ACTIONS

    # PPO Attacker
    ppo = PPOAttacker(obs_dim=obs_dim, seed=42)
    a_ppo = ppo.select_action(dummy_obs)
    assert 0 <= a_ppo < N_ATTACKER_ACTIONS

    # PPO Attacker update
    next_obs = dummy_obs * 1.1
    ppo.update((dummy_obs, a_ppo, 1.0, next_obs, False))


def test_defender_agents():
    full_obs_dim = 30
    dummy_obs = np.ones(full_obs_dim, dtype=np.float32)

    # Static Defender
    static_def = StaticDefender()
    a_static = static_def.select_action(dummy_obs)
    assert 0 <= a_static < N_DEFENDER_ACTIONS

    # RL Defender (PPO)
    rl_def = RLDefender(obs_dim=full_obs_dim, seed=42)
    a_rl = rl_def.select_action(dummy_obs)
    assert 0 <= a_rl < N_DEFENDER_ACTIONS

    # RL Defender update
    next_obs = dummy_obs * 0.9
    rl_def.update((dummy_obs, a_rl, -0.5, next_obs, False))

