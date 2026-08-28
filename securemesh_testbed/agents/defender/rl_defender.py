# securemesh_testbed/agents/defender/rl_defender.py
"""RL-based adaptive defender using the shared NumPy PPO implementation.

Thin wrapper around ``PPOAgent`` that fixes the action dimension to
``len(DefenderAction)`` and provides Bayesian attacker-type belief
tracking as an observation augmentation.

Bayesian Belief Update
----------------------
The defender maintains a probability distribution over attacker types
(opportunistic, sophisticated, stealth) and updates it each step using
Bayes' rule:

    P(theta | O) ∝ P(O | theta) * P(theta)

The belief vector is appended to the raw environment observation before
it is fed to the PPO policy network, giving the agent access to its
current uncertainty about the attacker.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ...game.actions import DefenderAction
from ..common.ppo_agent import PPOAgent


# Attacker-type enumeration
ATTACKER_TYPES = ["opportunistic", "sophisticated", "stealth"]

# Likelihood profiles P(action | attacker_type)
# Rows: action index (0..12), Columns: attacker type
# These are hand-crafted priors; a more rigorous approach would learn
# them from data.
_LIKELIHOOD_TABLE = np.array([
    # RECON_SCAN  RECON_FP  AUTH_BF  AUTH_CS  EXPLOIT_SVC  EXPLOIT_IOT
    #  MAL_DROP  PERSIST_BD  PERSIST_C2  EVADE_OBF  EVADE_SLOW  LAT_MOVE  NOOP
    # --- opportunistic (noisy, brute-force heavy) ---
    [0.15, 0.05, 0.25, 0.10, 0.15, 0.10, 0.05, 0.02, 0.01, 0.02, 0.01, 0.04, 0.05],
    # --- sophisticated (exploit-heavy, persistent) ---
    [0.08, 0.10, 0.05, 0.08, 0.20, 0.12, 0.10, 0.10, 0.08, 0.03, 0.02, 0.02, 0.02],
    # --- stealth (slow, evasive) ---
    [0.05, 0.08, 0.03, 0.05, 0.08, 0.05, 0.05, 0.05, 0.05, 0.15, 0.20, 0.10, 0.06],
], dtype=np.float32)  # shape (3, 13)


class RLDefender(PPOAgent):
    """PPO defender with Bayesian attacker-type belief augmentation.

    Parameters
    ----------
    obs_dim : int
        Length of the *raw* environment observation (before belief
        augmentation).  The PPO network will receive
        ``obs_dim + len(ATTACKER_TYPES)`` inputs.
    """

    def __init__(self, obs_dim: int, seed: int = 0, **kwargs):
        n_types = len(ATTACKER_TYPES)
        # PPO sees raw obs + belief vector
        super().__init__(
            obs_dim=obs_dim + n_types,
            act_dim=len(DefenderAction),
            seed=seed,
            **kwargs,
        )
        self._raw_obs_dim = obs_dim
        # uniform prior
        self.belief: np.ndarray = np.ones(n_types, dtype=np.float32) / n_types

    # ------------------------------------------------------------------
    # Bayesian belief update
    # ------------------------------------------------------------------
    def update_belief(self, attacker_action_idx: int):
        """Update P(theta | history) given the observed attacker action."""
        if attacker_action_idx < 0 or attacker_action_idx >= _LIKELIHOOD_TABLE.shape[1]:
            return
        likelihoods = _LIKELIHOOD_TABLE[:, attacker_action_idx]
        posterior = likelihoods * self.belief
        total = posterior.sum()
        if total > 0:
            self.belief = posterior / total
        # else keep current belief unchanged

    def get_belief(self) -> Dict[str, float]:
        """Return a human-readable belief dictionary."""
        return {t: float(self.belief[i]) for i, t in enumerate(ATTACKER_TYPES)}

    # ------------------------------------------------------------------
    # Override select_action to augment observation with belief
    # ------------------------------------------------------------------
    def _augment(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).flatten()
        return np.concatenate([obs, self.belief])

    def select_action(self, observation) -> int:
        return super().select_action(self._augment(observation))

    def action_probs(self, observation) -> np.ndarray:
        return super().action_probs(self._augment(observation))

    def update(self, transition: tuple):
        obs, act, rew, next_obs, done = transition
        augmented = (self._augment(obs), act, rew, self._augment(next_obs), done)
        super().update(augmented)

    def reset_belief(self):
        """Reset to uniform prior at the start of a new episode."""
        self.belief = np.ones(len(ATTACKER_TYPES), dtype=np.float32) / len(ATTACKER_TYPES)
