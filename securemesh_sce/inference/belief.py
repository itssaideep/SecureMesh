# securemesh_sce/inference/belief.py
"""Belief state representation and prior profiles.

Provides convenience constructors for different prior configurations
and maintains the full belief object used by the Bayesian RL defender.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from ..game.actions import AttackerType, N_ATTACKER_TYPES
from .bayesian import BayesianInference


# =====================================================================
# Prior profiles
# =====================================================================

PRIOR_UNIFORM = np.ones(N_ATTACKER_TYPES, dtype=np.float64) / N_ATTACKER_TYPES

PRIOR_CONSERVATIVE = np.array([
    0.35,  # opportunistic — most common
    0.20,  # stealth
    0.25,  # adaptive
    0.20,  # resource_aware
], dtype=np.float64)

PRIOR_PARANOID = np.array([
    0.15,  # opportunistic
    0.30,  # stealth — assume worst
    0.35,  # adaptive — assume worst
    0.20,  # resource_aware
], dtype=np.float64)

NAMED_PRIORS: Dict[str, np.ndarray] = {
    "uniform": PRIOR_UNIFORM,
    "conservative": PRIOR_CONSERVATIVE,
    "paranoid": PRIOR_PARANOID,
}


def get_prior(name: str = "uniform") -> np.ndarray:
    """Return a named prior distribution.

    Parameters
    ----------
    name : str
        One of "uniform", "conservative", "paranoid".

    Returns
    -------
    np.ndarray
        Prior probability vector over attacker types.
    """
    if name not in NAMED_PRIORS:
        raise ValueError(f"Unknown prior: {name!r}. Choose from {list(NAMED_PRIORS.keys())}")
    return NAMED_PRIORS[name].copy()


# =====================================================================
# Belief state object
# =====================================================================

class BeliefState:
    """Full belief state maintained by the Bayesian defender.

    Wraps ``BayesianInference`` with additional state for the RL policy.
    """

    def __init__(
        self,
        prior_name: str = "uniform",
        prior: Optional[np.ndarray] = None,
    ):
        p = prior if prior is not None else get_prior(prior_name)
        self.inference = BayesianInference(prior=p)
        self._prior_name = prior_name

    def update(self, observed_action) -> np.ndarray:
        """Update belief given an observed attacker action."""
        return self.inference.update(observed_action)

    @property
    def vector(self) -> np.ndarray:
        """Belief distribution as float32 vector for RL input."""
        return self.inference.belief

    @property
    def entropy(self) -> float:
        return self.inference.entropy

    @property
    def confidence(self) -> float:
        """1 - normalised entropy. Higher = more certain."""
        max_entropy = np.log2(N_ATTACKER_TYPES)
        return 1.0 - (self.entropy / max_entropy) if max_entropy > 0 else 1.0

    @property
    def most_likely(self) -> AttackerType:
        return self.inference.most_likely_type

    def reset(self, prior: Optional[np.ndarray] = None):
        self.inference.reset(prior=prior)

    def trajectory(self) -> np.ndarray:
        """Full belief history, shape (T+1, n_types)."""
        return self.inference.belief_trajectory()
