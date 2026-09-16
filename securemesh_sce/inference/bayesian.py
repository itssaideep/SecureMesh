# securemesh_sce/inference/bayesian.py
"""Exact Bayesian posterior update for attacker-type inference.

Implements:
    P(θ_i | O_t) = P(O_t | θ_i) · P(θ_i) / Σ_j P(O_t | θ_j) · P(θ_j)

The defender maintains the full posterior distribution over attacker types
at every step. The posterior is never collapsed to argmax — the full
uncertainty vector is part of the defender's state.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from ..game.actions import AttackerAction, AttackerType, N_ATTACKER_TYPES
from ..game.transitions import ATTACKER_LIKELIHOOD_TABLE


class BayesianInference:
    """Exact discrete Bayesian filter over attacker types.

    Attributes
    ----------
    prior : np.ndarray
        Initial prior distribution over attacker types.
    posterior : np.ndarray
        Current posterior P(θ_i | O_{0:t}).
    likelihood_table : np.ndarray
        P(action | θ_i) table, shape (n_types, n_actions).
    history : list
        Full trajectory of posterior distributions for analysis.
    """

    def __init__(
        self,
        prior: Optional[np.ndarray] = None,
        likelihood_table: Optional[np.ndarray] = None,
    ):
        """Initialise the Bayesian filter.

        Parameters
        ----------
        prior
            Prior distribution over θ. If None, uses uniform prior.
        likelihood_table
            P(action | θ_i) table of shape (n_types, n_actions).
            If None, uses the default from transitions.py.
        """
        if prior is not None:
            self.prior = np.array(prior, dtype=np.float64)
            assert len(self.prior) == N_ATTACKER_TYPES
        else:
            self.prior = np.ones(N_ATTACKER_TYPES, dtype=np.float64) / N_ATTACKER_TYPES

        self.likelihood_table = (
            np.array(likelihood_table, dtype=np.float64)
            if likelihood_table is not None
            else ATTACKER_LIKELIHOOD_TABLE.astype(np.float64)
        )

        self.posterior = self.prior.copy()
        self.history: List[np.ndarray] = [self.posterior.copy()]
        self._step_count = 0

    def update(self, observed_action: AttackerAction) -> np.ndarray:
        """Perform one Bayesian update given an observed attacker action.

        P(θ_i | O_t) = P(O_t | θ_i) · b_{t-1}(θ_i)  /  Σ_j P(O_t | θ_j) · b_{t-1}(θ_j)

        Parameters
        ----------
        observed_action
            The attacker action observed this step.

        Returns
        -------
        np.ndarray
            Updated posterior distribution.
        """
        # Action index (1-based enum to 0-based index)
        action_idx = list(AttackerAction).index(observed_action)

        # Likelihood vector P(O_t | θ_i) for all types
        likelihood = self.likelihood_table[:, action_idx]

        # Bayes update
        unnormalised = likelihood * self.posterior
        evidence = unnormalised.sum()

        if evidence > 0:
            self.posterior = unnormalised / evidence
        else:
            # Fallback: uniform (all likelihoods were zero — shouldn't happen)
            self.posterior = np.ones(N_ATTACKER_TYPES, dtype=np.float64) / N_ATTACKER_TYPES

        self.history.append(self.posterior.copy())
        self._step_count += 1
        return self.posterior.copy()

    def update_batch(self, observed_actions: List[AttackerAction]) -> np.ndarray:
        """Apply a sequence of Bayesian updates."""
        for action in observed_actions:
            self.update(action)
        return self.posterior.copy()

    def reset(self, prior: Optional[np.ndarray] = None):
        """Reset to initial prior."""
        if prior is not None:
            self.prior = np.array(prior, dtype=np.float64)
        self.posterior = self.prior.copy()
        self.history = [self.posterior.copy()]
        self._step_count = 0

    # ----- query methods -----

    @property
    def belief(self) -> np.ndarray:
        """Current belief distribution (float32 for RL input)."""
        return self.posterior.astype(np.float32)

    @property
    def entropy(self) -> float:
        """Shannon entropy of the current posterior (bits)."""
        p = self.posterior
        p_safe = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p_safe * np.log2(p_safe)))

    @property
    def max_posterior(self) -> float:
        """Maximum posterior probability (confidence in best guess)."""
        return float(self.posterior.max())

    @property
    def most_likely_type(self) -> AttackerType:
        """The attacker type with highest posterior probability."""
        idx = int(np.argmax(self.posterior))
        return list(AttackerType)[idx]

    def type_probability(self, attacker_type: AttackerType) -> float:
        """Return P(θ = attacker_type | O_{0:t})."""
        idx = list(AttackerType).index(attacker_type)
        return float(self.posterior[idx])

    @property
    def step_count(self) -> int:
        return self._step_count

    def belief_trajectory(self) -> np.ndarray:
        """Return full history of posteriors, shape (T+1, n_types)."""
        return np.array(self.history, dtype=np.float64)
