# securemesh_testbed/agents/attacker/rl_attacker.py
"""RL-based adaptive attacker using the shared NumPy PPO implementation.

This is a thin wrapper around ``PPOAgent`` that fixes the action
dimension to ``len(AttackerAction)`` and provides a convenient
constructor matching the experiment runner's interface.
"""

from ...game.actions import AttackerAction
from ..common.ppo_agent import PPOAgent


class RLAttacker(PPOAgent):
    """PPO attacker agent.

    Parameters
    ----------
    obs_dim : int
        Length of the flattened observation vector (set dynamically
        after the first environment reset).
    **kwargs
        Forwarded to ``PPOAgent`` (gamma, lam, lr, etc.).
    """

    def __init__(self, obs_dim: int, seed: int = 0, **kwargs):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=len(AttackerAction),
            seed=seed,
            **kwargs,
        )
