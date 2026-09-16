# securemesh_sce/agents/defender/__init__.py
"""Defender agent hierarchy (Levels 1-5).

Defenders:
- Level 1: StaticDefender (Rule-based heuristics)
- Level 2: RandomForestDefender (Supervised Machine Learning)
- Level 3: RLDefender (PPO without Bayesian belief, x_t only)
- Level 4: BayesianRLDefender (PPO with full posterior [x_t, b_t, h_t, r_t])
- Level 5: ConstrainedDefender (Bayesian RL + Safety Gate & Bounded Autonomy)
"""

from .static import StaticDefender
from .ml import RandomForestDefender
from .rl import RLDefender
from .bayesian_rl import BayesianRLDefender
from .constrained import ConstrainedDefender, SafetyGate

__all__ = [
    "StaticDefender",
    "RandomForestDefender",
    "RLDefender",
    "BayesianRLDefender",
    "ConstrainedDefender",
    "SafetyGate",
]
