# securemesh_sce/agents/defender/__init__.py
"""Defender agent hierarchy.

Defenders:
- Level 1: StaticDefender (Rule-based heuristics)
- Level 2: RLDefender (PPO reinforcement learning on system state)
"""

from .static import StaticDefender
from .rl import RLDefender

__all__ = [
    "StaticDefender",
    "RLDefender",
]
