# securemesh_sce/agents/attacker/__init__.py
"""Attacker agent hierarchy.

Attackers:
- Level 0: ScriptedAttacker (Deterministic kill-chain sequences)
- Level 1: PPOAttacker (Adaptive autonomous reinforcement learning)
"""

from .scripted import BaseAgent, ScriptedAttacker
from .ppo import PPOAttacker

__all__ = [
    "BaseAgent",
    "ScriptedAttacker",
    "PPOAttacker",
]
