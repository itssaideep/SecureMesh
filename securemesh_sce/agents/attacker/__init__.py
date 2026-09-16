# securemesh_sce/agents/attacker/__init__.py
"""Attacker agent hierarchy (Levels 0-4).

Attackers:
- Level 0: ScriptedAttacker (Deterministic kill-chain sequences)
- Level 1: BCAttacker (Behavioral Cloning imitation policy)
- Level 2: GAILAttacker (Generative Adversarial Imitation Learning)
- Level 3: PPOAttacker (Adaptive autonomous reinforcement learning)
- Level 4: LLMAttacker (LLM-assisted offensive reasoning)
"""

from .scripted import BaseAgent, ScriptedAttacker
from .bc import BCAttacker
from .gail import GAILAttacker
from .ppo import PPOAttacker
from .llm_policy import LLMAttacker

__all__ = [
    "BaseAgent",
    "ScriptedAttacker",
    "BCAttacker",
    "GAILAttacker",
    "PPOAttacker",
    "LLMAttacker",
]
