# securemesh_testbed/agents/base_agent.py
"""Abstract base class for agents.

Both attacker and defender agents inherit from this class. The contract is
simple: `select_action(observation)` returns an action enum value, and
`update(transition)` can be used to learn from the (obs, action, reward,
next_obs, done) tuple. Optional `save`/`load` methods allow persisting
learned models.
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, observation: Any) -> int:
        """Return the integer index of the chosen action.
        The environment will map this index back to the appropriate Enum.
        """
        pass

    def update(self, transition: tuple):
        """Optional learning step.
        `transition` is a tuple (obs, action, reward, next_obs, done).
        The default implementation does nothing – useful for random or
        scripted agents.
        """
        pass

    def save(self, path: str):
        """Persist the agent's state. No‑op for stateless agents."""
        pass

    def load(self, path: str):
        """Load a persisted state. No‑op for stateless agents."""
        pass
