"""Base abstract agent class for tabular RL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable


class BaseAgent(ABC):
    """Interface that all agents should follow."""

    def __init__(self, state_size: int, action_size: int, config: dict[str, Any]) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        # this might need tweaking if we add more state features

    @abstractmethod
    def select_action(self, state: Hashable, training: bool = True) -> int:
        """Pick an action given the current state."""

    @abstractmethod
    def update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
    ) -> None:
        """Update the internal value estimates."""

    def save(self, path: str) -> None:
        """Persist agent parameters to disk."""
        # TODO: implement persistence once tables/params are finalized
        raise NotImplementedError("save() not implemented yet")

    def load(self, path: str) -> None:
        """Load agent parameters from disk."""
        # based on the paper we read in class, we will plug this later
        raise NotImplementedError("load() not implemented yet")
