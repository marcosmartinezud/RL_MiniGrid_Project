"""Placeholder SARSA agent implementation."""

from __future__ import annotations

from typing import Any, Dict, Hashable

import numpy as np

from .base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    """On-policy SARSA agent for MiniGrid."""

    def __init__(self, state_size: int, action_size: int, config: dict[str, Any]) -> None:
        super().__init__(state_size, action_size, config)
        self.q_table: Dict[Hashable, np.ndarray] = {}
        self.epsilon = config.get("epsilon_start", 1.0)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon_min = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

    def _ensure_state(self, state: Hashable) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)

    def select_action(self, state: Hashable, training: bool = True) -> int:
        self._ensure_state(state)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        """SARSA update with epsilon decay on episode end."""
        self._ensure_state(state)
        self._ensure_state(next_state)
        if next_action is None:
            next_action = self.select_action(next_state, training=True)
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_state][next_action]
        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:  # type: ignore[override]
        # TODO: store q_table with numpy.save or pickle
        raise NotImplementedError("save() not implemented yet")

    def load(self, path: str) -> None:  # type: ignore[override]
        # TODO: load q_table from disk
        raise NotImplementedError("load() not implemented yet")
