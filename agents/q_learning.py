"""Placeholder Q-learning agent implementation."""

from __future__ import annotations

from typing import Any, Dict, Hashable, Tuple

import numpy as np
import pickle

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-learning agent for MiniGrid."""

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
        """Epsilon-greedy action selection."""
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
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        best_next = np.max(self.q_table[next_state])
        td_target = reward + (0.0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save the Q-table and other relevant parameters to a file."""
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")


    def load(self, path: str) -> None:
        """Load the Q-table and parameters from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        print(f"Model loaded from {path}")
