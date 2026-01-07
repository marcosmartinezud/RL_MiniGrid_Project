"""Tabular Q-learning agent with simple persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable

import numpy as np
import pickle


@dataclass
class QLearningConfig:
    alpha: float = 0.3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    optimistic_init: float = 0.0


class QLearningAgent:
    """Minimal Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, action_size: int, config: QLearningConfig) -> None:
        self.action_size = action_size
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.optimistic_init = config.optimistic_init
        self.q_table: Dict[Hashable, np.ndarray] = {}

    def _ensure_state(self, state: Hashable) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.full(self.action_size, self.optimistic_init, dtype=np.float32)

    def select_action(self, state: Hashable, training: bool = True) -> int:
        self._ensure_state(state)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: Hashable, action: int, reward: float, next_state: Hashable, done: bool) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        best_next = float(np.max(self.q_table[next_state]))
        td_target = reward + (0.0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "action_size": self.action_size,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "optimistic_init": self.optimistic_init,
            "q_table": self.q_table,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.action_size = int(payload["action_size"])
        self.epsilon = float(payload["epsilon"])
        self.alpha = float(payload["alpha"])
        self.gamma = float(payload["gamma"])
        self.epsilon_min = float(payload["epsilon_min"])
        self.epsilon_decay = float(payload["epsilon_decay"])
        self.optimistic_init = float(payload.get("optimistic_init", 0.0))
        self.q_table = payload["q_table"]
