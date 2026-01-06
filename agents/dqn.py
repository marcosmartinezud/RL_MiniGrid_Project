"""Deep Q-Network agent for MiniGrid.

Based on Mnih et al. (2015) DQN paper.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple MLP for approximating Q-values."""

    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = state_size
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class ReplayBuffer:
    capacity: int

    def __post_init__(self) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def push(self, *args: Any) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states = np.stack([b.state for b in batch])
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.stack([b.next_state for b in batch])
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


class DQNAgent:
    """DQN agent with target network and replay buffer."""

    def __init__(self, state_size: int, action_size: int, config: dict[str, Any]) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.lr = config.get("learning_rate", 5e-4)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 500)
        hidden_sizes = config.get("hidden_sizes", [256, 256])
        buffer_size = config.get("buffer_size", 50000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN using device: {self.device}")

        self.policy_net = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(batch.state).float().to(self.device)
        actions = torch.from_numpy(batch.action).long().to(self.device)
        rewards = torch.from_numpy(batch.reward).float().to(self.device)
        next_states = torch.from_numpy(batch.next_state).float().to(self.device)
        dones = torch.from_numpy(batch.done).float().to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
