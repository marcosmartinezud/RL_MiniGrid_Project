"""Dueling DQN agent inspired by MiniGrid MultiRoom baselines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device(device: str | None) -> torch.device:
    if device and device.lower() != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_obs(obs: np.ndarray, target_size: int) -> torch.Tensor:
    """Extract green channel and downsample to a square tensor."""
    if obs.ndim != 3 or obs.shape[2] < 2:
        raise ValueError(f"Expected RGB observation, got shape {obs.shape}")
    green = obs[:, :, 1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(green).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    if tensor.shape[-1] != target_size or tensor.shape[-2] != target_size:
        tensor = F.interpolate(tensor, size=(target_size, target_size), mode="area")
    return tensor.squeeze(0)  # 1xHxW


class DuelingDQN(nn.Module):
    """CNN-based dueling network for Q-values."""

    def __init__(self, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DQNConfig:
    gamma: float = 0.95
    learning_rate: float = 5e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay: float = 0.99995
    batch_size: int = 64
    buffer_size: int = 10000
    target_update: int = 1000
    train_start: int = 1000
    update_freq: int = 1
    hidden_size: int = 256
    device: str | None = "auto"


class DQNAgent:
    def __init__(self, action_size: int, config: DQNConfig) -> None:
        self.action_size = action_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.target_update = config.target_update
        self.train_start = config.train_start
        self.update_freq = config.update_freq
        self.device = resolve_device(config.device)

        self.policy_net = DuelingDQN(action_size, config.hidden_size).to(self.device)
        self.target_net = DuelingDQN(action_size, config.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.train_steps = 0

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        with torch.no_grad():
            q_values = self.policy_net(state.to(self.device).unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        self.replay_buffer.push(
            state.cpu().numpy(),
            action,
            float(reward),
            next_state.cpu().numpy(),
            bool(done),
        )

    def update(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.train_start):
            return None
        if self.train_steps % self.update_freq != 0:
            self.train_steps += 1
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.from_numpy(states).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
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


@dataclass
class ShapingConfig:
    enabled: bool = True
    explore_bonus: float = 0.05
    door_open_bonus: float = 0.5
    toggle_open_penalty: float = -0.3
    repeat_penalty: float = -0.1
    forward_after_open_bonus: float = 0.2
    done_bonus: float = 1.0
    step_penalty: float = -0.001


class RewardShaper:
    def __init__(self, config: ShapingConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.visited: set[Tuple[int, int]] = set()
        self.opened_doors: set[Tuple[int, int]] = set()
        self.last_actions: Deque[int] = deque(maxlen=3)

    def apply(self, env, action_idx: int, reward: float, done: bool) -> float:
        if not self.config.enabled:
            return reward

        shaped = reward
        agent_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
        if agent_pos not in self.visited:
            self.visited.add(agent_pos)
            shaped += self.config.explore_bonus

        if action_idx == 3:  # toggle
            front_pos = tuple(int(x) for x in env.unwrapped.front_pos)
            cell = env.unwrapped.grid.get(*front_pos)
            if cell is not None and getattr(cell, "type", None) == "door":
                if cell.is_open:
                    if front_pos in self.opened_doors:
                        shaped += self.config.toggle_open_penalty
                    else:
                        self.opened_doors.add(front_pos)
                        shaped += self.config.door_open_bonus

        if action_idx == 2 and self.opened_doors:
            shaped += self.config.forward_after_open_bonus

        if len(self.last_actions) == self.last_actions.maxlen and all(
            a == action_idx for a in self.last_actions
        ):
            shaped += self.config.repeat_penalty

        if self.config.step_penalty != 0:
            shaped += self.config.step_penalty

        if done and reward > 0:
            shaped += self.config.done_bonus

        self.last_actions.append(action_idx)
        return shaped

    def doors_opened(self) -> int:
        return len(self.opened_doors)
