"""Wrappers to tweak MiniGrid observations for tabular methods."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class FlatObsWrapper(gym.ObservationWrapper):
    """Flatten the MiniGrid image observation to a 1D tuple for hashing."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # MiniGrid returns dict obs; we flatten the image part only
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(np.prod(env.observation_space["image"].shape),),
            dtype=np.uint8,
        )

    def observation(self, observation: Dict[str, Any]) -> Tuple[int, ...]:
        image = observation.get("image")
        if image is None:
            raise ValueError("Observation missing 'image' key; check env wrappers")
        flat = np.array(image, dtype=np.uint8).flatten()
        return tuple(int(x) for x in flat)


class FlatFloatObsWrapper(gym.ObservationWrapper):
    """Flatten image obs to float32 vector in [0,1] for function approximation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        shape = (np.prod(env.observation_space["image"].shape),)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

    def observation(self, observation: Dict[str, Any]) -> np.ndarray:
        image = observation.get("image")
        if image is None:
            raise ValueError("Observation missing 'image' key; check env wrappers")
        flat = np.array(image, dtype=np.float32).flatten() / 255.0
        return flat


class SimpleObsWrapper(gym.ObservationWrapper):
    """Simplify MiniGrid observations for tabular agents.

    Keeps only direction and what is in front, plus door/goal visibility flags.
    This reduces the state space massively compared to raw pixels.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Direction (0-3), front_type, front_state, goal_visible, door_visible
        self.observation_space = gym.spaces.MultiDiscrete([4, 11, 3, 2, 2])

    def observation(self, observation: Dict[str, Any]) -> Tuple[int, int, int, bool, bool]:
        image = observation.get("image")
        direction = int(observation.get("direction", 0))
        if image is None:
            raise ValueError("Observation missing 'image' key; check env wrappers")

        # Agent looks toward increasing x; in the cropped view front is at row 3, col 6
        front = image[3, 6]
        front_type = int(front[0])
        front_state = int(front[2])  # door state encoded here

        obj_layer = image[:, :, 0]
        goal_visible = bool((obj_layer == 8).any())
        door_visible = bool((obj_layer == 4).any())

        return (direction, front_type, front_state, goal_visible, door_visible)


class RewardShapingWrapper(gym.Wrapper):
    """Add small shaping rewards to ease sparse reward learning.

    Based on the idea from Ng et al. (1999) about potential-based shaping,
    we provide mild bonuses for opening doors and exploring new areas.
    """

    def __init__(self, env: gym.Env, config: dict[str, Any]) -> None:
        super().__init__(env)
        self.door_bonus = config.get("door_open_bonus", 0.1)
        self.room_bonus = config.get("new_room_bonus", 0.2)
        self.step_penalty = config.get("step_penalty", -0.001)
        self.visited_positions: set[Tuple[int, int]] = set()
        self.doors_opened: set[Tuple[int, int]] = set()

    def reset(self, **kwargs):
        self.visited_positions = set()
        self.doors_opened = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward + self.step_penalty

        # Track agent position for simple exploration bonus
        agent_pos = tuple(self.unwrapped.agent_pos)
        if agent_pos not in self.visited_positions:
            self.visited_positions.add(agent_pos)
            shaped_reward += 0.01  # tiny exploration boost

        # Bonus when toggling an unopened door open (action 5 = toggle)
        if action == 5:
            fwd_pos = self.unwrapped.front_pos
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.type == "door" and fwd_cell.is_open:
                door_id = tuple(fwd_pos)
                if door_id not in self.doors_opened:
                    self.doors_opened.add(door_id)
                    shaped_reward += self.door_bonus

        # Rough exploration bonus for moving into new cells; could approximate rooms
        # (We keep it light to avoid overpowering the true goal reward.)
        # Note: room_bonus retained for future tweaks, not directly used here.

        return obs, shaped_reward, terminated, truncated, info


class PositionAwareWrapper(gym.ObservationWrapper):
    """Add agent position and richer local cues for navigation.

    State: (x, y, direction, front_type, front_state, left_type,
    right_type, goal_visible, door_visible).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # x,y up to 24 (grid cap), dir 0-3, object types 0-10, states 0-2, bools 0-1
        self.observation_space = gym.spaces.MultiDiscrete([25, 25, 4, 11, 3, 11, 11, 2, 2])

    def observation(self, observation: Dict[str, Any]):
        image = observation.get("image")
        if image is None:
            raise ValueError("Observation missing 'image' key; check env wrappers")

        agent_pos = self.unwrapped.agent_pos
        agent_dir = self.unwrapped.agent_dir
        x, y = int(agent_pos[0]), int(agent_pos[1])
        direction = int(agent_dir)

        front = image[3, 6]
        front_type = int(front[0])
        front_state = int(front[2])

        left = image[0, 6]
        right = image[6, 6]
        left_type = int(left[0])
        right_type = int(right[0])

        obj_layer = image[:, :, 0]
        goal_visible = bool((obj_layer == 8).any())
        door_visible = bool((obj_layer == 4).any())

        return (x, y, direction, front_type, front_state, left_type, right_type, goal_visible, door_visible)
