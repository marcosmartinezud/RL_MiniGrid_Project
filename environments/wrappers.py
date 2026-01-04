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
