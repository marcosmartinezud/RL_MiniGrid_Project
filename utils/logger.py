"""Simple TensorBoard logger helper."""

from __future__ import annotations

import pathlib
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "SummaryWriter not found. Install PyTorch to use TensorBoard logging."
    ) from exc


class TensorBoardLogger:
    """Lightweight wrapper around SummaryWriter for quick logging."""

    def __init__(self, log_dir: str | pathlib.Path) -> None:
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir.as_posix())

    def log_scalar(self, tag: str, value: float | int, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode: int, reward: float, steps: int, epsilon: float) -> None:
        # keeping it simple; we can add more metrics later
        self.writer.add_scalar("episode/reward", reward, episode)
        self.writer.add_scalar("episode/steps", steps, episode)
        self.writer.add_scalar("episode/epsilon", epsilon, episode)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
