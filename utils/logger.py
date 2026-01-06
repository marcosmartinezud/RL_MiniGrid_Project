"""Simple TensorBoard logger helper without needing PyTorch."""

from __future__ import annotations

import pathlib
from typing import Any

from collections import deque

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class TensorBoardLogger:
    """Lightweight scalar logger using TensorBoard event files."""

    def __init__(self, log_dir: str | pathlib.Path) -> None:
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # using raw EventFileWriter to stay dependency-light
        self.writer = EventFileWriter(self.log_dir.as_posix())
        self.success_window: deque[int] = deque(maxlen=100)

    def _add_scalar(self, tag: str, value: float | int, step: int) -> None:
        summary = summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
        )
        event = event_pb2.Event(step=step, summary=summary)
        self.writer.add_event(event)
        self.writer.flush()

    def log_scalar(self, tag: str, value: float | int, step: int) -> None:
        self._add_scalar(tag, value, step)

    def log_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        epsilon: float,
        success: bool = False,
        loss: float | None = None,
    ) -> None:
        # keeping it simple; we can add more metrics later
        self._add_scalar("episode/reward", reward, episode)
        self._add_scalar("episode/steps", steps, episode)
        self._add_scalar("episode/epsilon", epsilon, episode)
        self._add_scalar("episode/success", int(success), episode)
        if loss is not None:
            self._add_scalar("episode/loss", loss, episode)

    def log_episode_with_window(
        self,
        episode: int,
        reward: float,
        steps: int,
        epsilon: float,
        success: bool = False,
        loss: float | None = None,
    ) -> None:
        """Log episode plus a rolling success rate over the last 100 eps."""
        self.log_episode(episode, reward, steps, epsilon, success, loss)
        self.success_window.append(1 if success else 0)
        success_rate = sum(self.success_window) / len(self.success_window)
        self._add_scalar("episode/success_rate_100", success_rate, episode)

    def close(self) -> None:
        self.writer.close()
