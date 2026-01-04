"""Main training loop placeholder for MiniGrid agents."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import yaml

# TODO: hook up actual MiniGrid env and agent selection


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """Load YAML hyperparameters."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agents on MiniGrid MultiRoom")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent.parent / "config" / "hyperparams.yaml"),
        help="Path to hyperparameter YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print("Loaded config:")
    print(config)
    print("TODO: initialize env, agent, and training loop")


if __name__ == "__main__":
    main()
