"""Evaluation script placeholder."""

from __future__ import annotations

import argparse
import pathlib


# TODO: add proper evaluation routine once training is ready

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained agents on MiniGrid")
    parser.add_argument(
        "--model",
        type=str,
        default=str(pathlib.Path("results") / "latest_agent.pkl"),
        help="Path to saved agent file",
    )
    args = parser.parse_args()
    print(f"TODO: load model from {args.model} and run evaluation")


if __name__ == "__main__":
    main()
