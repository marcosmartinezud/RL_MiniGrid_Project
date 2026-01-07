"""Visualize a trained Q-learning agent in MiniGrid MultiRoom."""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any, Dict

import gymnasium as gym
import minigrid
import numpy as np
import yaml

from imitation_learning.env_tools import ACTION_MAP, build_state, scan_doors
from imitation_learning.expert_planner import compute_path_info
from imitation_learning.qlearning_agent import QLearningAgent, QLearningConfig


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reset_seed(seed: int | None, random_layout: bool, episode: int) -> int | None:
    if random_layout:
        if seed is None:
            return None
        return seed + episode
    return seed


def resolve_model_path(config: Dict[str, Any], override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    output_dir = pathlib.Path(config.get("output_dir", pathlib.Path(__file__).resolve().parent / "output"))
    return output_dir / "q_table.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Q-learning agent")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to saved Q-table")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps (seconds)")
    parser.add_argument("--random-layout", action="store_true", help="Use random layouts each episode")
    parser.add_argument("--seed", type=int, default=None, help="Seed for resets")
    parser.add_argument("--random-policy", action="store_true", help="Use random actions (debug)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode")
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    env_name = config.get("env", "MiniGrid-MultiRoom-N4-S5-v0")
    training_cfg = config.get("training", {})
    max_steps = int(args.max_steps or training_cfg.get("max_steps", 400))
    seed = args.seed if args.seed is not None else config.get("seed", None)
    random_layout = args.random_layout or bool(config.get("random_layout", False))
    state_mode = str(config.get("state_mode", "local")).lower()
    if state_mode not in {"local", "layout", "bfs"}:
        raise ValueError(f"Unsupported state_mode: {state_mode}")
    state_cfg = config.get("state", {})
    distance_bucket = int(state_cfg.get("distance_bucket", 1))
    max_distance = int(state_cfg.get("max_distance", 50))

    model_path = resolve_model_path(config, args.model)
    agent = None
    if not args.random_policy:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        q_cfg = QLearningConfig(**config.get("q_learning", {}))
        agent = QLearningAgent(len(ACTION_MAP), q_cfg)
        agent.load(str(model_path))
        agent.epsilon = 0.0

    env = gym.make(env_name, render_mode="human", max_steps=max_steps)
    if seed is not None:
        np.random.seed(seed)

    print(f"Visualizing env={env_name} episodes={args.episodes} random_layout={random_layout}")
    if agent is None:
        print("Running random policy.")
    else:
        print(f"Loaded model: {model_path}")

    try:
        for ep in range(1, args.episodes + 1):
            reset = reset_seed(seed, random_layout, ep)
            obs, _ = env.reset(seed=reset)
            door_positions = scan_doors(env) if state_mode == "layout" else []
            path_info = compute_path_info(env) if state_mode == "bfs" else None
            state = build_state(
                obs, env, door_positions, state_mode, path_info, distance_bucket, max_distance
            )
            done = False
            steps = 0
            total_reward = 0.0

            while not done and steps < max_steps:
                env.render()
                if agent is None:
                    action = int(np.random.choice(ACTION_MAP))
                else:
                    action_idx = agent.select_action(state, training=False)
                    action = ACTION_MAP[action_idx]

                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += reward
                steps += 1
                path_info = compute_path_info(env) if state_mode == "bfs" else None
                state = build_state(
                    next_obs, env, door_positions, state_mode, path_info, distance_bucket, max_distance
                )

                if args.delay > 0:
                    time.sleep(args.delay)

            success = total_reward > 0
            print(f"Episode {ep}: reward={total_reward:.2f}, steps={steps}, success={success}")
    except KeyboardInterrupt:
        print("Visualization interrupted. Closing.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
