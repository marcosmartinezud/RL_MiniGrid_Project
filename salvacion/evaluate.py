"""Evaluate a saved Q-learning policy on MiniGrid MultiRoom."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, Tuple

import gymnasium as gym
import minigrid  # registers envs
import yaml

from salvacion.env_tools import ACTION_MAP, build_state, scan_doors
from salvacion.expert_planner import compute_path_info
from salvacion.qlearning_agent import QLearningAgent, QLearningConfig


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_agent(config: Dict[str, Any]) -> QLearningAgent:
    q_cfg = QLearningConfig(**config.get("q_learning", {}))
    agent = QLearningAgent(len(ACTION_MAP), q_cfg)
    return agent


def eval_reset_seed(seed: int | None, random_layout: bool, episode: int) -> int | None:
    if random_layout:
        if seed is None:
            return None
        return seed + episode
    return seed


def evaluate_agent(
    agent: QLearningAgent,
    env_name: str,
    episodes: int,
    max_steps: int,
    seed: int | None,
    random_layout: bool,
    state_mode: str,
    distance_bucket: int = 1,
    max_distance: int = 50,
) -> Dict[str, float]:
    env = gym.make(env_name, max_steps=max_steps)
    successes = 0
    total_reward = 0.0
    total_steps = 0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        for ep in range(1, episodes + 1):
            reset_seed = eval_reset_seed(seed, random_layout, ep)
            obs, _ = env.reset(seed=reset_seed)
            door_positions = scan_doors(env) if state_mode == "layout" else []
            path_info = compute_path_info(env) if state_mode == "bfs" else None
            state = build_state(
                obs, env, door_positions, state_mode, path_info, distance_bucket, max_distance
            )
            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < max_steps:
                action_idx = agent.select_action(state, training=False)
                action = ACTION_MAP[action_idx]
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += reward
                steps += 1
                path_info = compute_path_info(env) if state_mode == "bfs" else None
                state = build_state(
                    next_obs, env, door_positions, state_mode, path_info, distance_bucket, max_distance
                )
            total_reward += ep_reward
            total_steps += steps
            if ep_reward > 0:
                successes += 1
    finally:
        agent.epsilon = original_epsilon
        env.close()

    success_rate = successes / max(1, episodes)
    avg_reward = total_reward / max(1, episodes)
    avg_steps = total_steps / max(1, episodes)
    return {
        "success_rate": float(success_rate),
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate salvacion Q-learning policy")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to saved Q-table")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--random-layout", action="store_true", help="Use random layouts each episode")
    parser.add_argument("--seed", type=int, default=None, help="Seed for resets")
    parser.add_argument("--output", type=str, default=None, help="Path to write eval metrics JSON")
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    env_name = config.get("env", "MiniGrid-MultiRoom-N4-S5-v0")
    training_cfg = config.get("training", {})
    max_steps = int(training_cfg.get("max_steps", 400))
    episodes = int(args.episodes or training_cfg.get("eval_episodes", 200))
    seed = args.seed if args.seed is not None else config.get("seed", None)
    random_layout = args.random_layout or bool(config.get("random_layout", False))
    state_mode = str(config.get("state_mode", "local")).lower()
    if state_mode not in {"local", "layout", "bfs"}:
        raise ValueError(f"Unsupported state_mode: {state_mode}")
    state_cfg = config.get("state", {})
    distance_bucket = int(state_cfg.get("distance_bucket", 1))
    max_distance = int(state_cfg.get("max_distance", 50))

    agent = make_agent(config)
    model_path = args.model
    if model_path is None:
        output_dir = pathlib.Path(config.get("output_dir", pathlib.Path(__file__).resolve().parent / "output"))
        model_path = str(output_dir / "q_table.pkl")
    agent.load(model_path)

    metrics = evaluate_agent(
        agent,
        env_name,
        episodes,
        max_steps,
        seed,
        random_layout,
        state_mode,
        distance_bucket,
        max_distance,
    )
    print(
        "Eval results: "
        f"success_rate={metrics['success_rate']:.3f} "
        f"avg_reward={metrics['avg_reward']:.3f} "
        f"avg_steps={metrics['avg_steps']:.1f}"
    )

    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
