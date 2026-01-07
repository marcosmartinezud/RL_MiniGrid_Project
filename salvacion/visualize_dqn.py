"""Visualize a trained dueling DQN agent in MultiRoom."""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any, Dict

import gymnasium as gym
import minigrid  # registers envs
import numpy as np
import yaml
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from salvacion.dqn_agent import DQNAgent, DQNConfig, preprocess_obs
from salvacion.env_tools import ACTION_MAP


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_name: str, max_steps: int, render_mode: str | None = "human") -> gym.Env:
    env = gym.make(env_name, max_steps=max_steps, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env


def reset_seed(seed: int | None, random_layout: bool, episode: int) -> int | None:
    if random_layout:
        if seed is None:
            return None
        return seed + episode
    return seed


def resolve_model_path(config: Dict[str, Any], override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    output_dir = pathlib.Path(config.get("output_dir", pathlib.Path(__file__).resolve().parent / "dqn_output"))
    return output_dir / "dueling_dqn.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dueling DQN agent")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "dqn_config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to saved model weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps (seconds)")
    parser.add_argument("--random-layout", action="store_true", help="Use random layouts each episode")
    parser.add_argument("--seed", type=int, default=None, help="Seed for resets")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode")
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    env_name = config.get("env", "MiniGrid-MultiRoom-N4-S5-v0")
    training_cfg = config.get("training", {})
    max_steps = int(args.max_steps or training_cfg.get("max_steps", 100))
    seed = args.seed if args.seed is not None else config.get("seed", None)
    random_layout = args.random_layout or bool(config.get("random_layout", True))
    target_size = int(config.get("obs", {}).get("target_size", 14))

    model_path = resolve_model_path(config, args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    agent = DQNAgent(len(ACTION_MAP), DQNConfig(**config.get("dqn", {})))
    agent.load(str(model_path))
    agent.epsilon = 0.0

    env = make_env(env_name, max_steps=max_steps, render_mode="human")
    if seed is not None:
        np.random.seed(seed)

    print(f"Visualizing env={env_name} episodes={args.episodes} random_layout={random_layout}")
    print(f"Loaded model: {model_path}")

    try:
        for ep in range(1, args.episodes + 1):
            reset = reset_seed(seed, random_layout, ep)
            obs, _ = env.reset(seed=reset)
            state = preprocess_obs(obs, target_size)
            done = False
            steps = 0
            total_reward = 0.0

            while not done and steps < max_steps:
                env.render()
                action_idx = agent.select_action(state, training=False)
                action = ACTION_MAP[action_idx]
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += reward
                steps += 1
                state = preprocess_obs(next_obs, target_size)

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
