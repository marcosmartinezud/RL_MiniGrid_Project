"""Training loop for tabular Q-learning and SARSA on MiniGrid MultiRoom."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
from typing import Any, Hashable

import gymnasium as gym
import minigrid  # registers MiniGrid envs with gymnasium
import yaml

from agents import QLearningAgent, SarsaAgent
from environments import FlatObsWrapper
from utils import TensorBoardLogger


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """Load YAML hyperparameters."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_name: str) -> gym.Env:
    """Create a MiniGrid env wrapped for tabular observations."""
    base_env = gym.make(env_name)
    return FlatObsWrapper(base_env)


def to_state(obs: Any) -> Hashable:
    """Ensure observations are hashable tuples for Q-table keys."""
    if isinstance(obs, tuple):
        return obs
    try:
        return tuple(obs)
    except TypeError:
        return tuple()


def build_agent(name: str, state_size: int, action_size: int, config: dict[str, Any]) -> Any:
    """Factory for Q-learning or SARSA agent."""
    if name == "qlearning":
        return QLearningAgent(state_size, action_size, config["q_learning"])
    if name == "sarsa":
        return SarsaAgent(state_size, action_size, config["sarsa"])
    raise ValueError(f"Unsupported agent: {name}")


def default_log_dir(agent: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/{agent}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agents on MiniGrid MultiRoom")
    parser.add_argument("--agent", type=str, choices=["qlearning", "sarsa"], default="qlearning")
    parser.add_argument("--env", type=str, default="MiniGrid-MultiRoom-N2-S4-v0")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent.parent / "config" / "hyperparams.yaml"),
        help="Path to hyperparameter YAML file",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom TensorBoard log directory")
    args = parser.parse_args()

    config = load_config(args.config)
    episodes = args.episodes or config["training"].get("num_episodes", 5000)
    max_steps = config["training"].get("max_steps", 100)

    env = make_env(args.env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = build_agent(args.agent, state_size, action_size, config)
    log_dir = args.log_dir or default_log_dir(args.agent)
    logger = TensorBoardLogger(log_dir)

    print(f"Starting training with {args.agent} on {args.env}")
    print(f"Episodes: {episodes}, max_steps per episode: {max_steps}")
    print(f"Logging to: {log_dir}")

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        state = to_state(obs)
        total_reward = 0.0
        steps = 0

        if args.agent == "sarsa":
            action = agent.select_action(state, training=True)

        for _ in range(max_steps):
            if args.agent == "qlearning":
                action = agent.select_action(state, training=True)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = to_state(next_obs)
            done = bool(terminated or truncated)

            total_reward += reward
            steps += 1

            if args.agent == "sarsa":
                next_action = None if done else agent.select_action(next_state, training=True)
                agent.update(state, action, reward, next_state, done, next_action)
                state = next_state
                if next_action is None:
                    break
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

        logger.log_episode(episode, total_reward, steps, getattr(agent, "epsilon", 0.0))

        if episode % 100 == 0:
            print(
                f"Ep {episode}/{episodes} | reward={total_reward:.2f} | steps={steps} | epsilon={getattr(agent, 'epsilon', 0.0):.3f}"
            )

    print("Training done. TODO: implement agent.save() to persist models.")
    logger.close()


if __name__ == "__main__":
    main()
