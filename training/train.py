"""Training loop for tabular Q-learning/SARSA and DQN on MiniGrid MultiRoom."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
from typing import Any, Hashable, Iterable

import gymnasium as gym
import minigrid  # registers MiniGrid envs with gymnasium
import numpy as np
import yaml

from agents import DQNAgent, QLearningAgent, SarsaAgent
from environments import (
    FlatFloatObsWrapper,
    FlatObsWrapper,
    PositionAwareWrapper,
    RewardShapingWrapper,
    SimpleObsWrapper,
)
from utils import TensorBoardLogger


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """Load YAML hyperparameters."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(
    env_name: str,
    max_steps: int = 500,
    wrapper_type: str = "position",
    shaping_config: dict[str, Any] | None = None,
) -> gym.Env:
    """Create a MiniGrid env with selectable wrapper and optional reward shaping."""
    base_env = gym.make(env_name, max_steps=max_steps)

    if wrapper_type == "position":
        wrapped: gym.Env = PositionAwareWrapper(base_env)
    elif wrapper_type == "simple":
        wrapped = SimpleObsWrapper(base_env)
    elif wrapper_type == "dqn":
        wrapped = FlatFloatObsWrapper(base_env)
    else:
        wrapped = FlatObsWrapper(base_env)

    if shaping_config and shaping_config.get("enabled", False):
        wrapped = RewardShapingWrapper(wrapped, shaping_config)
    return wrapped


def to_state(obs: Any) -> Hashable:
    """Ensure observations are hashable tuples for tabular agents."""
    if isinstance(obs, tuple):
        return obs
    if isinstance(obs, np.ndarray):
        return obs
    try:
        return tuple(obs)
    except TypeError:
        return tuple()


def build_agent(name: str, state_size: int, action_size: int, config: dict[str, Any]) -> Any:
    """Factory for Q-learning, SARSA, or DQN agent."""
    if name == "qlearning":
        return QLearningAgent(state_size, action_size, config["q_learning"])
    if name == "sarsa":
        return SarsaAgent(state_size, action_size, config["sarsa"])
    if name == "dqn":
        return DQNAgent(state_size, action_size, config["dqn"])
    raise ValueError(f"Unsupported agent: {name}")


def default_log_dir(agent: str, curriculum: bool, shaped: bool) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if curriculum:
        suffix += "_curriculum"
    if shaped:
        suffix += "_shaped"
    return f"logs/{agent}{suffix}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agents on MiniGrid MultiRoom")
    parser.add_argument("--agent", type=str, choices=["qlearning", "sarsa", "dqn"], default="qlearning")
    parser.add_argument("--env", type=str, default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent.parent / "config" / "hyperparams.yaml"),
        help="Path to hyperparameter YAML file",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom TensorBoard log directory")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning stages")
    parser.add_argument("--shaped", action="store_true", help="Enable reward shaping")
    parser.add_argument(
        "--wrapper",
        type=str,
        choices=["simple", "position", "flat", "dqn"],
        default="position",
        help="Observation wrapper to use",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    max_steps = config["training"].get("max_steps", 100)
    shaping_config = config.get("reward_shaping", {}) if args.shaped else None

    # Force appropriate wrapper for DQN
    wrapper_choice = "dqn" if args.agent == "dqn" else args.wrapper

    # Prepare curriculum stages if enabled
    stages: Iterable[dict[str, Any]]
    if args.curriculum:
        stages = config.get("curriculum", {}).get("stages", [])
        if not stages:
            raise ValueError("Curriculum flag set but no stages defined in config")
        episodes = sum(int(s["episodes"]) for s in stages)
        first_env = stages[0]["env"]
    else:
        episodes = args.episodes or config["training"].get("num_episodes", 5000)
        stages = [{"env": args.env, "episodes": episodes}]
        first_env = args.env

    env = make_env(first_env, max_steps=max_steps, wrapper_type=wrapper_choice, shaping_config=shaping_config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = build_agent(args.agent, state_size, action_size, config)
    log_dir = args.log_dir or default_log_dir(args.agent, args.curriculum, args.shaped)
    logger = TensorBoardLogger(log_dir)

    print(f"Starting training with {args.agent}")
    print(f"Total episodes: {episodes}, max_steps per episode: {max_steps}")
    print(f"Logging to: {log_dir}")
    print(f"Reward shaping: {'enabled' if args.shaped else 'disabled'}")
    print(f"Wrapper: {wrapper_choice}")

    global_episode = 0
    for idx, stage in enumerate(stages, start=1):
        env_name = stage["env"]
        stage_episodes = int(stage["episodes"])
        if idx > 1:
            print(f"Switching to Stage {idx}/{len(stages)}: {env_name}")
            env = make_env(env_name, max_steps=max_steps, wrapper_type=wrapper_choice, shaping_config=shaping_config)
        else:
            print(f"Stage {idx}/{len(stages)}: Training on {env_name} for {stage_episodes} episodes")

        for _ in range(stage_episodes):
            global_episode += 1
            obs, _ = env.reset()
            state = to_state(obs)
            total_reward = 0.0
            steps = 0
            success = False
            last_loss: float | None = None

            if args.agent == "sarsa":
                action = agent.select_action(state, training=True)

            for _ in range(max_steps):
                if args.agent in {"qlearning", "dqn"}:
                    action = agent.select_action(state, training=True)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = to_state(next_obs)
                done = bool(terminated or truncated)

                total_reward += reward
                steps += 1
                if terminated and reward > 0:
                    success = True

                if args.agent == "sarsa":
                    next_action = None if done else agent.select_action(next_state, training=True)
                    agent.update(state, action, reward, next_state, done, next_action)
                    state = next_state
                    if next_action is None:
                        break
                    action = next_action
                elif args.agent == "dqn":
                    agent.store_transition(state, action, reward, next_state, done)
                    loss_val = agent.update()
                    if loss_val is not None:
                        last_loss = loss_val
                    state = next_state
                    if done:
                        break
                else:
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break

            if args.agent == "dqn":
                agent.decay_epsilon()

            logger.log_episode_with_window(
                global_episode,
                total_reward,
                steps,
                getattr(agent, "epsilon", 0.0),
                success,
                last_loss,
            )

            if global_episode % 100 == 0:
                print(
                    f"Ep {global_episode}/{episodes} | reward={total_reward:.2f} | steps={steps} | epsilon={getattr(agent, 'epsilon', 0.0):.3f}"
                )

    # Save model at the end of training
    if args.agent == "dqn":
        model_path = pathlib.Path(log_dir) / "model.pt"
        agent.save(str(model_path))
        print(f"Model saved to {model_path}")
    
    print("Training done.")
    logger.close()


if __name__ == "__main__":
    main()
