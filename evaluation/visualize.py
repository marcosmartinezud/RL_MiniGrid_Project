"""Visualize agents acting in MiniGrid with render_mode=human."""

from __future__ import annotations

import argparse
import time
from typing import Any, Hashable

import gymnasium as gym
import minigrid
import numpy as np

from agents import DQNAgent, QLearningAgent, SarsaAgent
from environments import FlatFloatObsWrapper, FlatObsWrapper, PositionAwareWrapper, SimpleObsWrapper


def make_env(env_name: str, wrapper_type: str, max_steps: int = 500) -> gym.Env:
    """Create MiniGrid env with human render and chosen wrapper."""
    base_env = gym.make(env_name, render_mode="human", max_steps=max_steps)
    if wrapper_type == "position":
        wrapped: gym.Env = PositionAwareWrapper(base_env)
    elif wrapper_type == "simple":
        wrapped = SimpleObsWrapper(base_env)
    elif wrapper_type == "dqn":
        wrapped = FlatFloatObsWrapper(base_env)
    else:
        wrapped = FlatObsWrapper(base_env)
    return wrapped


def to_state(obs: Any) -> Hashable:
    """Convert observation to a state for the chosen agent."""
    if isinstance(obs, (tuple, np.ndarray)):
        return obs
    try:
        return tuple(obs)
    except TypeError:
        return tuple()


def build_agent(name: str, state_size: int, action_size: int) -> Any:
    """Instantiate agent with lightweight defaults for inference."""
    dummy = {"gamma": 0.99}
    if name == "dqn":
        return DQNAgent(state_size, action_size, {"gamma": 0.99, "hidden_sizes": [256, 256]})
    if name == "qlearning":
        return QLearningAgent(state_size, action_size, dummy)
    if name == "sarsa":
        return SarsaAgent(state_size, action_size, dummy)
    raise ValueError(f"Unsupported agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize agent behavior with human rendering")
    parser.add_argument("--agent", type=str, choices=["dqn", "qlearning", "sarsa"], default="dqn")
    parser.add_argument("--wrapper", type=str, choices=["dqn", "simple", "position", "flat"], default="dqn")
    parser.add_argument("--env", type=str, default="MiniGrid-MultiRoom-N2-S4-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps in seconds")
    parser.add_argument("--model", type=str, default=None, help="Path to trained DQN weights")
    parser.add_argument("--random", action="store_true", help="Run a random policy for comparison")
    args = parser.parse_args()

    env = make_env(args.env, args.wrapper)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = None
    if not args.random:
        agent = build_agent(args.agent, state_size, action_size)
        if args.agent == "dqn":
            agent.epsilon = 0.05  # keep a tiny bit of exploration
            if args.model:
                try:
                    agent.load(args.model)
                    print(f"Loaded DQN weights from {args.model}")
                except Exception as exc:
                    print(f"Failed to load model: {exc}. Running untrained policy.")
        else:
            agent.epsilon = 1.0

    print(f"Visualizing agent={args.agent} wrapper={args.wrapper} env={args.env}")
    print(f"Random policy: {args.random}")

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            state = to_state(obs)
            done = False
            steps = 0
            total_reward = 0.0

            while not done:
                env.render()
                if args.random or agent is None:
                    action = env.action_space.sample()
                else:
                    if args.agent == "dqn":
                        action = agent.select_action(
                            state if isinstance(state, np.ndarray) else np.array(state, dtype=np.float32),
                            training=True,
                        )
                    else:
                        action = agent.select_action(state, training=True)

                next_obs, reward, terminated, truncated, _info = env.step(action)
                next_state = to_state(next_obs)
                total_reward += reward
                steps += 1
                done = bool(terminated or truncated)
                state = next_state

                if args.delay > 0:
                    time.sleep(args.delay)

            success = total_reward > 0
            print(f"Episode {ep}: reward={total_reward:.2f}, steps={steps}, success={success}")
    except KeyboardInterrupt:
        print("Visualization interrupted by user (Ctrl+C). Closing.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
