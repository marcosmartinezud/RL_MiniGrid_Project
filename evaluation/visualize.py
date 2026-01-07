"""Visualize agents acting in MiniGrid with render_mode=human."""

from __future__ import annotations

import argparse
import time
from typing import Any, Hashable

import gymnasium as gym
import minigrid
import numpy as np

from agents import DQNAgent
from agents.q_learning import load_q_table, make_env as make_q_env, render_episode
from agents.sarsa import render_episode as render_sarsa
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
    if name == "dqn":
        return DQNAgent(state_size, action_size, {"gamma": 0.99, "hidden_sizes": [256, 256]})
    raise ValueError(f"Unsupported agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize agent behavior with human rendering")
    parser.add_argument("--agent", type=str, choices=["dqn", "qlearning", "sarsa"], default="dqn")
    parser.add_argument("--wrapper", type=str, choices=["dqn", "simple", "position", "flat"], default="dqn")
    parser.add_argument("--env", type=str, default="MiniGrid-MultiRoom-N2-S4-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps in seconds")
    parser.add_argument("--fully-obs", action="store_true", help="Use FullyObsWrapper for tabular Q-learning")
    parser.add_argument("--q-table", type=str, default=None, help="Path to a saved Q-table (pickle) for Q-learning")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed")
    parser.add_argument("--model", type=str, default=None, help="Path to trained DQN weights")
    parser.add_argument("--random", action="store_true", help="Run a random policy for comparison")
    args = parser.parse_args()

    if args.agent in {"qlearning", "sarsa"}:
        fully_obs = args.fully_obs
        q_table = {}

        if not args.random:
            probe_env = make_q_env(env_name=args.env, fully_obs=fully_obs)
            allowed_actions = (0, 1, 2, 5) if fully_obs else tuple(range(probe_env.action_space.n))
            probe_env.close()
            if args.q_table:
                q_table = load_q_table(args.q_table, n_actions=len(allowed_actions))
                print(f"Loaded Q-table from {args.q_table}")
            else:
                print("No Q-table provided; running with an empty table (random actions).")
        else:
            print("Random policy enabled; ignoring Q-table path if provided.")

        print(f"Visualizing tabular {args.agent} | env={args.env} | fully_obs={fully_obs}")

        render_fn = render_episode if args.agent == "qlearning" else render_sarsa

        try:
            for ep in range(1, args.episodes + 1):
                render_fn(
                    q_table,
                    fully_obs=fully_obs,
                    max_steps_override=args.max_steps,
                    seed=args.seed,
                    env_name=args.env,
                )
                if args.delay > 0:
                    time.sleep(args.delay)
        except KeyboardInterrupt:
            print("Visualization interrupted by user (Ctrl+C). Closing.")
        return

    env = make_env(args.env, args.wrapper, max_steps=args.max_steps or 500)
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
            obs, _ = env.reset(seed=args.seed)
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
