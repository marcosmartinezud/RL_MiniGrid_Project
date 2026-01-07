"""Training loop for tabular Q-learning/SARSA and DQN on MiniGrid MultiRoom."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import pickle
from typing import Any, Hashable, Iterable

import gymnasium as gym
import minigrid
import numpy as np
import yaml

from agents import DQNAgent
from agents.q_learning import (
    evaluate_policy as eval_q_learning,
    load_q_table,
    make_env as make_q_env,
    q_learning as run_q_learning,
)
from agents.sarsa import evaluate_policy as eval_sarsa
from agents.sarsa import sarsa as run_sarsa
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
    """Factory for DQN agent (tabular methods handled separately)."""
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
    parser.add_argument("--env", type=str, default="MiniGrid-MultiRoom-N2-S4-v0")
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
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode")
    parser.add_argument("--fully-obs", action="store_true", help="Use FullyObsWrapper for tabular Q-learning")
    parser.add_argument(
        "--wrapper",
        type=str,
        choices=["simple", "position", "flat", "dqn"],
        default="position",
        help="Observation wrapper to use",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    max_steps = args.max_steps or config["training"].get("max_steps", 100)
    shaping_config = config.get("reward_shaping", {}) if args.shaped else None

    if args.agent == "qlearning":
        if args.curriculum:
            print("Curriculum is ignored for tabular Q-learning; using a single environment.")
        if args.shaped:
            print("Reward shaping wrapper flag is ignored for tabular Q-learning; built-in shaping terms are used instead.")

        q_cfg = config["q_learning"]
        episodes = args.episodes or config["training"].get("num_episodes", 50000)
        fully_obs = args.fully_obs or bool(q_cfg.get("fully_obs", False))
        env_name = args.env

        if args.max_steps is not None:
            max_steps_override = args.max_steps
        elif q_cfg.get("max_steps") is not None:
            max_steps_override = int(q_cfg["max_steps"])
        else:
            max_steps_override = config["training"].get("max_steps")

        log_dir = args.log_dir or default_log_dir("qlearning", False, False)

        probe_env = make_q_env(env_name=env_name, fully_obs=fully_obs)
        allowed_actions = (0, 1, 2, 5) if fully_obs else tuple(range(probe_env.action_space.n))
        probe_env.close()

        init_q = None
        if q_cfg.get("load_q"):
            init_q = load_q_table(q_cfg["load_q"], n_actions=len(allowed_actions))
            print(f"Loaded initial Q-table from {q_cfg['load_q']}")

        checkpoint_path = q_cfg.get("checkpoint_path", "checkpoints/q_table_ep{ep}.pkl")
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str(pathlib.Path(log_dir) / checkpoint_path)

        q_table, returns = run_q_learning(
            episodes=episodes,
            alpha=q_cfg.get("alpha", 0.1),
            gamma=q_cfg.get("gamma", 0.99),
            epsilon_start=q_cfg.get("epsilon_start", 1.0),
            epsilon_end=q_cfg.get("epsilon_end", 0.1),
            epsilon_decay=q_cfg.get("epsilon_decay", 0.9995),
            step_penalty=q_cfg.get("step_penalty", -0.01),
            door_bonus=q_cfg.get("door_bonus", 0.2),
            success_scale=q_cfg.get("success_scale", 1.0),
            progress_scale=q_cfg.get("progress_scale", 0.1),
            toggle_penalty=q_cfg.get("toggle_penalty", -0.02),
            checkpoint_every=int(q_cfg.get("checkpoint_every", 0) or 0),
            checkpoint_path=checkpoint_path,
            allowed_actions=allowed_actions,
            env_name=env_name,
            fully_obs=fully_obs,
            max_steps_override=max_steps_override,
            seed=q_cfg.get("seed", 0),
            initial_q_table=init_q,
            log_dir=log_dir,
        )

        eval_eps = int(q_cfg.get("eval_episodes", 50))
        rewards = eval_q_learning(
            q_table,
            episodes=eval_eps,
            fully_obs=fully_obs,
            max_steps_override=max_steps_override,
            seed=q_cfg.get("seed", 0),
            env_name=env_name,
        )
        print(f"Eval mean reward over {eval_eps} episodes: {np.mean(rewards):.3f}")

        save_path = q_cfg.get("save_q")
        if save_path is None:
            save_path = str(pathlib.Path(log_dir) / "q_table_final.pkl")
        os.makedirs(pathlib.Path(save_path).parent, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(dict(q_table), f)
        print(f"Saved final Q-table to {save_path}")

        return

    if args.agent == "sarsa":
        if args.curriculum:
            print("Curriculum is ignored for tabular SARSA; using a single environment.")
        if args.shaped:
            print("Reward shaping wrapper flag is ignored for tabular SARSA; built-in shaping terms are used instead.")

        s_cfg = config["sarsa"]
        episodes = args.episodes or config["training"].get("num_episodes", 50000)
        fully_obs = args.fully_obs or bool(s_cfg.get("fully_obs", False))
        env_name = args.env

        if args.max_steps is not None:
            max_steps_override = args.max_steps
        elif s_cfg.get("max_steps") is not None:
            max_steps_override = int(s_cfg["max_steps"])
        else:
            max_steps_override = config["training"].get("max_steps")

        log_dir = args.log_dir or default_log_dir("sarsa", False, False)

        probe_env = make_q_env(env_name=env_name, fully_obs=fully_obs)
        allowed_actions = (0, 1, 2, 5) if fully_obs else tuple(range(probe_env.action_space.n))
        probe_env.close()

        init_q = None
        if s_cfg.get("load_q"):
            init_q = load_q_table(s_cfg["load_q"], n_actions=len(allowed_actions))
            print(f"Loaded initial Q-table from {s_cfg['load_q']}")

        checkpoint_path = s_cfg.get("checkpoint_path", "checkpoints/{env}_sarsa_ep{ep}.pkl")
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str(pathlib.Path(log_dir) / checkpoint_path)

        q_table, returns = run_sarsa(
            episodes=episodes,
            alpha=s_cfg.get("alpha", 0.1),
            gamma=s_cfg.get("gamma", 0.99),
            epsilon_start=s_cfg.get("epsilon_start", 1.0),
            epsilon_end=s_cfg.get("epsilon_end", 0.1),
            epsilon_decay=s_cfg.get("epsilon_decay", 0.9995),
            step_penalty=s_cfg.get("step_penalty", -0.01),
            door_bonus=s_cfg.get("door_bonus", 0.2),
            success_scale=s_cfg.get("success_scale", 1.0),
            progress_scale=s_cfg.get("progress_scale", 0.1),
            toggle_penalty=s_cfg.get("toggle_penalty", -0.02),
            checkpoint_every=int(s_cfg.get("checkpoint_every", 0) or 0),
            checkpoint_path=checkpoint_path,
            allowed_actions=allowed_actions,
            env_name=env_name,
            fully_obs=fully_obs,
            max_steps_override=max_steps_override,
            seed=s_cfg.get("seed", 0),
            initial_q_table=init_q,
            log_dir=log_dir,
        )

        eval_eps = int(s_cfg.get("eval_episodes", 50))
        rewards = eval_sarsa(
            q_table,
            episodes=eval_eps,
            fully_obs=fully_obs,
            max_steps_override=max_steps_override,
            seed=s_cfg.get("seed", 0),
            env_name=env_name,
        )
        print(f"Eval mean reward over {eval_eps} episodes: {np.mean(rewards):.3f}")

        save_path = s_cfg.get("save_q")
        if save_path is None:
            save_path = str(pathlib.Path(log_dir) / "sarsa_table_final.pkl")
        os.makedirs(pathlib.Path(save_path).parent, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(dict(q_table), f)
        print(f"Saved final Q-table to {save_path}")

        return

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
