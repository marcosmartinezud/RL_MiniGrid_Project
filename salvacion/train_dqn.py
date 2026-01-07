"""Train a dueling DQN agent on MultiRoom with random layouts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
from collections import deque
from typing import Any, Dict

import gymnasium as gym
import minigrid  # registers envs
import numpy as np
import yaml
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from salvacion.dqn_agent import (
    DQNAgent,
    DQNConfig,
    RewardShaper,
    ShapingConfig,
    preprocess_obs,
)
from salvacion.env_tools import ACTION_MAP
from salvacion.expert_planner import ACTION_INDEX, ExpertPlanner
from utils import TensorBoardLogger


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_name: str, max_steps: int, render_mode: str | None = None) -> gym.Env:
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


def evaluate_agent(
    agent: DQNAgent,
    env_name: str,
    episodes: int,
    max_steps: int,
    seed: int | None,
    random_layout: bool,
    target_size: int,
) -> Dict[str, float]:
    env = make_env(env_name, max_steps=max_steps)
    successes = 0
    total_reward = 0.0
    total_steps = 0

    prev_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        for ep in range(1, episodes + 1):
            reset = reset_seed(seed, random_layout, ep)
            obs, _ = env.reset(seed=reset)
            state = preprocess_obs(obs, target_size)
            done = False
            steps = 0
            ep_reward = 0.0

            while not done and steps < max_steps:
                action_idx = agent.select_action(state, training=False)
                action = ACTION_MAP[action_idx]
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += reward
                steps += 1
                state = preprocess_obs(next_obs, target_size)

            total_reward += ep_reward
            total_steps += steps
            if ep_reward > 0:
                successes += 1
    finally:
        agent.epsilon = prev_epsilon
        env.close()

    success_rate = successes / max(1, episodes)
    avg_reward = total_reward / max(1, episodes)
    avg_steps = total_steps / max(1, episodes)
    return {
        "success_rate": float(success_rate),
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
    }


def train(config: Dict[str, Any], output_dir: pathlib.Path) -> Dict[str, Any]:
    env_name = config.get("env", "MiniGrid-MultiRoom-N4-S5-v0")
    seed = config.get("seed", None)
    random_layout = bool(config.get("random_layout", True))
    obs_cfg = config.get("obs", {})
    target_size = int(obs_cfg.get("target_size", 14))

    training_cfg = config.get("training", {})
    episodes = int(training_cfg.get("episodes", 40000))
    max_steps = int(training_cfg.get("max_steps", 100))
    log_every = int(training_cfg.get("log_every", 200))
    success_window = int(training_cfg.get("success_window", 200))
    stop_success_rate = float(training_cfg.get("stop_success_rate", 0.85))
    min_episodes = int(training_cfg.get("min_episodes", 2000))

    dqn_cfg = DQNConfig(**config.get("dqn", {}))
    shaping_cfg = ShapingConfig(**config.get("shaping", {}))
    expert_cfg = config.get("expert", {})
    expert_prefill_steps = int(expert_cfg.get("prefill_steps", 0))
    expert_mix_prob = float(expert_cfg.get("mix_prob", 0.0))
    expert_mix_episodes = int(expert_cfg.get("mix_episodes", 0))

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch_seed = int(seed)
        import torch

        torch.manual_seed(torch_seed)

    env = make_env(env_name, max_steps=max_steps)
    agent = DQNAgent(len(ACTION_MAP), dqn_cfg)
    shaper = RewardShaper(shaping_cfg)
    expert = ExpertPlanner() if expert_prefill_steps > 0 or expert_mix_prob > 0 else None
    print(f"Dueling DQN device: {agent.device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path("logs") / f"salvacion_dueling_{timestamp}"
    logger = TensorBoardLogger(log_dir)
    logger.log_scalar("config/expert_prefill_steps", expert_prefill_steps, 0)
    logger.log_scalar("config/expert_mix_prob", expert_mix_prob, 0)

    model_path = output_dir / "dueling_dqn.pt"
    metrics_path = output_dir / "train_metrics.csv"
    summary_path = output_dir / "summary.json"

    success_tracker: deque[int] = deque(maxlen=success_window)
    best_success_rate = 0.0

    if expert is not None and expert_prefill_steps > 0:
        print(f"Prefilling replay buffer with {expert_prefill_steps} expert steps...")
        prefill_episode = 0
        obs, _ = env.reset(seed=reset_seed(seed, random_layout, prefill_episode))
        state = preprocess_obs(obs, target_size)
        for step in range(expert_prefill_steps):
            action = expert.next_action(env)
            action_idx = ACTION_INDEX[action]
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            shaped_reward = shaper.apply(env, action_idx, reward, done)
            next_state = preprocess_obs(next_obs, target_size)
            agent.store_transition(state, action_idx, shaped_reward, next_state, done)
            state = next_state
            if done:
                prefill_episode += 1
                obs, _ = env.reset(seed=reset_seed(seed, random_layout, prefill_episode))
                shaper.reset()
                state = preprocess_obs(obs, target_size)

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success", "epsilon", "success_rate", "loss"])

        for episode in range(1, episodes + 1):
            reset = reset_seed(seed, random_layout, episode)
            obs, _ = env.reset(seed=reset)
            shaper.reset()
            state = preprocess_obs(obs, target_size)

            total_reward = 0.0
            steps = 0
            success = False
            last_loss: float | None = None

            for _ in range(max_steps):
                use_expert = (
                    expert is not None
                    and episode <= expert_mix_episodes
                    and random.random() < expert_mix_prob
                )
                if use_expert:
                    action = expert.next_action(env)
                    action_idx = ACTION_INDEX[action]
                else:
                    action_idx = agent.select_action(state, training=True)
                    action = ACTION_MAP[action_idx]
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

                shaped_reward = shaper.apply(env, action_idx, reward, done)
                next_state = preprocess_obs(next_obs, target_size)
                agent.store_transition(state, action_idx, shaped_reward, next_state, done)

                loss = agent.update()
                if loss is not None:
                    last_loss = loss

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    success = bool(terminated and reward > 0)
                    break

            agent.decay_epsilon()
            success_tracker.append(1 if success else 0)
            success_rate = sum(success_tracker) / len(success_tracker)

            writer.writerow(
                [
                    episode,
                    f"{total_reward:.3f}",
                    steps,
                    int(success),
                    f"{agent.epsilon:.4f}",
                    f"{success_rate:.3f}",
                    "" if last_loss is None else f"{last_loss:.6f}",
                ]
            )
            logger.log_episode_with_window(
                episode,
                total_reward,
                steps,
                agent.epsilon,
                success,
                last_loss,
            )
            logger.log_scalar("episode/success_rate_window", success_rate, episode)
            logger.log_scalar("episode/doors_opened", shaper.doors_opened(), episode)

            if episode % log_every == 0 or episode == 1:
                print(
                    f"Ep {episode}/{episodes} | reward={total_reward:.2f} | steps={steps} "
                    f"| epsilon={agent.epsilon:.3f} | success_rate={success_rate:.2f}"
                )

            if success_rate > best_success_rate and episode >= min_episodes:
                best_success_rate = success_rate
                agent.save(str(model_path))

            if episode >= min_episodes and len(success_tracker) == success_window and success_rate >= stop_success_rate:
                print(f"Early stop at episode {episode}: success_rate={success_rate:.2f}")
                break

    # Save final model if none saved yet
    if not model_path.exists():
        agent.save(str(model_path))

    summary = {
        "episodes_completed": episode,
        "success_rate_window": float(success_rate),
        "epsilon": float(agent.epsilon),
        "model_path": str(model_path),
        "log_dir": str(log_dir),
        "random_layout": random_layout,
        "obs_target_size": target_size,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env.close()
    logger.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dueling DQN on MultiRoom")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "dqn_config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--env", type=str, default=None, help="Override environment name")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--random-layout", action="store_true", help="Force random layouts")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    config = load_config(config_path)
    if args.episodes is not None:
        config.setdefault("training", {})["episodes"] = int(args.episodes)
    if args.env is not None:
        config["env"] = args.env
    if args.seed is not None:
        config["seed"] = args.seed
    if args.random_layout:
        config["random_layout"] = True

    output_dir = pathlib.Path(
        args.output_dir or config.get("output_dir", pathlib.Path(__file__).resolve().parent / "dqn_output")
    )
    summary = train(config, output_dir)
    print(f"Model saved to {summary['model_path']}")
    print(f"Logging to: {summary['log_dir']}")

    eval_episodes = int(config.get("training", {}).get("eval_episodes", 200))
    if eval_episodes > 0:
        agent = DQNAgent(len(ACTION_MAP), DQNConfig(**config.get("dqn", {})))
        agent.load(summary["model_path"])
        metrics = evaluate_agent(
            agent,
            config.get("env", "MiniGrid-MultiRoom-N4-S5-v0"),
            eval_episodes,
            int(config.get("training", {}).get("max_steps", 100)),
            config.get("seed", None),
            bool(config.get("random_layout", True)),
            int(config.get("obs", {}).get("target_size", 14)),
        )
        metrics_path = output_dir / "eval_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(
            "Eval results: "
            f"success_rate={metrics['success_rate']:.3f} "
            f"avg_reward={metrics['avg_reward']:.3f} "
            f"avg_steps={metrics['avg_steps']:.1f}"
        )


if __name__ == "__main__":
    main()
