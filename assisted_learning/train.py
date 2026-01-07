"""Train a fast tabular Q-learning agent on MultiRoom."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
from collections import defaultdict, deque
from typing import Any, Dict

import gymnasium as gym
import minigrid  # registers envs
import numpy as np
import yaml

from assisted_learning.env_tools import (
    ACTION_MAP,
    build_state,
    scan_doors,
    shortest_path_length,
)
from assisted_learning.expert_planner import ACTION_INDEX, action_from_direction, compute_path_info
from assisted_learning.evaluate import evaluate_agent
from assisted_learning.qlearning_agent import QLearningAgent, QLearningConfig
from utils import TensorBoardLogger


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reset_seed(seed: int | None, random_layout: bool, episode: int) -> int | None:
    if random_layout:
        if seed is None:
            return None
        return seed + episode
    return seed


def distance_to_goal(env) -> int:
    uw = env.unwrapped
    start = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))
    goal = (int(uw.goal_pos[0]), int(uw.goal_pos[1]))
    dist = shortest_path_length(env, start, goal)
    if dist is None:
        dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return int(dist)


def default_log_dir() -> pathlib.Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return pathlib.Path("logs") / f"assisted_learning_qlearning_{timestamp}"


def train(config: Dict[str, Any], output_dir: pathlib.Path) -> Dict[str, Any]:
    env_name = config.get("env", "MiniGrid-MultiRoom-N4-S5-v0")
    seed = config.get("seed", None)
    random_layout = bool(config.get("random_layout", False))
    training_cfg = config.get("training", {})
    episodes = int(training_cfg.get("episodes", 4000))
    max_steps = int(training_cfg.get("max_steps", 400))
    log_every = int(training_cfg.get("log_every", 100))
    success_window = int(training_cfg.get("success_window", 200))
    stop_success_rate = float(training_cfg.get("stop_success_rate", 0.85))
    min_episodes = int(training_cfg.get("min_episodes", 500))

    state_mode = str(config.get("state_mode", "local")).lower()
    if state_mode not in {"local", "layout", "bfs"}:
        raise ValueError(f"Unsupported state_mode: {state_mode}")

    state_cfg = config.get("state", {})
    distance_bucket = int(state_cfg.get("distance_bucket", 1))
    max_distance = int(state_cfg.get("max_distance", 50))

    shaping_cfg = config.get("shaping", {})
    shaping_enabled = bool(shaping_cfg.get("enabled", False))
    step_penalty = float(shaping_cfg.get("step_penalty", -0.001)) if shaping_enabled else 0.0
    door_open_bonus = float(shaping_cfg.get("door_open_bonus", 0.2)) if shaping_enabled else 0.0
    distance_weight = float(shaping_cfg.get("distance_weight", 0.02)) if shaping_enabled else 0.0
    exploration_weight = float(shaping_cfg.get("exploration_weight", 0.005)) if shaping_enabled else 0.0

    expert_cfg = config.get("expert", {})
    expert_enabled = bool(expert_cfg.get("enabled", False))
    prefill_steps = int(expert_cfg.get("prefill_steps", 0)) if expert_enabled else 0
    mix_prob = float(expert_cfg.get("mix_prob", 0.0)) if expert_enabled else 0.0
    mix_episodes = int(expert_cfg.get("mix_episodes", 0)) if expert_enabled else 0
    expert_bonus = float(expert_cfg.get("expert_bonus", 0.0)) if expert_enabled else 0.0

    q_cfg = QLearningConfig(**config.get("q_learning", {}))
    agent = QLearningAgent(len(ACTION_MAP), q_cfg)

    env = gym.make(env_name, max_steps=max_steps)
    if seed is not None:
        np.random.seed(seed)

    log_dir_cfg = config.get("log_dir")
    log_dir = pathlib.Path(log_dir_cfg) if log_dir_cfg else default_log_dir()
    logger = TensorBoardLogger(log_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.csv"
    summary_path = output_dir / "summary.json"
    model_path = output_dir / "q_table.pkl"

    success_tracker: deque[int] = deque(maxlen=success_window)

    use_path_info = state_mode == "bfs" or distance_weight != 0 or expert_enabled

    def compute_shaped_reward(
        reward: float,
        action: int,
        front_pos: tuple[int, int],
        opened_doors: set[tuple[int, int]],
        prev_dist: float,
        next_dist: float,
        pos_visits: Dict[tuple[int, int], int],
    ) -> float:
        shaped_reward = reward
        if action == 5 and door_open_bonus != 0:
            cell = env.unwrapped.grid.get(*front_pos)
            if cell is not None and cell.type == "door" and cell.is_open:
                if front_pos not in opened_doors:
                    opened_doors.add(front_pos)
                    shaped_reward += door_open_bonus

        if distance_weight != 0:
            shaped_reward += distance_weight * (prev_dist - next_dist)

        if exploration_weight != 0:
            pos_key = tuple(int(x) for x in env.unwrapped.agent_pos)
            visits = pos_visits[pos_key]
            shaped_reward += exploration_weight / np.sqrt(1 + visits)
            pos_visits[pos_key] = visits + 1

        if step_penalty != 0:
            shaped_reward += step_penalty

        return shaped_reward

    if expert_enabled and prefill_steps > 0:
        print(f"Prefill experto: {prefill_steps} pasos")
        prefill_done = 0
        prefill_episode = 0
        while prefill_done < prefill_steps:
            prefill_episode += 1
            reset = reset_seed(seed, random_layout, prefill_episode)
            obs, _ = env.reset(seed=reset)
            door_positions = scan_doors(env) if state_mode == "layout" else []
            opened_doors: set[tuple[int, int]] = set()
            pos_visits: Dict[tuple[int, int], int] = defaultdict(int)
            path_info = compute_path_info(env) if use_path_info else None
            prev_dist = (
                float(path_info[0])
                if distance_weight != 0 and path_info is not None
                else float(distance_to_goal(env) if distance_weight != 0 else 0)
            )
            state = build_state(obs, env, door_positions, state_mode, path_info, distance_bucket, max_distance)
            done = False

            while not done and prefill_done < prefill_steps:
                front_pos = tuple(int(x) for x in env.unwrapped.front_pos)
                if path_info is None and use_path_info:
                    path_info = compute_path_info(env)
                desired_dir = int(path_info[1]) if path_info is not None else int(env.unwrapped.agent_dir)
                action = action_from_direction(env, desired_dir)
                action_idx = ACTION_INDEX[action]

                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

                next_path_info = compute_path_info(env) if use_path_info else None
                next_state = build_state(
                    next_obs, env, door_positions, state_mode, next_path_info, distance_bucket, max_distance
                )

                if distance_weight != 0:
                    next_dist = (
                        float(next_path_info[0]) if next_path_info is not None else float(distance_to_goal(env))
                    )
                else:
                    next_dist = prev_dist

                shaped_reward = compute_shaped_reward(
                    reward,
                    action,
                    front_pos,
                    opened_doors,
                    prev_dist,
                    next_dist,
                    pos_visits,
                )
                agent.update(state, action_idx, shaped_reward, next_state, done)

                state = next_state
                if distance_weight != 0:
                    prev_dist = next_dist
                path_info = next_path_info
                prefill_done += 1

        print("Prefill experto completado.")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success", "epsilon", "success_rate"])

        total_episodes = episodes
        for episode in range(1, episodes + 1):
            reset = reset_seed(seed, random_layout, episode)
            obs, _ = env.reset(seed=reset)
            door_positions = scan_doors(env) if state_mode == "layout" else []
            opened_doors: set[tuple[int, int]] = set()
            pos_visits: Dict[tuple[int, int], int] = defaultdict(int)
            path_info = compute_path_info(env) if use_path_info else None
            state = build_state(obs, env, door_positions, state_mode, path_info, distance_bucket, max_distance)
            if distance_weight != 0:
                prev_dist = float(path_info[0]) if path_info is not None else float(distance_to_goal(env))
            else:
                prev_dist = 0.0
            total_reward = 0.0
            steps = 0
            success = False

            for _ in range(max_steps):
                front_pos = tuple(int(x) for x in env.unwrapped.front_pos)
                expert_action_idx = None
                if expert_enabled and (episode <= mix_episodes or expert_bonus != 0.0):
                    if path_info is None and use_path_info:
                        path_info = compute_path_info(env)
                    desired_dir = int(path_info[1]) if path_info is not None else int(env.unwrapped.agent_dir)
                    expert_action = action_from_direction(env, desired_dir)
                    expert_action_idx = ACTION_INDEX[expert_action]

                use_expert = (
                    expert_action_idx is not None
                    and episode <= mix_episodes
                    and mix_prob > 0.0
                    and np.random.rand() < mix_prob
                )
                if use_expert:
                    action_idx = expert_action_idx
                else:
                    action_idx = agent.select_action(state, training=True)

                action = ACTION_MAP[action_idx]
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

                next_path_info = compute_path_info(env) if use_path_info else None
                next_state = build_state(
                    next_obs, env, door_positions, state_mode, next_path_info, distance_bucket, max_distance
                )
                if distance_weight != 0:
                    next_dist = (
                        float(next_path_info[0]) if next_path_info is not None else float(distance_to_goal(env))
                    )
                else:
                    next_dist = prev_dist

                shaped_reward = compute_shaped_reward(
                    reward,
                    action,
                    front_pos,
                    opened_doors,
                    prev_dist,
                    next_dist,
                    pos_visits,
                )
                if expert_bonus != 0.0 and expert_action_idx is not None and action_idx == expert_action_idx:
                    shaped_reward += expert_bonus

                agent.update(state, action_idx, shaped_reward, next_state, done)

                total_reward += reward
                steps += 1
                state = next_state
                if distance_weight != 0:
                    prev_dist = next_dist
                path_info = next_path_info

                if done:
                    success = bool(terminated and reward > 0)
                    break

            agent.decay_epsilon()
            success_tracker.append(1 if success else 0)
            success_rate = sum(success_tracker) / len(success_tracker)

            writer.writerow([episode, f"{total_reward:.3f}", steps, int(success), f"{agent.epsilon:.4f}", f"{success_rate:.3f}"])
            logger.log_episode_with_window(
                episode,
                total_reward,
                steps,
                agent.epsilon,
                success,
                None,
            )
            logger.log_scalar("episode/success_rate_window", success_rate, episode)
            if episode % log_every == 0 or episode == 1:
                print(
                    f"Ep {episode}/{total_episodes} | reward={total_reward:.2f} "
                    f"| steps={steps} | epsilon={agent.epsilon:.3f} | success_rate={success_rate:.2f}"
                )

            if episode >= min_episodes and len(success_tracker) == success_window and success_rate >= stop_success_rate:
                print(f"Early stop at episode {episode}: success_rate={success_rate:.2f}")
                break

    agent.save(str(model_path))
    summary = {
        "episodes_completed": episode,
        "success_rate_window": float(success_rate),
        "epsilon": float(agent.epsilon),
        "model_path": str(model_path),
        "log_dir": str(log_dir),
        "shaping_enabled": shaping_enabled,
        "state_mode": state_mode,
        "expert_enabled": expert_enabled,
        "prefill_steps": int(prefill_steps),
        "mix_prob": float(mix_prob),
        "mix_episodes": int(mix_episodes),
        "expert_bonus": float(expert_bonus),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env.close()
    logger.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-learning agent")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--env", type=str, default=None, help="Override environment name")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed")
    parser.add_argument("--random-layout", action="store_true", help="Use random layouts each episode")
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
        args.output_dir or config.get("output_dir", pathlib.Path(__file__).resolve().parent / "output")
    )

    state_mode = str(config.get("state_mode", "local")).lower()
    if state_mode not in {"local", "layout", "bfs"}:
        raise ValueError(f"Unsupported state_mode: {state_mode}")

    summary = train(config, output_dir)
    print(f"Model saved to {summary['model_path']}")
    print(f"Logging to: {summary['log_dir']}")

    eval_episodes = int(config.get("training", {}).get("eval_episodes", 200))
    if eval_episodes > 0:
        state_cfg = config.get("state", {})
        distance_bucket = int(state_cfg.get("distance_bucket", 1))
        max_distance = int(state_cfg.get("max_distance", 50))
        agent = QLearningAgent(len(ACTION_MAP), QLearningConfig(**config.get("q_learning", {})))
        agent.load(summary["model_path"])
        metrics = evaluate_agent(
            agent,
            config.get("env", "MiniGrid-MultiRoom-N4-S5-v0"),
            eval_episodes,
            int(config.get("training", {}).get("max_steps", 400)),
            config.get("seed", None),
            bool(config.get("random_layout", False)),
            state_mode=state_mode,
            distance_bucket=distance_bucket,
            max_distance=max_distance,
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
