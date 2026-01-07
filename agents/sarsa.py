"""On-policy SARSA implementation for MiniGrid MultiRoom.

This mirrors the provided reference algorithm and shares utilities with the
Q-learning module (wrappers, reward shaping helpers, checkpointing, and logging).
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .q_learning import (
    EnvName,
    compact_state_from_env,
    door_in_front,
    get_max_steps,
    load_q_table,
    make_env,
    manhattan_to_goal,
    obs_to_state,
)


def select_action(q_table, state, n_actions: int, epsilon: float) -> int:
    if random.random() < epsilon or state not in q_table:
        return random.randrange(n_actions)
    return int(np.argmax(q_table[state]))


def sarsa(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    step_penalty: float,
    door_bonus: float,
    success_scale: float,
    progress_scale: float,
    toggle_penalty: float,
    checkpoint_every: int,
    checkpoint_path: str,
    allowed_actions: Tuple[int, ...],
    env_name: str,
    fully_obs: bool,
    max_steps_override: int | None,
    seed: int | None = None,
    initial_q_table=None,
    log_dir: str | None = None,
):
    random.seed(seed)
    np.random.seed(seed)

    env = make_env(render_mode=None, seed=seed, fully_obs=fully_obs, env_name=env_name)
    n_actions = len(allowed_actions)
    max_steps = get_max_steps(env, max_steps_override)

    if initial_q_table is not None:
        q_table: Dict[Tuple[int, bytes], np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32), initial_q_table
        )
    else:
        q_table: Dict[Tuple[int, bytes], np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32)
        )

    returns = []
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    for ep in range(episodes):
        td_err2_sum = 0.0
        td_count = 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** ep))
        obs, info = env.reset()
        state = compact_state_from_env(env) if fully_obs else obs_to_state(obs)
        action = select_action(q_table, state, n_actions, epsilon)
        total_reward = 0.0
        door_was_open = False
        prev_dist = manhattan_to_goal(state) if fully_obs else 0

        for _ in range(max_steps):
            env_action = allowed_actions[action]
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            next_state = compact_state_from_env(env) if fully_obs else obs_to_state(next_obs)

            shaped_reward = reward * success_scale + step_penalty

            if fully_obs and env_action == 5 and not door_in_front(env):
                shaped_reward += toggle_penalty

            if fully_obs:
                _, _, _, door_open, *rest = next_state
                if door_open and not door_was_open:
                    shaped_reward += door_bonus
                door_was_open = bool(door_open)

                next_dist = manhattan_to_goal(next_state)
                if next_dist < prev_dist:
                    shaped_reward += progress_scale * (prev_dist - next_dist)
                prev_dist = next_dist

            next_action = select_action(q_table, next_state, n_actions, epsilon) if not done else 0
            best_next = 0.0 if done else q_table[next_state][next_action]
            td_target = shaped_reward + gamma * best_next
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            td_err2_sum += float(td_error * td_error)
            td_count += 1

            state, action = next_state, next_action
            total_reward += reward
            if done:
                break

        returns.append(total_reward)

        if writer:
            writer.add_scalar("train/return", total_reward, ep + 1)
            writer.add_scalar("train/epsilon", epsilon, ep + 1)
            if td_count > 0:
                writer.add_scalar("train/td_error2", td_err2_sum / td_count, ep + 1)

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0:
            path = checkpoint_path.format(ep=ep + 1, env=env_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(dict(q_table), f)
            print(f"Saved checkpoint at episode {ep + 1} -> {path}")

        if (ep + 1) % 500 == 0:
            avg = np.mean(returns[-500:])
            print(f"Episode {ep + 1}/{episodes} | epsilon={epsilon:.3f} | avg_reward(last 500)={avg:.3f}")

    env.close()
    if writer:
        writer.flush()
        writer.close()
    return q_table, returns


def evaluate_policy(
    q_table,
    episodes: int,
    fully_obs: bool,
    max_steps_override: int | None,
    seed: int | None = None,
    env_name: str = EnvName,
):
    env = make_env(render_mode=None, seed=seed, fully_obs=fully_obs, env_name=env_name)
    allowed_actions = (0, 1, 2, 5) if fully_obs else tuple(range(env.action_space.n))
    n_actions = len(allowed_actions)
    max_steps = get_max_steps(env, max_steps_override)

    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        state = compact_state_from_env(env) if fully_obs else obs_to_state(obs)
        total_reward = 0.0

        for _ in range(max_steps):
            action = int(np.argmax(q_table[state])) if state in q_table else random.randrange(n_actions)
            env_action = allowed_actions[action]
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            state = compact_state_from_env(env) if fully_obs else obs_to_state(next_obs)
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
    env.close()
    return rewards


def render_episode(
    q_table,
    fully_obs: bool,
    max_steps_override: int | None,
    seed: int | None = None,
    env_name: str = EnvName,
):
    env = make_env(render_mode="human", seed=seed, fully_obs=fully_obs, env_name=env_name)
    allowed_actions = (0, 1, 2, 5) if fully_obs else tuple(range(env.action_space.n))
    n_actions = len(allowed_actions)
    max_steps = get_max_steps(env, max_steps_override)

    obs, info = env.reset()
    state = compact_state_from_env(env) if fully_obs else obs_to_state(obs)
    total_reward = 0.0

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state])) if state in q_table else random.randrange(n_actions)
        env_action = allowed_actions[action]
        obs, reward, terminated, truncated, _ = env.step(env_action)
        env.render()
        total_reward += reward
        state = compact_state_from_env(env) if fully_obs else obs_to_state(obs)
        if terminated or truncated:
            break

    print(f"Render episode reward: {total_reward:.3f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="SARSA for MiniGrid MultiRoom")
    parser.add_argument("--env", type=str, default=EnvName, help="Env id")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--door-bonus", type=float, default=0.2)
    parser.add_argument("--success-scale", type=float, default=1.0)
    parser.add_argument("--progress-scale", type=float, default=0.1)
    parser.add_argument("--toggle-penalty", type=float, default=-0.02)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/{env}_sarsa_ep{ep}.pkl",
    )
    parser.add_argument("--save-q", type=str, default=None)
    parser.add_argument("--load-q", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fully-obs", action="store_true")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    probe_env = make_env(env_name=args.env)
    allowed_actions = (0, 1, 2, 5) if args.fully_obs else tuple(range(probe_env.action_space.n))
    probe_env.close()

    init_q = None
    if args.load_q:
        init_q = load_q_table(args.load_q, n_actions=len(allowed_actions))
        print(f"Loaded initial Q-table from {args.load_q}")

    q_table, returns = sarsa(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        step_penalty=args.step_penalty,
        door_bonus=args.door_bonus,
        success_scale=args.success_scale,
        progress_scale=args.progress_scale,
        toggle_penalty=args.toggle_penalty,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.checkpoint_path,
        allowed_actions=allowed_actions,
        env_name=args.env,
        fully_obs=args.fully_obs,
        max_steps_override=args.max_steps,
        seed=args.seed,
        initial_q_table=init_q,
        log_dir=args.logdir,
    )

    rewards = evaluate_policy(
        q_table,
        episodes=args.eval_episodes,
        fully_obs=args.fully_obs,
        max_steps_override=args.max_steps,
        seed=args.seed,
        env_name=args.env,
    )
    print(f"Eval mean reward over {args.eval_episodes} episodes: {np.mean(rewards):.3f}")

    if args.logdir:
        writer = SummaryWriter(log_dir=args.logdir)
        writer.add_scalar("eval/mean_return", float(np.mean(rewards)), len(returns))
        writer.flush()
        writer.close()

    if args.save_q:
        os.makedirs(os.path.dirname(args.save_q) or ".", exist_ok=True)
        with open(args.save_q, "wb") as f:
            pickle.dump(dict(q_table), f)
        print(f"Saved final Q-table to {args.save_q}")

    if args.render:
        render_episode(q_table, fully_obs=args.fully_obs, max_steps_override=args.max_steps, seed=args.seed, env_name=args.env)


if __name__ == "__main__":
    main()

