"""Visualize the expert planner solving random MultiRoom layouts."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import minigrid  # registers envs

from salvacion.expert_planner import ExpertPlanner


def reset_seed(seed: int | None, random_layout: bool, episode: int) -> int | None:
    if random_layout:
        if seed is None:
            return None
        return seed + episode
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize expert planner in MultiRoom")
    parser.add_argument("--env", type=str, default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.03)
    parser.add_argument("--random-layout", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human", max_steps=args.max_steps)
    expert = ExpertPlanner()

    try:
        for ep in range(1, args.episodes + 1):
            reset = reset_seed(args.seed, args.random_layout, ep)
            obs, _ = env.reset(seed=reset)
            done = False
            steps = 0
            total_reward = 0.0

            while not done and steps < args.max_steps:
                env.render()
                action = expert.next_action(env)
                obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += reward
                steps += 1
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
