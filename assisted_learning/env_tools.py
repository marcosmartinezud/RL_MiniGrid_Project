"""Helpers for extracting compact state and shaping metrics."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

ACTION_MAP: List[int] = [0, 1, 2, 5]  # left, right, forward, toggle
GOAL_IDX = 8
DOOR_IDX = 4


def scan_doors(env) -> List[Tuple[int, int]]:
    grid = env.unwrapped.grid
    doors: List[Tuple[int, int]] = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "door":
                doors.append((int(x), int(y)))
    doors.sort()
    return doors


def door_bits(env, door_positions: Iterable[Tuple[int, int]]) -> List[int]:
    grid = env.unwrapped.grid
    bits: List[int] = []
    for x, y in door_positions:
        cell = grid.get(x, y)
        is_open = bool(cell is not None and cell.type == "door" and cell.is_open)
        bits.append(1 if is_open else 0)
    return bits


def layout_signature(env, door_positions: Iterable[Tuple[int, int]]) -> Tuple[int, ...]:
    goal_pos = getattr(env.unwrapped, "goal_pos", None)
    if goal_pos is None:
        goal = (0, 0)
    else:
        goal = (int(goal_pos[0]), int(goal_pos[1]))
    flat_doors: List[int] = []
    for x, y in door_positions:
        flat_doors.extend([int(x), int(y)])
    return (goal[0], goal[1], *flat_doors)


def state_from_obs(obs: Any) -> Tuple[int, ...]:
    """Extract a compact, egocentric state from a MiniGrid observation dict."""
    if isinstance(obs, tuple):
        return obs
    if isinstance(obs, np.ndarray):
        return tuple(obs.tolist())
    if not isinstance(obs, dict):
        return tuple()

    image = obs.get("image")
    if image is None:
        return tuple()
    direction = int(obs.get("direction", 0))

    size_x, size_y = image.shape[0], image.shape[1]
    center_x = size_x // 2
    agent_y = size_y - 1
    front_y = max(agent_y - 1, 0)
    left_x = max(center_x - 1, 0)
    right_x = min(center_x + 1, size_x - 1)

    front = image[center_x, front_y]
    left = image[left_x, agent_y]
    right = image[right_x, agent_y]

    front_type = int(front[0])
    front_state = int(front[2])
    left_type = int(left[0])
    right_type = int(right[0])

    obj_layer = image[:, :, 0]
    goal_visible = bool((obj_layer == GOAL_IDX).any())
    door_visible = bool((obj_layer == DOOR_IDX).any())

    return (direction, front_type, front_state, left_type, right_type, goal_visible, door_visible)


def state_from_env(env, door_positions: Iterable[Tuple[int, int]], include_layout: bool = True) -> Tuple[int, ...]:
    uw = env.unwrapped
    ax, ay = uw.agent_pos
    agent = (int(ax), int(ay), int(uw.agent_dir))
    doors = tuple(door_bits(env, door_positions))
    if include_layout:
        return (*layout_signature(env, door_positions), *agent, *doors)
    return (*agent, *doors)


def shortest_path_length(env, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
    if start == goal:
        return 0
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            pos = (nx, ny)
            if pos in visited:
                continue
            cell = grid.get(nx, ny)
            if cell is not None and getattr(cell, "type", None) == "wall":
                continue
            if pos == goal:
                return dist + 1
            visited.add(pos)
            queue.append((pos, dist + 1))
    return None


def build_state(
    obs: Any,
    env,
    door_positions: Iterable[Tuple[int, int]],
    state_mode: str,
    path_info: Tuple[int, int] | None = None,
    distance_bucket: int = 1,
    max_distance: int = 50,
) -> Tuple[int, ...]:
    mode = state_mode.lower()
    if mode == "layout":
        return state_from_env(env, door_positions, include_layout=True)

    local_state = state_from_obs(obs)
    if mode == "local":
        return local_state
    if mode != "bfs":
        raise ValueError(f"Unsupported state_mode: {state_mode}")

    if len(local_state) >= 7:
        _direction, front_type, front_state, _left_type, _right_type, goal_visible, door_visible = local_state[:7]
    else:
        front_type = 0
        front_state = 0
        goal_visible = 0
        door_visible = 0

    if path_info is None:
        distance = None
        desired_dir = None
    else:
        distance, desired_dir = path_info

    if distance is None:
        uw = env.unwrapped
        ax, ay = uw.agent_pos
        gx, gy = uw.goal_pos
        distance = abs(int(ax) - int(gx)) + abs(int(ay) - int(gy))

    if desired_dir is None:
        desired_dir = int(env.unwrapped.agent_dir)

    bucket_size = max(1, int(distance_bucket))
    distance_bucketed = int(distance) // bucket_size
    if max_distance is not None:
        distance_bucketed = min(distance_bucketed, int(max_distance))

    agent_dir = int(env.unwrapped.agent_dir)
    desired_turn = int((int(desired_dir) - agent_dir) % 4)
    return (
        int(front_type),
        int(front_state),
        int(goal_visible),
        int(door_visible),
        desired_turn,
        int(distance_bucketed),
    )
