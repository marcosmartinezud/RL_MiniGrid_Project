"""Expert planner that solves MultiRoom layouts via BFS on the grid."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from minigrid.core.constants import DIR_TO_VEC


ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_TOGGLE = 5

ACTION_INDEX: Dict[int, int] = {ACTION_LEFT: 0, ACTION_RIGHT: 1, ACTION_FORWARD: 2, ACTION_TOGGLE: 3}


def _is_blocked(cell) -> bool:
    if cell is None:
        return False
    if getattr(cell, "type", None) == "wall":
        return True
    if getattr(cell, "type", None) == "lava":
        return True
    return False


def bfs_path(grid, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    if start == goal:
        return [start]
    queue: Deque[Tuple[int, int]] = deque([start])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= grid.width or ny >= grid.height:
                continue
            next_pos = (nx, ny)
            if next_pos in came_from:
                continue
            cell = grid.get(nx, ny)
            if _is_blocked(cell):
                continue
            came_from[next_pos] = (x, y)
            if next_pos == goal:
                queue.clear()
                break
            queue.append(next_pos)

    if goal not in came_from:
        return [start]

    path: List[Tuple[int, int]] = []
    current: Optional[Tuple[int, int]] = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def direction_to(target_vec: Tuple[int, int]) -> int:
    vec = np.array(target_vec)
    for idx, dvec in enumerate(DIR_TO_VEC):
        if int(dvec[0]) == int(vec[0]) and int(dvec[1]) == int(vec[1]):
            return idx
    return 0


def compute_path_info(env) -> Tuple[int, int]:
    uw = env.unwrapped
    grid = uw.grid
    start = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))
    goal = (int(uw.goal_pos[0]), int(uw.goal_pos[1]))
    path = bfs_path(grid, start, goal)
    if len(path) < 2:
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        desired_dir = int(uw.agent_dir)
        if manhattan > 0:
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            if abs(dx) >= abs(dy):
                desired_vec = (1 if dx > 0 else -1, 0)
            else:
                desired_vec = (0, 1 if dy > 0 else -1)
            desired_dir = direction_to(desired_vec)
        return int(manhattan), int(desired_dir)

    next_pos = path[1]
    dx = next_pos[0] - start[0]
    dy = next_pos[1] - start[1]
    desired_dir = direction_to((dx, dy))
    return int(len(path) - 1), int(desired_dir)


def action_from_direction(env, desired_dir: int) -> int:
    uw = env.unwrapped
    agent_dir = int(uw.agent_dir)
    if agent_dir != int(desired_dir):
        diff = (int(desired_dir) - agent_dir) % 4
        if diff == 1:
            return ACTION_RIGHT
        if diff == 3:
            return ACTION_LEFT
        return ACTION_RIGHT

    front_pos = tuple(int(x) for x in uw.front_pos)
    cell = uw.grid.get(*front_pos)
    if cell is not None and getattr(cell, "type", None) == "door" and not cell.is_open:
        return ACTION_TOGGLE
    return ACTION_FORWARD


class ExpertPlanner:
    """Simple expert that plans a shortest path and follows it."""

    def next_action(self, env) -> int:
        _dist, desired_dir = compute_path_info(env)
        return action_from_direction(env, desired_dir)

    def next_action_index(self, env) -> int:
        action = self.next_action(env)
        return ACTION_INDEX[action]
