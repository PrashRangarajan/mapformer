"""Optimal solver for MiniGrid-DoorKey envs via state-space BFS.

State: (x, y, dir, has_key, door_open). Transitions cover the 6
non-trivial MiniGrid actions (turn left/right, forward, pickup, toggle).
Drop and Done are not useful for the optimal solution.

The solver reads env.unwrapped.grid to find Key/Door/Goal positions
(MiniGrid is fully observable to the solver — partial observability is
the model's problem).

Used to generate expert demonstrations for behavioural-cloning training
on MiniGrid-DoorKey-* environments.
"""

from __future__ import annotations

from collections import deque
from typing import Optional


DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # MiniGrid convention


def _find_objects(env) -> dict[str, tuple[int, int]]:
    grid = env.unwrapped.grid
    out: dict[str, tuple[int, int]] = {}
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is None: continue
            t = getattr(cell, "type", cell.__class__.__name__.lower())
            if t in ("key", "door", "goal"):
                out[t] = (x, y)
    return out


def _transition(state, action, grid, objs):
    x, y, d, has_key, door_open = state
    if action == 0:
        return (x, y, (d - 1) % 4, has_key, door_open)
    if action == 1:
        return (x, y, (d + 1) % 4, has_key, door_open)
    dx, dy = DIR_VEC[d]
    fx, fy = x + dx, y + dy
    if not (0 <= fx < grid.width and 0 <= fy < grid.height):
        return None
    front = grid.get(fx, fy)
    front_type = getattr(front, "type", None) if front is not None else None
    key_pos = objs.get("key"); door_pos = objs.get("door"); goal_pos = objs.get("goal")
    front_is_key = (fx, fy) == key_pos and not has_key
    front_is_door = (fx, fy) == door_pos
    front_is_wall = front_type == "wall"
    front_is_goal = (fx, fy) == goal_pos

    if action == 2:  # forward
        if front_is_wall: return None
        if front_is_key: return None  # key blocks
        if front_is_door and not door_open: return None
        return (fx, fy, d, has_key, door_open)
    if action == 3:  # pickup
        if front_is_key:
            return (x, y, d, True, door_open)
        return None
    if action == 5:  # toggle door with key
        if front_is_door and has_key and not door_open:
            return (x, y, d, has_key, True)
        return None
    return None


def solve_doorkey(env) -> Optional[list[int]]:
    """BFS for optimal action sequence solving the current DoorKey env state."""
    objs = _find_objects(env)
    if "goal" not in objs:
        return None
    agent_x, agent_y = int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])
    agent_dir = int(env.unwrapped.agent_dir)
    initial = (agent_x, agent_y, agent_dir, False, False)
    goal_xy = objs["goal"]
    grid = env.unwrapped.grid

    visited = {initial}
    queue = deque([(initial, [])])
    while queue:
        state, path = queue.popleft()
        x, y, *_ = state
        if (x, y) == goal_xy:
            return path
        for a in (0, 1, 2, 3, 5):  # skip drop / done
            ns = _transition(state, a, grid, objs)
            if ns is not None and ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [a]))
    return None
