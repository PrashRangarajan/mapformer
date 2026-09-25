"""
Rooms-with-WALLS-and-DOORS: a genuine hierarchical navigation problem.

The open-plan RoomsGoalWorld turned out to be a non-test: with no obstacles,
100% of BFS-optimal actions are simply "reduce distance to goal", so a greedy
direction-following policy is optimal and there is NO route-planning problem
for a hierarchy to decompose. Flat and hierarchical models tied exactly, as
they must.

Here rooms are separated by walls, with exactly ONE door per adjacent room
pair. Movement across a room boundary is only possible at that door. Now:

  - greedy "head toward the goal" FAILS (walls block the direct line),
  - the route genuinely decomposes: which sequence of ROOMS (coarse graph)
    -> which DOOR -> local steps within the room (fine),
  - a flat model must memorise the whole maze topology; a hierarchical one
    can factor it into a small room-graph plan plus local navigation.

This is the structure under which hierarchy demonstrably wins in the
options/HRL literature, and (per our derived principle) the coarse room-level
route IS a sufficient statistic for the routing decision — so hierarchy is
permitted to help here, unlike any exact-recall task.

All cells remain occupiable; walls restrict *crossings* (edges), not cells.
"""
from __future__ import annotations
from collections import deque
from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


class RoomsMazeWorld:
    N_ACTIONS = GridWorld.N_ACTIONS
    ACTION_DELTAS = GridWorld.ACTION_DELTAS

    def __init__(self, size: int = 64, n_obs_types: int = 16,
                 rooms_per_side: int = 8, theme_size: int = 3,
                 connectivity: str = "full", seed: Optional[int] = None):
        """connectivity: 'full' = door on every adjacent room pair (weak maze);
        'tree' = doors only on a random spanning tree of the room graph, so
        exactly ONE (often winding) route connects any two rooms. 'tree' is the
        real hierarchical-planning regime: greedy fails and the room-graph route
        must actually be planned."""
        self.connectivity = connectivity
        assert size % rooms_per_side == 0
        self.size = size
        self.R = rooms_per_side
        self.rs = size // rooms_per_side
        self.n_obs_types = n_obs_types

        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.blank_token = n_obs_types
        self.n_goals = self.R * self.R
        self.first_goal_unified = self.obs_offset + n_obs_types + 1
        self.unified_vocab_size = self.first_goal_unified + self.n_goals

        rng = np.random.RandomState(seed)
        # Room themes (coarse structure also visible in observations)
        self.room_themes = np.zeros((self.R, self.R, theme_size), dtype=np.int64)
        for i in range(self.R):
            for j in range(self.R):
                self.room_themes[i, j] = rng.choice(n_obs_types, theme_size, replace=False)
        obs = np.zeros((size, size), dtype=np.int64)
        for x in range(size):
            for y in range(size):
                th = self.room_themes[x // self.rs, y // self.rs]
                obs[x, y] = th[rng.randint(0, theme_size)]
        self.obs_map = torch.from_numpy(obs).long()

        # Doors. doors_v[i,j] = absolute y of the door on the boundary between
        # room (i,j) and room (i+1,j). doors_h[i,j] = absolute x of the door
        # between room (i,j) and room (i,j+1). Torus-wrapping boundaries.
        self.doors_v = np.full((self.R, self.R), -1, dtype=np.int64)
        self.doors_h = np.full((self.R, self.R), -1, dtype=np.int64)
        if connectivity == "full":
            open_v = [(i, j) for i in range(self.R) for j in range(self.R)]
            open_h = list(open_v)
        else:
            # Randomised Prim over the room graph (torus-adjacent rooms).
            open_v, open_h = [], []
            seen = {(0, 0)}
            frontier = []
            def push(i, j):
                for (ni, nj, kind, ei, ej) in (
                    ((i + 1) % self.R, j, 'v', i, j),
                    ((i - 1) % self.R, j, 'v', (i - 1) % self.R, j),
                    (i, (j + 1) % self.R, 'h', i, j),
                    (i, (j - 1) % self.R, 'h', i, (j - 1) % self.R)):
                    frontier.append(((i, j), (ni, nj), kind, ei, ej))
            push(0, 0)
            while frontier and len(seen) < self.R * self.R:
                k = rng.randint(0, len(frontier))
                _, nb, kind, ei, ej = frontier.pop(k)
                if nb in seen:
                    continue
                seen.add(nb)
                (open_v if kind == 'v' else open_h).append((ei, ej))
                push(*nb)
        for (i, j) in open_v:
            self.doors_v[i, j] = j * self.rs + rng.randint(0, self.rs)
        for (i, j) in open_h:
            self.doors_h[i, j] = i * self.rs + rng.randint(0, self.rs)

        self.goal_cells = [(i * self.rs + rng.randint(0, self.rs),
                            j * self.rs + rng.randint(0, self.rs))
                           for i in range(self.R) for j in range(self.R)]

    def room_of(self, x, y):
        return (x // self.rs, y // self.rs)

    def can_move(self, x, y, a) -> bool:
        dx, dy = self.ACTION_DELTAS[a]
        S, rs, R = self.size, self.rs, self.R
        if dx == 1:
            if (x + 1) % rs == 0:
                return y == self.doors_v[x // rs, y // rs]
        elif dx == -1:
            if x % rs == 0:
                return y == self.doors_v[(x // rs - 1) % R, y // rs]
        elif dy == 1:
            if (y + 1) % rs == 0:
                return x == self.doors_h[x // rs, y // rs]
        elif dy == -1:
            if y % rs == 0:
                return x == self.doors_h[x // rs, (y // rs - 1) % R]
        return True

    def step(self, x, y, a):
        """Returns new (x,y); blocked moves are no-ops."""
        if not self.can_move(x, y, a):
            return x, y
        dx, dy = self.ACTION_DELTAS[a]
        return (x + dx) % self.size, (y + dy) % self.size

    def bfs(self, start, goal) -> list[int]:
        """Shortest action sequence respecting walls. [] if already at goal."""
        if start == goal:
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for a in range(self.N_ACTIONS):
                nxt = self.step(cur[0], cur[1], a)
                if nxt != cur and nxt not in prev:
                    prev[nxt] = (cur, a)
                    q.append(nxt)
        if goal not in prev:
            return []
        path, cur = [], goal
        while prev[cur] is not None:
            cur, a = prev[cur]
            path.append(a)
        return path[::-1]

    def generate_goal_episode(self, T_explore=64, T_navigate=64,
                              rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random
        x, y = int(rng.randint(0, self.size)), int(rng.randint(0, self.size))
        g_idx = int(rng.randint(0, self.n_goals))
        gx, gy = self.goal_cells[g_idx]
        goal_token = self.first_goal_unified + g_idx

        tokens: list[int] = []
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            x, y = self.step(x, y, a)
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        rdist = self._room_dist(self.room_of(x, y), self.room_of(gx, gy))
        path = self.bfs((x, y), (gx, gy))

        is_bfs = []
        for i in range(T_navigate):
            if i < len(path):
                a = path[i]; is_bfs.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS)); is_bfs.append(False)
            x, y = self.step(x, y, a)
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        full = torch.tensor([goal_token] + tokens, dtype=torch.long)
        amask = torch.zeros(full.shape[0], dtype=torch.bool)
        for i in range(T_navigate):
            if is_bfs[i]:
                amask[2 * (T_explore + i)] = True
        return full, amask, {"room_distance": rdist, "bfs_len": len(path), "goal_idx": g_idx}

    def _room_dist(self, r1, r2):
        R = self.R
        di = min(abs(r1[0] - r2[0]), R - abs(r1[0] - r2[0]))
        dj = min(abs(r1[1] - r2[1]), R - abs(r1[1] - r2[1]))
        return max(di, dj)
