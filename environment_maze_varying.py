"""
VARYING maze + cognitive map: the genuine "build a map, then plan on it" task.

Why this exists. The fixed-maze test was invalid: both models scored 0.94 on
the training maze but collapsed to ~0.68 on a novel maze (at/below the greedy
baseline of ~0.70), i.e. they MEMORISED one layout and learned no transferable
navigation strategy. Memorisation, not planning -- the same class of validity
flaw as the open-plan task being greedy-solvable.

Here a FRESH maze is generated every episode, so memorisation is impossible by
construction. To act well the model must, within a single sequence:
  1. infer the maze's connectivity from what it walked through (explore),
  2. remember WHERE it saw the goal landmark (cognitive map),
  3. plan a route to it through the doors (navigate).

Scale is chosen so map-building is actually feasible: a 16x16 world (256 cells)
with 4x4 rooms, explored for T_explore steps before the goal is revealed. In a
64x64 world a 64-step walk sees ~2% of the space and the task would be
impossible rather than hard.

Goal selection: the goal is always a landmark the agent ACTUALLY OBSERVED
during exploration, so the information needed is present in the sequence.

Headroom: a greedy "head toward the goal" policy ignores walls and scores
~0.7. Beating that requires using the explored map. Chance = 0.25.
"""
from __future__ import annotations
from collections import deque
from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


class VaryingMazeWorld:
    N_ACTIONS = GridWorld.N_ACTIONS
    ACTION_DELTAS = GridWorld.ACTION_DELTAS

    def __init__(self, size: int = 16, rooms_per_side: int = 4,
                 n_obs_types: int = 8, n_landmarks: int = 16,
                 seed: Optional[int] = None):
        assert size % rooms_per_side == 0
        self.size = size
        self.R = rooms_per_side
        self.rs = size // rooms_per_side
        self.n_obs_types = n_obs_types
        self.n_landmarks = n_landmarks

        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.blank_token = n_obs_types
        self.first_landmark = self.obs_offset + n_obs_types + 1
        self.unified_vocab_size = self.first_landmark + n_landmarks
        self._rng = np.random.RandomState(seed)

    # ---------------------------------------------------------------- maze
    def _sample_maze(self, rng):
        """Random spanning tree over the room graph -> doors. Returns (dv, dh)."""
        R, rs = self.R, self.rs
        dv = np.full((R, R), -1, dtype=np.int64)
        dh = np.full((R, R), -1, dtype=np.int64)
        seen = {(0, 0)}
        frontier = []

        def push(i, j):
            frontier.extend([
                ((i + 1) % R, j, 'v', i, j),
                ((i - 1) % R, j, 'v', (i - 1) % R, j),
                (i, (j + 1) % R, 'h', i, j),
                (i, (j - 1) % R, 'h', i, (j - 1) % R)])
        push(0, 0)
        while frontier and len(seen) < R * R:
            k = rng.randint(0, len(frontier))
            ni, nj, kind, ei, ej = frontier.pop(k)
            if (ni, nj) in seen:
                continue
            seen.add((ni, nj))
            if kind == 'v':
                dv[ei, ej] = ej * rs + rng.randint(0, rs)
            else:
                dh[ei, ej] = ei * rs + rng.randint(0, rs)
            push(ni, nj)
        return dv, dh

    def _can_move(self, dv, dh, x, y, a):
        ddx, ddy = self.ACTION_DELTAS[a]
        rs, R = self.rs, self.R
        if ddx == 1 and (x + 1) % rs == 0:
            return y == dv[x // rs, y // rs]
        if ddx == -1 and x % rs == 0:
            return y == dv[(x // rs - 1) % R, y // rs]
        if ddy == 1 and (y + 1) % rs == 0:
            return x == dh[x // rs, y // rs]
        if ddy == -1 and y % rs == 0:
            return x == dh[x // rs, (y // rs - 1) % R]
        return True

    def _step(self, dv, dh, x, y, a):
        if not self._can_move(dv, dh, x, y, a):
            return x, y
        ddx, ddy = self.ACTION_DELTAS[a]
        return (x + ddx) % self.size, (y + ddy) % self.size

    def _bfs(self, dv, dh, start, goal):
        if start == goal:
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for a in range(self.N_ACTIONS):
                nxt = self._step(dv, dh, cur[0], cur[1], a)
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

    # ------------------------------------------------------------- episode
    def generate_episode(self, T_explore: int = 256, T_navigate: int = 48,
                         rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = self._rng
        S = self.size
        dv, dh = self._sample_maze(rng)

        obs = rng.randint(0, self.n_obs_types, size=(S, S))
        lm_cells = {}
        flat = rng.permutation(S * S)[:self.n_landmarks]
        for idx, f in enumerate(flat):
            lm_cells[(int(f // S), int(f % S))] = idx

        x, y = int(rng.randint(0, S)), int(rng.randint(0, S))
        tokens: list[int] = []
        seen_lms: dict[int, tuple[int, int]] = {}
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            x, y = self._step(dv, dh, x, y, a)
            tokens.append(a + self.action_offset)
            if (x, y) in lm_cells:
                li = lm_cells[(x, y)]
                seen_lms[li] = (x, y)
                tokens.append(self.first_landmark + li)
            else:
                tokens.append(int(obs[x, y]) + self.obs_offset)

        if not seen_lms:
            return None  # caller resamples
        # Pick the FARTHEST observed landmark (by true maze distance). Choosing
        # a random seen landmark yields ~6-step paths -- too short to require
        # planning -- because recently-seen landmarks are nearby.
        cand = []
        for li_, (lx, ly) in seen_lms.items():
            p_ = self._bfs(dv, dh, (x, y), (lx, ly))
            cand.append((len(p_), li_, (lx, ly), p_))
        cand.sort(key=lambda t: -t[0])
        _, li, (gx, gy), path = cand[0]
        goal_token = self.first_landmark + li

        is_bfs = []
        for i in range(T_navigate):
            if i < len(path):
                a = path[i]; is_bfs.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS)); is_bfs.append(False)
            x, y = self._step(dv, dh, x, y, a)
            tokens.append(a + self.action_offset)
            if (x, y) in lm_cells:
                tokens.append(self.first_landmark + lm_cells[(x, y)])
            else:
                tokens.append(int(obs[x, y]) + self.obs_offset)

        full = torch.tensor([goal_token] + tokens, dtype=torch.long)
        amask = torch.zeros(full.shape[0], dtype=torch.bool)
        for i in range(T_navigate):
            if is_bfs[i]:
                amask[2 * (T_explore + i)] = True
        return full, amask, {"bfs_len": len(path), "goal_idx": li,
                             "start": (x, y), "path": path}

    def generate_batch(self, B, T_explore, T_navigate, rng):
        tk, am = [], []
        while len(tk) < B:
            r = self.generate_episode(T_explore, T_navigate, rng)
            if r is None:
                continue
            tk.append(r[0]); am.append(r[1])
        return torch.stack(tk), torch.stack(am)
