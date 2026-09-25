"""
Hierarchical goal-directed navigation: rooms (hierarchical space) + distant goals.

This is the regime where hierarchy demonstrably wins elsewhere (options/HRL,
Swin-style multi-scale features): reaching a distant goal DECOMPOSES into
coarse (which rooms to traverse) + fine (where within a room). A flat model
must learn shortest paths over all size^2 cells; a hierarchical one can factor
the problem into a small room-graph plan plus local steps.

Crucially this is NOT exact-recall — it is planning, where a coarse summary IS
a sufficient statistic for the routing decision. Our derived principle
("hierarchy helps only when a lossy summary is a sufficient statistic")
therefore predicts hierarchy CAN help here, unlike every retrieval task tested.

Key experimental variable: ROOM DISTANCE (how many room-hops to the goal).
Prediction: hierarchy's advantage should GROW with room distance.

Layout: size x size torus split into rooms_per_side^2 rooms. Each room has a
THEME (small subset of obs types its cells draw from) and exactly ONE goal
cell with a unique goal token, so "go to goal token g" == "go to room r, then
to its goal cell".
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from .environment import GridWorld
from .environment_goal import bfs_torus


class RoomsGoalWorld:
    N_ACTIONS = GridWorld.N_ACTIONS
    ACTION_DELTAS = GridWorld.ACTION_DELTAS

    def __init__(self, size: int = 64, n_obs_types: int = 16,
                 rooms_per_side: int = 8, theme_size: int = 3,
                 p_empty: float = 0.0, seed: Optional[int] = None):
        assert size % rooms_per_side == 0
        self.size = size
        self.n_obs_types = n_obs_types
        self.rooms_per_side = rooms_per_side
        self.room_size = size // rooms_per_side
        self.theme_size = theme_size

        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.blank_token = n_obs_types
        self.unified_blank = self.obs_offset + self.blank_token
        self.n_goals = rooms_per_side * rooms_per_side
        self.first_goal_unified = self.obs_offset + n_obs_types + 1
        self.unified_vocab_size = self.first_goal_unified + self.n_goals

        rng = np.random.RandomState(seed)
        R = rooms_per_side
        self.room_themes = np.zeros((R, R, theme_size), dtype=np.int64)
        for i in range(R):
            for j in range(R):
                self.room_themes[i, j] = rng.choice(n_obs_types, theme_size, replace=False)

        obs_map = np.zeros((size, size), dtype=np.int64)
        for x in range(size):
            for y in range(size):
                if rng.random() < p_empty:
                    obs_map[x, y] = self.blank_token
                else:
                    th = self.room_themes[x // self.room_size, y // self.room_size]
                    obs_map[x, y] = th[rng.randint(0, theme_size)]
        self.obs_map = torch.from_numpy(obs_map).long()

        # One goal cell per room; goal index = flattened room index.
        self.goal_cells: list[tuple[int, int]] = []
        for i in range(R):
            for j in range(R):
                gx = i * self.room_size + rng.randint(0, self.room_size)
                gy = j * self.room_size + rng.randint(0, self.room_size)
                self.goal_cells.append((gx, gy))

    def room_of(self, x: int, y: int) -> tuple[int, int]:
        return (x // self.room_size, y // self.room_size)

    def room_distance(self, r1: tuple[int, int], r2: tuple[int, int]) -> int:
        """Chebyshev distance on the room grid, with torus wrap."""
        R = self.rooms_per_side
        di = min(abs(r1[0] - r2[0]), R - abs(r1[0] - r2[0]))
        dj = min(abs(r1[1] - r2[1]), R - abs(r1[1] - r2[1]))
        return max(di, dj)

    def generate_goal_episode(self, T_explore: int = 64, T_navigate: int = 64,
                              rng: Optional[np.random.RandomState] = None):
        """Returns tokens, action_target_mask, info(room_distance, bfs_len)."""
        if rng is None:
            rng = np.random
        x = int(rng.randint(0, self.size)); y = int(rng.randint(0, self.size))
        g_idx = int(rng.randint(0, self.n_goals))
        gx, gy = self.goal_cells[g_idx]
        goal_token = self.first_goal_unified + g_idx

        tokens: list[int] = []
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        start_room = self.room_of(x, y)
        rdist = self.room_distance(start_room, self.room_of(gx, gy))
        bfs_path = bfs_torus((x, y), (gx, gy), self.size)

        is_bfs = []
        for i in range(T_navigate):
            if i < len(bfs_path):
                a = bfs_path[i]; is_bfs.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS)); is_bfs.append(False)
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        full = torch.tensor([goal_token] + tokens, dtype=torch.long)
        L = full.shape[0]
        action_target_mask = torch.zeros(L, dtype=torch.bool)
        for i in range(T_navigate):
            if is_bfs[i]:
                act_pos = 1 + 2 * (T_explore + i)      # index of that action token
                action_target_mask[act_pos - 1] = True  # predict it from the prior position
        info = {"room_distance": rdist, "bfs_len": len(bfs_path), "goal_idx": g_idx}
        return full, action_target_mask, info
