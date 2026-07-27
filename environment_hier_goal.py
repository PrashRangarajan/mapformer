"""Hierarchical goal-directed navigation on a room-tiled torus.

Motivation
----------
Our compositional results showed MapFormer (multi-scale position, via the omega
spectrum) and a time-hierarchy (Hourglass) help ORTHOGONAL metrics, with no
clean synergy -- because the task never forced reasoning about absolute position
at multiple scales AT ONCE. This task creates exactly that demand.

Episode:  [room_goal, local_goal,  explore (a,o)*T_e,  navigate (a,o)*T_n]

- START is a FIXED anchor (0,0) every episode, so path integration yields
  ABSOLUTE position (displacement from a known origin), not just relative drift.
- The goal is HIERARCHICAL: a room id (coarse -- which room_size x room_size
  block) and a local id (fine -- which cell within that block). The target cell
  = room_origin(room) + local_offset(local). Reaching it requires knowing
  absolute position at BOTH scales simultaneously: which room you are in vs the
  target room (coarse displacement) AND where within the room (fine).
- EXPLORE = random walk; forces path integration to track position before the
  goal must be acted on.
- NAVIGATE = BFS-optimal actions from the end-of-explore position to the target.
  Loss = next-action cross-entropy at navigate positions. Chance = 1/4.
- obs_map is REDRAWN per episode (aliased, fresh) so obs cannot be memorised for
  absolute localisation -- path integration from the fixed anchor is the
  reliable signal.

Oracle: knowing start, the explore actions, and the goal, current position is
exact and BFS is optimal -> 100%. So chance 0.25, ceiling 1.00.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .environment import GridWorld
from .environment_goal import bfs_torus


class HierGoalGridWorld(GridWorld):
    def __init__(self, size: int = 64, room_size: int = 8, n_obs_types: int = 16,
                 p_empty: float = 0.5, seed: Optional[int] = None):
        assert size % room_size == 0
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                         n_landmarks=0, seed=seed)
        self.room_size = room_size
        self.rooms_per_side = size // room_size
        self.n_rooms = self.rooms_per_side ** 2
        self.n_local = room_size ** 2
        # Extend the vocab with hierarchical goal tokens above the base grid vocab
        # (base = [0..3] actions, [4..4+K-1] obs, [4+K] blank).
        base = self.unified_vocab_size
        self.room_tok0 = base
        self.local_tok0 = base + self.n_rooms
        self.unified_vocab_size = base + self.n_rooms + self.n_local

    def _draw_obs(self, rng):
        obs = np.full((self.size, self.size), self.blank_token, dtype=np.int64)
        occ = rng.random((self.size, self.size)) >= self.p_empty
        obs[occ] = rng.randint(0, self.n_obs_types, int(occ.sum()))
        return obs

    def room_local_to_cell(self, room: int, local: int) -> tuple[int, int]:
        rri, rrj = divmod(room, self.rooms_per_side)
        lx, ly = divmod(local, self.room_size)
        return rri * self.room_size + lx, rrj * self.room_size + ly

    def generate_hier_episode(self, T_explore: int = 64, T_navigate: int = 64,
                              rng=None, goal=None, start=(0, 0)):
        if rng is None:
            rng = np.random
        obs_map = self._draw_obs(rng)
        if goal is None:
            room = int(rng.randint(0, self.n_rooms))
            local = int(rng.randint(0, self.n_local))
        else:
            room, local = goal
        gx, gy = self.room_local_to_cell(room, local)
        room_tok = self.room_tok0 + room
        local_tok = self.local_tok0 + local

        x, y = start
        tokens: list[int] = []
        for _ in range(T_explore):                       # explore: random walk
            a = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(obs_map[x, y]) + self.obs_offset)

        bfs_path = bfs_torus((x, y), (gx, gy), self.size)
        is_bfs: list[bool] = []
        for i in range(T_navigate):                      # navigate: BFS to target
            if i < len(bfs_path):
                a = bfs_path[i]; is_bfs.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS)); is_bfs.append(False)
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(obs_map[x, y]) + self.obs_offset)

        full = torch.tensor([room_tok, local_tok] + tokens, dtype=torch.long)
        L = full.shape[0]
        # Layout: 0=room_goal, 1=local_goal, then (a,o) pairs from index 2.
        obs_mask = torch.zeros(L, dtype=torch.bool); obs_mask[3::2] = True
        act_mask = torch.zeros(L, dtype=torch.bool)
        for i, was_bfs in enumerate(is_bfs):
            if was_bfs:
                pred_pos = 1 + 2 * T_explore + 2 * i     # token preceding nav action i
                act_mask[pred_pos] = True
        info = {"goal_room": room, "goal_local": local, "goal_cell": (gx, gy),
                "bfs_distance": len(bfs_path), "start": tuple(start),
                "T_explore": T_explore, "T_navigate": T_navigate}
        return full, obs_mask, act_mask, info

    def generate_hier_batch(self, batch_size: int, T_explore: int = 64,
                            T_navigate: int = 64, rng=None):
        toks, oms, ams, infos = [], [], [], []
        for _ in range(batch_size):
            t, om, am, info = self.generate_hier_episode(T_explore, T_navigate, rng=rng)
            toks.append(t); oms.append(om); ams.append(am); infos.append(info)
        return torch.stack(toks), torch.stack(oms), torch.stack(ams), infos
