"""
Nested-room ("hierarchical space") environment.

The flat torus has no spatial hierarchy: every cell's observation is an
independent random draw, so an UNVISITED cell is fundamentally unpredictable
and the only useful operation is exact retrieval of visited cells. That task
structurally favours full-resolution flat attention (see HIER_ATTN_LONGT.md).

Here the space is hierarchically structured. The grid is partitioned into
rooms_per_side^2 rooms; each room gets a THEME — a small subset of obs types
that all of its cells draw from. That creates genuine coarse+fine structure:

  coarse (which room  -> which theme -> which obs types are possible)
  fine   (which cell  -> which specific obs)

Crucially this makes a NOVEL (never-visited) cell *partially predictable*:
you cannot retrieve it, but if you have inferred the room's theme from other
cells in that room you can narrow 16 types down to `theme_size`. That is a
spatial AGGREGATION, the operation pooling is good at — whereas exact revisit
prediction remains a retrieval (needle) task.

So one trained model yields two separable metrics:
  - revisit_mask     : retrieval  (needle)  -> expect flat attention to win
  - room_novel_mask  : theme inference (haystack) -> expect hierarchy to win

Interface mirrors GridWorld so existing models run unchanged.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


class RoomsGridWorld:
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
        self.p_empty = p_empty

        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.blank_token = n_obs_types                     # relative id of blank
        self.unified_blank = self.obs_offset + self.blank_token
        self.obs_vocab_size = n_obs_types + 1
        self.unified_vocab_size = self.N_ACTIONS + self.obs_vocab_size

        rng = np.random.RandomState(seed)
        # Per-room theme: theme_size distinct obs types.
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
                    theme = self.room_themes[x // self.room_size, y // self.room_size]
                    obs_map[x, y] = theme[rng.randint(0, theme_size)]
        self.obs_map = torch.from_numpy(obs_map).long()

    def room_of(self, x: int, y: int) -> tuple[int, int]:
        return (x // self.room_size, y // self.room_size)

    def generate_trajectory(self, n_steps: int = 128,
                            start: Optional[tuple[int, int]] = None):
        """Returns tokens, obs_mask, revisit_mask, room_novel_mask.

        room_novel_mask: True at an obs position whose cell is being visited for
        the FIRST time, but whose ROOM has already been visited (>=1 other cell
        seen there) -> the theme is inferable, so the obs is partially
        predictable without retrieval.
        """
        if start is not None:
            x, y = start
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)

        tokens: list[int] = []
        is_revisit: list[bool] = []
        is_room_novel: list[bool] = []
        seen_cells: set[tuple[int, int]] = set()
        seen_rooms: set[tuple[int, int]] = set()

        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.N_ACTIONS)
            k = np.random.randint(1, 11)
            for _ in range(k):
                if t >= n_steps:
                    break
                dx, dy = self.ACTION_DELTAS[a]
                x = (x + dx) % self.size
                y = (y + dy) % self.size

                tokens.append(a + self.action_offset)
                tokens.append(self.obs_map[x, y].item() + self.obs_offset)

                cell = (x, y); room = self.room_of(x, y)
                rev = cell in seen_cells
                is_revisit.append(rev)
                is_room_novel.append((not rev) and (room in seen_rooms))
                seen_cells.add(cell); seen_rooms.add(room)
                t += 1

        tokens_t = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool); obs_mask[1::2] = True
        revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        room_novel_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for s, (rev, rn) in enumerate(zip(is_revisit, is_room_novel)):
            if rev: revisit_mask[2 * s + 1] = True
            if rn: room_novel_mask[2 * s + 1] = True
        return tokens_t, obs_mask, revisit_mask, room_novel_mask
