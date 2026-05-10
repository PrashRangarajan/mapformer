"""Goal-directed extension of GridWorld.

Episode structure (token sequence after interleaving):

    [goal_token, a_1, o_1, a_2, o_2, ..., a_E, o_E,        # explore phase (random walk)
                 a_{E+1}, o_{E+1}, ..., a_{E+N}, o_{E+N}]  # navigate phase (BFS to goal)

- goal_token = the unified-vocab landmark token at the chosen goal cell.
  The model learns by convention that position 0 is the goal indicator.
- Explore phase: random walks for T_explore steps so the model can build a
  cognitive map of the environment.
- Navigate phase: BFS shortest path from the agent's position at the end of
  the explore phase to the goal cell. The agent then continues random walks
  if T_total > T_explore + BFS_distance (padding).

Training signal: action-prediction loss at the obs positions IMMEDIATELY
PRECEDING navigate-phase actions. I.e. mask is True at position 2*(E+i)
for i in [0, BFS_distance) — these positions predict the next action,
which is the BFS-optimal action toward the goal.

This tests whether the cognitive map supports goal-directed action
selection — the cleanest analog of animal goal-navigation experiments
that fits MapFormer's prediction-based training setup.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


def bfs_torus(start: tuple[int, int], goal: tuple[int, int], size: int) -> list[int]:
    """BFS shortest path on torus. Returns list of action ids (0=N,1=S,2=W,3=E).

    On a 64×64 torus, max shortest-path length is 32+32 = 64."""
    if start == goal:
        return []
    deltas = GridWorld.ACTION_DELTAS  # {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (x, y), path = queue.popleft()
        for a in range(4):
            dx, dy = deltas[a]
            nx = (x + dx) % size
            ny = (y + dy) % size
            if (nx, ny) == goal:
                return path + [a]
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [a]))
    return []


class GoalDirectedGridWorld(GridWorld):
    """GridWorld with goal-directed navigation episodes.

    Requires n_landmarks > 0 so we have well-defined goal cells (a goal in an
    aliased cell would be ambiguous). Goal cells are randomly sampled from
    the landmark cells per episode.
    """

    def __init__(
        self,
        size: int = 64,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                         n_landmarks=n_landmarks, seed=seed)
        assert n_landmarks > 0, "GoalDirectedGridWorld requires n_landmarks > 0"

    def generate_goal_episode(
        self,
        T_explore: int = 64,
        T_navigate: int = 64,
        goal_cell: Optional[tuple[int, int, int]] = None,
        start: Optional[tuple[int, int]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generate a goal-directed episode.

        Args:
            T_explore: random-walk steps before goal is revealed (builds the
                cognitive map).
            T_navigate: maximum BFS steps. If BFS distance < T_navigate, the
                remaining slots are filled with random walks AFTER reaching
                the goal (still recorded, but the action mask is False there).
            goal_cell: optional (x, y, landmark_idx) tuple to fix the goal.
                Default: random landmark per episode.
            start: optional starting position.
            rng: numpy RandomState for reproducibility.

        Returns:
            tokens: (1 + 2*(T_explore + T_navigate),) long tensor.
                Position 0 = goal token; rest is interleaved (a, o) pairs.
            obs_mask: same shape, True at obs positions.
            action_target_mask: same shape, True at positions whose NEXT TOKEN
                is a navigate-phase BFS-optimal action (i.e. the loss positions
                for goal-directed action prediction).
            info: dict with episode metadata (goal cell, bfs path, start pos, ...)
        """
        if rng is None:
            rng = np.random

        # Choose start and goal
        if start is None:
            start_x = int(rng.randint(0, self.size))
            start_y = int(rng.randint(0, self.size))
        else:
            start_x, start_y = start

        if goal_cell is None:
            gx, gy, g_idx = self.landmark_cells[int(rng.randint(0, len(self.landmark_cells)))]
        else:
            gx, gy, g_idx = goal_cell
        goal_token = self.first_landmark_unified + g_idx

        # Explore phase: random walk
        x, y = start_x, start_y
        tokens: list[int] = []
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size
            y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        # Navigate phase: BFS to goal
        bfs_path = bfs_torus((x, y), (gx, gy), self.size)

        # We always emit T_navigate steps; first len(bfs_path) are BFS actions,
        # the rest (after reaching goal) are random walks. Action target mask
        # is True only for the BFS portion.
        is_bfs_action = []
        for i in range(T_navigate):
            if i < len(bfs_path):
                a = bfs_path[i]
                is_bfs_action.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS))
                is_bfs_action.append(False)
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size
            y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        # Prepend goal token at position 0
        full = [goal_token] + tokens
        full = torch.tensor(full, dtype=torch.long)

        L = full.shape[0]
        obs_mask = torch.zeros(L, dtype=torch.bool)
        # After the goal token (offset 1), interleaving is (a, o, a, o, ...).
        # Action positions: 1, 3, 5, ... (odd >= 1)
        # Obs positions: 2, 4, 6, ... (even >= 2)
        obs_mask[2::2] = True

        # Action target mask: True at the OBS POSITION right before each
        # BFS-optimal action (predicting the NEXT TOKEN which is that action).
        # The first BFS action is at flat index 1 + 2*T_explore, predicted from
        # position 2*T_explore (the goal token if T_explore=0, else last
        # explore obs).
        action_target_mask = torch.zeros(L, dtype=torch.bool)
        for i in range(T_navigate):
            if is_bfs_action[i]:
                # The i-th navigate action sits at flat index 1 + 2*T_explore + 2*i.
                # The position predicting it is the previous token, at index 2*T_explore + 2*i.
                pred_pos = 2 * T_explore + 2 * i
                action_target_mask[pred_pos] = True

        info = {
            "start": (start_x, start_y),
            "goal": (gx, gy, g_idx),
            "goal_token": goal_token,
            "bfs_path": bfs_path,
            "bfs_distance": len(bfs_path),
            "T_explore": T_explore,
            "T_navigate": T_navigate,
        }
        return full, obs_mask, action_target_mask, info

    def generate_goal_batch(
        self, batch_size: int, T_explore: int = 64, T_navigate: int = 64,
        rng: Optional[np.random.RandomState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        toks, obs_masks, act_masks, infos = [], [], [], []
        for _ in range(batch_size):
            t, om, am, info = self.generate_goal_episode(T_explore, T_navigate, rng=rng)
            toks.append(t); obs_masks.append(om); act_masks.append(am); infos.append(info)
        return (torch.stack(toks), torch.stack(obs_masks), torch.stack(act_masks), infos)
