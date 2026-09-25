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

    # --- Task A: multi-stop (visit goals in sequence) -----------------------
    def generate_multistop_episode(
        self, T_explore: int = 64, T_per_segment: int = 32, n_stops: int = 2,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Episode: [goal_1, ..., goal_K, explore, BFS(start→g1)+BFS(g1→g2)+...].

        All goals announced up front (positions 0..K-1). Model must remember
        the sequence and navigate to each in turn. T_per_segment is the
        navigate-budget for each leg (BFS path padded with random walks
        after reaching that subgoal)."""
        if rng is None: rng = np.random

        # Choose K distinct landmark goals
        idxs = rng.choice(len(self.landmark_cells), size=n_stops, replace=False)
        goals = [self.landmark_cells[i] for i in idxs]
        goal_tokens = [self.first_landmark_unified + g[2] for g in goals]

        # Start
        x = int(rng.randint(0, self.size)); y = int(rng.randint(0, self.size))

        tokens: list[int] = []
        # Explore phase
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        # Navigate phases (one segment per goal)
        is_bfs_per_step: list[bool] = []
        seg_starts_in_navigate: list[int] = []  # index of segment-start in navigate-token-stream
        for (gx, gy, _) in goals:
            seg_starts_in_navigate.append(len(is_bfs_per_step))
            bfs_path = bfs_torus((x, y), (gx, gy), self.size)
            for i in range(T_per_segment):
                if i < len(bfs_path):
                    a = bfs_path[i]; is_bfs_per_step.append(True)
                else:
                    a = int(rng.randint(0, self.N_ACTIONS)); is_bfs_per_step.append(False)
                dx, dy = self.ACTION_DELTAS[a]
                x = (x + dx) % self.size; y = (y + dy) % self.size
                tokens.append(a + self.action_offset)
                tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        full = goal_tokens + tokens
        full = torch.tensor(full, dtype=torch.long)
        L = full.shape[0]

        obs_mask = torch.zeros(L, dtype=torch.bool)
        # After the K goal tokens, interleaving starts: (a, o, a, o, ...).
        # Action positions: K, K+2, ..., Obs positions: K+1, K+3, ...
        obs_mask[(n_stops + 1)::2] = True

        action_target_mask = torch.zeros(L, dtype=torch.bool)
        nav_start_token_idx = n_stops + 2 * T_explore  # first navigate-action position
        for i, was_bfs in enumerate(is_bfs_per_step):
            if was_bfs:
                pred_pos = nav_start_token_idx + 2 * i - 1
                if pred_pos >= 0: action_target_mask[pred_pos] = True

        info = {
            "task": "multistop", "n_stops": n_stops,
            "goals": [(g[0], g[1], g[2]) for g in goals],
            "T_explore": T_explore, "T_per_segment": T_per_segment,
        }
        return full, obs_mask, action_target_mask, info

    # --- Task B: goal switching mid-episode --------------------------------
    def generate_switching_episode(
        self, T_explore: int = 64, T_pre_switch: int = 32, T_post_switch: int = 32,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Episode: [goal_A, explore, navigate_toward_A (T_pre), goal_B, navigate_toward_B (T_post)].

        Model starts heading to A, then mid-episode is told to switch to B.
        Tests on-the-fly replanning when goal changes."""
        if rng is None: rng = np.random

        idx_a, idx_b = rng.choice(len(self.landmark_cells), size=2, replace=False)
        ga = self.landmark_cells[idx_a]; gb = self.landmark_cells[idx_b]
        tok_a = self.first_landmark_unified + ga[2]
        tok_b = self.first_landmark_unified + gb[2]

        x = int(rng.randint(0, self.size)); y = int(rng.randint(0, self.size))
        tokens: list[int] = []

        # Explore
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        # Pre-switch navigate (BFS toward A)
        pre_is_bfs: list[bool] = []
        bfs_a = bfs_torus((x, y), (ga[0], ga[1]), self.size)
        for i in range(T_pre_switch):
            if i < len(bfs_a):
                a = bfs_a[i]; pre_is_bfs.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS)); pre_is_bfs.append(False)
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        # Insert switch token (goal B)
        tokens.append(tok_b)

        # Post-switch navigate (BFS toward B from current pos)
        post_is_bfs: list[bool] = []
        bfs_b = bfs_torus((x, y), (gb[0], gb[1]), self.size)
        for i in range(T_post_switch):
            if i < len(bfs_b):
                a = bfs_b[i]; post_is_bfs.append(True)
            else:
                a = int(rng.randint(0, self.N_ACTIONS)); post_is_bfs.append(False)
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        full = [tok_a] + tokens
        full = torch.tensor(full, dtype=torch.long)
        L = full.shape[0]

        obs_mask = torch.zeros(L, dtype=torch.bool)
        action_target_mask = torch.zeros(L, dtype=torch.bool)

        # Layout:
        # pos 0           : goal_A
        # pos 1..2T_e     : explore (a,o,a,o,...)
        # pos 2T_e+1..    : pre-switch nav (a,o)*T_pre
        # pos after that  : tok_b (switch token)
        # pos after that  : post-switch nav (a,o)*T_post
        pre_nav_start = 1 + 2 * T_explore  # first pre-nav action
        for i, was_bfs in enumerate(pre_is_bfs):
            pred_pos = pre_nav_start + 2 * i - 1  # obs position before this action
            if was_bfs and pred_pos >= 0: action_target_mask[pred_pos] = True

        switch_pos = pre_nav_start + 2 * T_pre_switch  # tok_b sits here
        post_nav_start = switch_pos + 1
        for i, was_bfs in enumerate(post_is_bfs):
            pred_pos = post_nav_start + 2 * i - 1
            if was_bfs: action_target_mask[pred_pos] = True

        # Obs positions: handle the interruption at switch_pos
        for p in range(L):
            if p == 0 or p == switch_pos: continue  # goal tokens
            # Without the switch insertion, obs positions are even after pos 0.
            # With the insertion at switch_pos, post-switch positions get shifted.
            if p < switch_pos:
                if p >= 2 and p % 2 == 0: obs_mask[p] = True
            else:
                rel = p - switch_pos
                if rel >= 2 and rel % 2 == 0: obs_mask[p] = True

        info = {
            "task": "switching",
            "goal_a": (ga[0], ga[1], ga[2]),
            "goal_b": (gb[0], gb[1], gb[2]),
            "switch_pos": switch_pos,
            "T_explore": T_explore,
            "T_pre_switch": T_pre_switch,
            "T_post_switch": T_post_switch,
        }
        return full, obs_mask, action_target_mask, info

    # --- Task C: closed-loop navigate under transition noise ---------------
    def generate_noisy_episode(
        self, T_explore: int = 64, T_navigate: int = 64,
        p_transition_noise: float = 0.1,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Episode: standard goal-directed, but during navigate the executed
        action differs from the commanded BFS action with probability
        p_transition_noise. BFS is RECOMPUTED from actual position after
        each step (closed-loop oracle). Loss target = current-step BFS-
        optimal action from actual position."""
        if rng is None: rng = np.random

        gx, gy, g_idx = self.landmark_cells[int(rng.randint(0, len(self.landmark_cells)))]
        goal_token = self.first_landmark_unified + g_idx

        x = int(rng.randint(0, self.size)); y = int(rng.randint(0, self.size))
        tokens: list[int] = []
        for _ in range(T_explore):
            a = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        is_bfs: list[bool] = []
        for i in range(T_navigate):
            if (x, y) == (gx, gy):
                a_cmd = int(rng.randint(0, self.N_ACTIONS))
                is_bfs.append(False)
            else:
                bfs_path = bfs_torus((x, y), (gx, gy), self.size)
                a_cmd = bfs_path[0] if bfs_path else int(rng.randint(0, self.N_ACTIONS))
                is_bfs.append(bool(bfs_path))
            # Execute (possibly noisy)
            a_exec = a_cmd
            if rng.random() < p_transition_noise:
                a_exec = int(rng.randint(0, self.N_ACTIONS))
            dx, dy = self.ACTION_DELTAS[a_exec]
            x = (x + dx) % self.size; y = (y + dy) % self.size
            tokens.append(a_cmd + self.action_offset)  # RECORD the commanded
            tokens.append(int(self.obs_map[x, y].item()) + self.obs_offset)

        full = torch.tensor([goal_token] + tokens, dtype=torch.long)
        L = full.shape[0]
        obs_mask = torch.zeros(L, dtype=torch.bool); obs_mask[2::2] = True
        action_target_mask = torch.zeros(L, dtype=torch.bool)
        for i, was_bfs in enumerate(is_bfs):
            if was_bfs:
                pred_pos = 2 * T_explore + 2 * i
                action_target_mask[pred_pos] = True

        info = {
            "task": "noisy", "goal": (gx, gy, g_idx),
            "T_explore": T_explore, "T_navigate": T_navigate,
            "p_transition_noise": p_transition_noise,
        }
        return full, obs_mask, action_target_mask, info
