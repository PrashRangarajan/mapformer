"""
Vectorised interactive maze for SPARSE-REWARD RL.

Why RL and not the earlier behaviour cloning. Options/feudal HRL earns its
reputation on sparse-reward long-horizon credit assignment: when reward arrives
only at the goal, a flat policy cannot tell which of ~60 actions mattered.
Our previous maze experiments gave DENSE per-step BFS supervision, which
removes that problem by construction -- so they could never test HRL's actual
claim. Here reward is +1 at the goal and 0 everywhere else.

Validated before building (measured, budget=64, 400 episodes):
  random policy success 11.7%  -> ~7.5 rewarded episodes per 64-episode batch
  BFS-optimal length mean 12.9 -> an optimal policy succeeds ~100% within 64
So the signal is sparse but non-zero: the flat baseline gets *some* gradient,
which keeps the comparison meaningful rather than "hierarchy learns, flat gets
literally nothing".

Episode structure
  context : [goal_token, explore tokens ...]   explore is an offline random walk
            so the agent has a map to work from; the goal is always a landmark
            it actually OBSERVED during exploration.
  acting  : the policy emits an action; env appends [action_token, obs_token]
  reward  : +1 on reaching the goal cell, 0 otherwise; done on goal or budget.

Bump tokens are ON by default. In an embodied RL setting collision feedback is
not a hack -- it is ordinary observability, and without it MapFormer's
cumsum(action) path integration desynchronises on 28% of moves.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from .environment_maze_varying import VaryingMazeWorld


class MazeRLEnv:
    def __init__(self, n_envs: int = 64, size: int = 12, rooms_per_side: int = 4,
                 n_obs_types: int = 8, n_landmarks: int = 24,
                 T_explore: int = 256, max_steps: int = 64,
                 seed: Optional[int] = None):
        self.n = n_envs
        self.T_explore = T_explore
        self.max_steps = max_steps
        self.world = VaryingMazeWorld(size=size, rooms_per_side=rooms_per_side,
                                      n_obs_types=n_obs_types,
                                      n_landmarks=n_landmarks,
                                      bump_tokens=True, seed=seed)
        self.size = size
        self.rs = self.world.rs
        self.R = self.world.R
        self.vocab_size = self.world.unified_vocab_size
        self.n_actions = self.world.N_ACTIONS
        self.rng = np.random.RandomState(seed)

    def _obs_token(self, e, x, y, blocked, a):
        if blocked:
            return self.world.first_bump + a
        if (x, y) in self.lm[e]:
            return self.world.first_landmark + self.lm[e][(x, y)]
        return int(self.obs[e][x, y]) + self.world.obs_offset

    def reset(self):
        """Returns context tokens (n_envs, 1 + 2*T_explore)."""
        self.dv, self.dh, self.obs, self.lm = [], [], [], []
        self.pos, self.goal, self.done, self.t = [], [], [], 0
        ctx = []
        for e in range(self.n):
            while True:
                dv, dh = self.world._sample_maze(self.rng)
                ob = self.rng.randint(0, self.world.n_obs_types,
                                      size=(self.size, self.size))
                flat = self.rng.permutation(self.size * self.size)[:self.world.n_landmarks]
                lm = {(int(f // self.size), int(f % self.size)): i
                      for i, f in enumerate(flat)}
                x, y = int(self.rng.randint(0, self.size)), int(self.rng.randint(0, self.size))
                toks, seen = [], {}
                for _ in range(self.T_explore):
                    a = int(self.rng.randint(0, self.n_actions))
                    px, py = x, y
                    if self.world._can_move(dv, dh, x, y, a):
                        ddx, ddy = self.world.ACTION_DELTAS[a]
                        x, y = (x + ddx) % self.size, (y + ddy) % self.size
                    toks.append(a)
                    blocked = (x, y) == (px, py)
                    if blocked:
                        toks.append(self.world.first_bump + a)
                    elif (x, y) in lm:
                        seen[lm[(x, y)]] = (x, y)
                        toks.append(self.world.first_landmark + lm[(x, y)])
                    else:
                        toks.append(int(ob[x, y]) + self.world.obs_offset)
                if seen:
                    break
            gi = int(self.rng.choice(list(seen.keys())))
            self.dv.append(dv); self.dh.append(dh); self.obs.append(ob); self.lm.append(lm)
            self.pos.append((x, y)); self.goal.append(seen[gi]); self.done.append(False)
            ctx.append([self.world.first_landmark + gi] + toks)
        self.t = 0
        return torch.tensor(ctx, dtype=torch.long)

    def step(self, actions):
        """actions (n_envs,) long. Returns (act_tok, obs_tok, reward, done)."""
        a_t = np.asarray(actions).astype(int)
        obs_tok = np.zeros(self.n, dtype=np.int64)
        rew = np.zeros(self.n, dtype=np.float32)
        for e in range(self.n):
            if self.done[e]:
                obs_tok[e] = self.world.unified_blank if hasattr(self.world, "unified_blank") \
                    else self.world.obs_offset
                continue
            a = int(a_t[e]); x, y = self.pos[e]
            px, py = x, y
            if self.world._can_move(self.dv[e], self.dh[e], x, y, a):
                ddx, ddy = self.world.ACTION_DELTAS[a]
                x, y = (x + ddx) % self.size, (y + ddy) % self.size
            self.pos[e] = (x, y)
            obs_tok[e] = self._obs_token(e, x, y, (x, y) == (px, py), a)
            if (x, y) == self.goal[e]:
                rew[e] = 1.0
                self.done[e] = True
        self.t += 1
        if self.t >= self.max_steps:
            self.done = [True] * self.n
        return (torch.from_numpy(a_t), torch.from_numpy(obs_tok),
                torch.from_numpy(rew), torch.tensor(self.done))

    def region_of(self, e):
        x, y = self.pos[e]
        return (x // self.rs) * self.R + (y // self.rs)

    def goal_region(self, e):
        gx, gy = self.goal[e]
        return (gx // self.rs) * self.R + (gy // self.rs)

    def optimal_len(self, e):
        return len(self.world._bfs(self.dv[e], self.dh[e], self.pos[e], self.goal[e]))
