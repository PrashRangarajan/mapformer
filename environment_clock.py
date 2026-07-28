"""Modular-clock navigation — a symbolic/language analog of the hier-goal task.

Why: MapFormer's position code is ω·cumsum(Δ) — modular addition, i.e. SO(2).
A clock is a product of circles (hands) at nested moduli, so a clock task lands
exactly on MapFormer's native structure, in a non-spatial (symbolic) domain.
This tests whether the MapFormer × hierarchy synergy transfers out of the grid.

State v ∈ [0, P), P = radix_slow · radix_fast (a mixed-radix "time"):
  fast hand (fine) = v mod radix_fast ;  slow hand (coarse) = v // radix_fast.
Actions are ±1 / ±2 ticks on the value (the fast hand, carrying into the slow).

Episode:  [goal_slow, goal_fast, explore·T_e, navigate·T_n]
- Fixed start v=0 → integrating the tick stream yields the ABSOLUTE time.
- Goal = a target time, given HAND BY HAND: slow (coarse) then fast (fine) —
  the hierarchical, two-scale goal, mirroring (room, local).
- Explore = random ±ticks; forces path integration to know the current time.
- Navigate = greedy-optimal ticks toward the goal on the ring; loss = next-action
  CE (chance 0.25, oracle 1.00). obs is aliased + redrawn per episode so the time
  must come from integrating ticks, not from reading obs.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch


class ModularClockWorld:
    N_ACTIONS = 4
    DV = [-2, -1, 1, 2]            # action index → tick delta

    def __init__(self, radix_fast: int = 16, radix_slow: int = 8,
                 n_obs_types: int = 16, p_empty: float = 0.5, seed: Optional[int] = None):
        self.rf, self.rs = radix_fast, radix_slow
        self.P = radix_fast * radix_slow
        self.n_obs = n_obs_types
        self.p_empty = p_empty
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS               # = 4
        self.blank = n_obs_types
        base = self.N_ACTIONS + n_obs_types + 1        # actions + obs + blank
        self.slow_tok0 = base
        self.fast_tok0 = base + self.rs
        self.unified_vocab_size = base + self.rs + self.rf
        self._rng = np.random.RandomState(seed)

    def _draw_obs(self, rng):
        obs = np.full(self.P, self.blank, dtype=np.int64)
        occ = rng.random(self.P) >= self.p_empty
        obs[occ] = rng.randint(0, self.n_obs, int(occ.sum()))
        return obs

    def hands(self, v):                                # (slow/coarse, fast/fine)
        return v // self.rf, v % self.rf

    def _greedy(self, v, g):                           # optimal signed tick, or None
        d = ((g - v + self.P // 2) % self.P) - self.P // 2
        if d == 0:
            return None
        return (1 if d > 0 else -1) * min(2, abs(d))

    def generate_clock_episode(self, T_explore=64, T_navigate=64, rng=None,
                               goal=None, start=0):
        if rng is None:
            rng = np.random
        obs = self._draw_obs(rng)
        g = int(rng.randint(0, self.P)) if goal is None else int(goal)
        slow, fast = self.hands(g)
        v = int(start)
        tokens: list[int] = []
        for _ in range(T_explore):                     # explore: random ticks
            dv = self.DV[int(rng.randint(0, self.N_ACTIONS))]
            v = (v + dv) % self.P
            tokens.append(self.DV.index(dv) + self.action_offset)
            tokens.append(int(obs[v]) + self.obs_offset)
        is_opt: list[bool] = []
        for i in range(T_navigate):                    # navigate: greedy to goal
            dv = self._greedy(v, g)
            if dv is None:
                dv = self.DV[int(rng.randint(0, self.N_ACTIONS))]; is_opt.append(False)
            else:
                is_opt.append(True)
            v = (v + dv) % self.P
            tokens.append(self.DV.index(dv) + self.action_offset)
            tokens.append(int(obs[v]) + self.obs_offset)

        full = torch.tensor([self.slow_tok0 + slow, self.fast_tok0 + fast] + tokens,
                            dtype=torch.long)
        L = full.shape[0]
        obs_mask = torch.zeros(L, dtype=torch.bool); obs_mask[3::2] = True
        act_mask = torch.zeros(L, dtype=torch.bool)
        for i, opt in enumerate(is_opt):
            if opt:
                act_mask[1 + 2 * T_explore + 2 * i] = True
        info = {"goal": g, "hands": (int(slow), int(fast)), "start": int(start),
                "T_explore": T_explore, "T_navigate": T_navigate}
        return full, obs_mask, act_mask, info

    def generate_clock_batch(self, batch_size, T_explore=64, T_navigate=64, rng=None):
        toks, oms, ams, infos = [], [], [], []
        for _ in range(batch_size):
            t, om, am, info = self.generate_clock_episode(T_explore, T_navigate, rng=rng)
            toks.append(t); oms.append(om); ams.append(am); infos.append(info)
        return torch.stack(toks), torch.stack(oms), torch.stack(ams), infos
