"""D-dimensional torus grid world, for the rank threshold test.

WHY THIS EXISTS. MapFormer's phase factors through an r-dimensional accumulator:
because cumsum is linear,

    theta_t = omega * cumsum(W_out W_in x)_t = omega * W_out * sum_{u<=t} W_in x_u,

so all H*n_b phase channels are linear readouts of s_t in R^r (verified to 9e-07,
see mapformer_math.tex sec. 5). On a D-dimensional lattice with actions +/- e_i the
state after a walk is s = sum_a n_a v_a with v_a in R^r, so the induced map
Z^D -> R^r has a KERNEL of dimension D - r whenever r < D. Positions differing by
any vector in that kernel carry identical phase in every block. That is an
impossibility, not a conditioning problem, and it predicts the paper's own
unexplained 5D collapse (v4 Table 6: MapWM 0.75/0.50/0.35, below CoPE) IF r=2 was
held fixed there.

This module supplies the D axis. It is a SEPARATE class rather than a patch to
environment.py, which ~40 experiments depend on and which is 2D throughout.

WHAT IS MIRRORED FROM GridWorld, exactly:
  - the DIRECTED walk: draw an action, repeat it k ~ U{1..10} times. Not an iid
    walk. This is what sets the revisit statistics, and every 2D number in this
    project is on it.
  - interleaved stream [a1, o1, a2, o2, ...], torus wrap, allocentric observation
    of the OCCUPIED cell, p_empty blanking, revisit keyed on the observed cell.
  - the same (tokens, obs_mask, revisit_mask) contract and generate_batch shape,
    so train.py needs no change.

WHAT IS DROPPED, deliberately: landmarks, rotate/ego modes, wall boundaries,
allocentric re-recording, transition noise, score_moves_only. None of them bear on
the rank question and each is a way for the comparison to differ from the 2D
anchor by something other than D.

RECURRENCE CAVEAT, which the gate must check rather than assume. A simple random
walk on Z^D is transient for D >= 3 (Polya), so revisits are guaranteed here only
because the torus is FINITE and small. The revisit-masked loss has signal only if
revisits actually occur; validate_nd.py measures the rate and refuses a config
where it collapses.
"""
from typing import Optional

import numpy as np
import torch


class GridWorldND:
    """D-dimensional torus. Actions: 2*D of them, a = 2i -> +e_i, a = 2i+1 -> -e_i."""

    def __init__(
        self,
        dims: int = 2,
        size: int = 32,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        seed: Optional[int] = None,
    ):
        assert dims >= 1, dims
        assert size >= 2, size
        self.dims = dims
        self.size = size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty

        # Unified vocabulary:
        #   [0 .. 2D-1]        actions
        #   [2D .. 2D+K-1]     K regular observation types
        #   [2D+K]             blank
        self.n_actions = 2 * dims
        self.action_offset = 0
        self.obs_offset = self.n_actions
        self.blank_token = n_obs_types
        self.unified_blank = self.obs_offset + self.blank_token
        self.obs_vocab_size = n_obs_types + 1
        self.unified_vocab_size = self.n_actions + self.obs_vocab_size
        # kept so downstream code that inspects landmarks does not need a branch
        self.n_landmarks = 0
        self.landmark_cells: list = []

        rng = np.random.RandomState(seed)
        shape = (size,) * dims
        obs_map = np.full(shape, self.blank_token, dtype=np.int64)
        occupied = rng.random(shape) >= p_empty
        obs_map[occupied] = rng.randint(0, n_obs_types, int(occupied.sum()))
        self.obs_map = torch.from_numpy(obs_map).long()

        # +e_i for even a, -e_i for odd a
        deltas = np.zeros((self.n_actions, dims), dtype=np.int64)
        for i in range(dims):
            deltas[2 * i, i] = 1
            deltas[2 * i + 1, i] = -1
        self.action_deltas = deltas

        self.visited_locations: list[tuple] = []

    @property
    def n_cells(self) -> int:
        return self.size ** self.dims

    def generate_trajectory(self, n_steps: int = 128, start=None):
        """Returns (tokens, obs_mask, revisit_mask), each of length 2*n_steps."""
        if start is not None:
            pos = np.array(start, dtype=np.int64)
        else:
            pos = np.random.randint(0, self.size, size=self.dims)

        tokens: list[int] = []
        is_revisit: list[bool] = []
        seen: set = set()
        self.visited_locations = []

        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.n_actions)
            k = np.random.randint(1, 11)          # directed run, as in GridWorld
            for _ in range(k):
                if t >= n_steps:
                    break
                pos = (pos + self.action_deltas[a]) % self.size
                tokens.append(a + self.action_offset)
                key = tuple(int(v) for v in pos)
                tokens.append(int(self.obs_map[key].item()) + self.obs_offset)
                is_revisit.append(key in seen)
                seen.add(key)
                self.visited_locations.append(key)
                t += 1

        tok = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        obs_mask[1::2] = True
        rev = torch.zeros(2 * n_steps, dtype=torch.bool)
        rev[1::2] = torch.tensor(is_revisit, dtype=torch.bool)
        return tok, obs_mask, rev

    def generate_batch(self, batch_size: int, n_steps: int = 128,
                       p_transition_noise: float = 0.0):
        assert p_transition_noise == 0.0, \
            "GridWorldND is the clean rank-test environment; noise lives in GridWorld"
        toks, masks, revs, locs = [], [], [], []
        for _ in range(batch_size):
            a, b, c = self.generate_trajectory(n_steps)
            toks.append(a); masks.append(b); revs.append(c)
            locs.append(list(self.visited_locations))
        return torch.stack(toks), torch.stack(masks), torch.stack(revs), locs
