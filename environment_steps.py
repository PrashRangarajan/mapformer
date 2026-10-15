"""1D ring with two move tokens of configurable steps (GAUGE_PREREG.md). A copy of environment_cancel.GridWorldCancel's
walk (draw a move, repeat it k ~ U{1..10} times; move 0 with probability p_first) with move 0 advancing the position by
deltas[0] and move 1 by deltas[1] (mod size). deltas = (1, -1) reproduces GridWorldCancel exactly (checked); (1, 2) is a
ring walk in which both moves go forward, so revisits come only from laps and a map needs no cancellation: the signed
solution has both move angles of the same sign, which T1 (relative-sign gauge) puts inside the non-negative class."""
import numpy as np
import torch

from mapformer.environment_cancel import GridWorldCancel


class GridWorldSteps(GridWorldCancel):
    def __init__(self, size: int = 32, n_obs_types: int = 16, p_empty: float = 0.5, seed=None, p_first: float = 0.5,
                 deltas=(1, -1)):
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty, seed=seed, p_plus=p_first)
        self.deltas = tuple(int(d) for d in deltas)

    def generate_trajectory(self, n_steps: int = 128, start=None):
        pos = int(start[0]) if start is not None else int(np.random.randint(0, self.size))
        occ = np.random.random(self.size) >= self.p_empty
        obs = np.full(self.size, self.blank_token, dtype=np.int64)
        obs[occ] = np.random.randint(0, self.n_obs_types, int(occ.sum()))
        tokens, is_revisit, seen = [], [], set()
        self.visited_locations = []
        t = 0
        while t < n_steps:
            a = 0 if np.random.random() < self.p_plus else 1
            k = np.random.randint(1, 11)
            for _ in range(k):
                if t >= n_steps:
                    break
                pos = (pos + self.deltas[a]) % self.size
                tokens.append(a + self.action_offset)
                tokens.append(int(obs[pos]) + self.obs_offset)
                is_revisit.append(pos in seen)
                seen.add(pos)
                self.visited_locations.append((pos,))
                t += 1
        tok = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool); obs_mask[1::2] = True
        rev = torch.zeros(2 * n_steps, dtype=torch.bool)
        rev[1::2] = torch.tensor(is_revisit, dtype=torch.bool)
        return tok, obs_mask, rev
