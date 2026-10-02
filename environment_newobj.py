"""New-object transfer task (NEWOBJ_PREREG.md): the walk of GridWorldND (D=2), with object identities that
are NEW IN EVERY SEQUENCE and, at test time, drawn from a pool of objects never seen in training.

Token ids: 0..3 actions (GridWorldND's), 4 blank, 5.. objects; objects 5 .. 5+P-1 are the TRAIN pool,
5+P .. 5+2P-1 the TEST pool. Each sequence draws K objects from the active pool and a fresh map: every cell
the walk reaches is blank with probability p_empty, else one of the K objects, fixed for the sequence. So no
map and no object can be memorised; the observation at a revisited cell is knowable only from an earlier
visit in the same sequence. Revisit flags and the walk are GridWorldND's (they depend only on positions).
"""
import numpy as np
import torch

from mapformer.environment_nd import GridWorldND

N_SPECIAL = 5          # 4 actions + blank
BLANK = 4


class NewObjectWorld:
    def __init__(self, size=32, seed=None, pool_size=1000, k_per_seq=16, p_empty=0.5, pool="train"):
        self.grid = GridWorldND(dims=2, size=size, n_obs_types=16, p_empty=0.5, seed=seed)
        self.P, self.K, self.p_empty = pool_size, k_per_seq, p_empty
        self.pool = pool
        self.unified_vocab_size = N_SPECIAL + 2 * pool_size
        self.N_ACTIONS = 4
        self.visited_locations = []

    def _base(self):
        return N_SPECIAL + (0 if self.pool == "train" else self.P)

    def generate_trajectory(self, n_steps=1024):
        tok, om, rev = self.grid.generate_trajectory(n_steps)
        locs = list(self.grid.visited_locations)
        r = np.random
        objs = r.choice(self.P, self.K, replace=False) + self._base()
        cell = {}
        tok = tok.clone()
        for t, L in enumerate(locs):
            if L not in cell:
                cell[L] = BLANK if r.random() < self.p_empty else int(objs[r.randint(self.K)])
            tok[2 * t + 1] = cell[L]
        self.visited_locations = locs
        return tok, om, rev

    def generate_batch(self, batch_size, n_steps=1024, p_transition_noise=0.0):
        assert p_transition_noise == 0.0
        T, M, R, L = [], [], [], []
        for _ in range(batch_size):
            t, m, rv = self.generate_trajectory(n_steps)
            T.append(t); M.append(m); R.append(rv); L.append(list(self.visited_locations))
        return torch.stack(T), torch.stack(M), torch.stack(R), L
