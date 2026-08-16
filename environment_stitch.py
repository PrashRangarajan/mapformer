"""Transitive inference: stitching disjoint room experiences. Ported from CSCG.

Source (George et al., bioRxiv 10.1101/864421v4), verbatim:

    "we randomly generate two square rooms of size 8 x 6 with 15 different
     observations each. We make both rooms share a 3 x 3 patch in their corners...
     We sample a random walk of length 10000 of action-observation pairs on each
     room, always avoiding to take actions that would make the random walk move
     outside of the room."

    "Notice that there is another patch in the first room that is identical to the
     merged patches, but was not merged. The model is using the sequential
     information to effectively identify patches that can be merged while
     respecting the observational data and context, and not simply looking for
     locally identical patches to merge."

That last sentence is the measurable core, and it is why this task is worth
running: it ships its own NEGATIVE CONTROL. A model that merges on local
appearance alone will merge the confounding patch too and be wrong.

Deviation from CSCG, stated
---------------------------
CSCG runs EM over two 10,000-step sequences. MapFormer builds maps IN CONTEXT, so
one episode contains both room walks and then a test phase, and the room layouts
are redrawn every episode.

Episode
-------
    phase A   random walk in room A, observations REVEALED
    phase B   random walk in room B, observations REVEALED
    phase T   walk starting INSIDE a 3x3 patch, observations WITHHELD (MASK);
              predict them

Phase T starts either in the SHARED patch (so continuing onward leads into room
B's cells) or in the CONFOUNDING patch (identical observations, but it sits
elsewhere in room A, so continuing leads into room A's cells). The two cases are
locally indistinguishable and have different correct answers. Scored at cells
already seen in the relevant room, deduped per cell.

Rooms are BOUNDED (walls), not toroidal, matching "always avoiding to take
actions that would make the random walk move outside of the room".
"""
import numpy as np
import torch

ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class StitchWorld:
    """Two rooms sharing a 3x3 corner patch, plus a confounding identical patch."""

    N_ACTIONS = 4

    def __init__(self, h: int = 8, w: int = 6, n_obs_types: int = 15,
                 patch: int = 3, seed: int = 0):
        self.h, self.w, self.n_obs_types, self.patch = h, w, n_obs_types, patch
        self.seed = seed
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.mask_tok = self.N_ACTIONS + n_obs_types
        self.unified_vocab_size = self.mask_tok + 1

    def _draw_rooms(self, rng):
        """Room A, room B, the shared patch corner, and a confounding patch in A."""
        P = self.patch
        A = rng.randint(0, self.n_obs_types, size=(self.h, self.w))
        B = rng.randint(0, self.n_obs_types, size=(self.h, self.w))
        shared = rng.randint(0, self.n_obs_types, size=(P, P))
        # shared patch sits in A's bottom-right corner and B's top-left corner
        A[self.h - P:, self.w - P:] = shared
        B[:P, :P] = shared
        # CONFOUNDER: an identical copy elsewhere in room A (top-left), which must
        # NOT be treated as the join. This is the paper's negative control.
        A[:P, :P] = shared
        return A, B

    def _walk(self, rng, grid, x, y, n):
        """Bounded random walk; invalid moves are resampled, never recorded."""
        for _ in range(n):
            valid = [a for a, (dx, dy) in ACTION_DELTAS.items()
                     if 0 <= x + dx < self.h and 0 <= y + dy < self.w]
            a = int(valid[rng.randint(len(valid))])
            dx, dy = ACTION_DELTAS[a]
            x, y = x + dx, y + dy
            yield a, x, y

    def generate_episode(self, T_a: int = 96, T_b: int = 96, T_test: int = 24,
                         rng=None, force_confound=None):
        if rng is None:
            rng = np.random
        P = self.h, self.w
        A, B = self._draw_rooms(rng)
        confound = bool(rng.randint(2)) if force_confound is None else force_confound

        tokens, revisit = [], []
        seenA, seenB = set(), set()

        x, y = int(rng.randint(self.h)), int(rng.randint(self.w))
        for a, x, y in self._walk(rng, A, x, y, T_a):
            tokens += [a + self.action_offset, int(A[x, y]) + self.obs_offset]
            revisit += [False, False]; seenA.add((x, y))
        x, y = int(rng.randint(self.h)), int(rng.randint(self.w))
        for a, x, y in self._walk(rng, B, x, y, T_b):
            tokens += [a + self.action_offset, int(B[x, y]) + self.obs_offset]
            revisit += [False, False]; seenB.add((x, y))

        # phase T: start inside one of the two identical patches
        pp = self.patch
        if confound:                       # confounding patch, top-left of room A
            x, y = int(rng.randint(pp)), int(rng.randint(pp))
            grid, seen = A, seenA
        else:                              # the true join: A's bottom-right corner
            x, y = self.h - pp + int(rng.randint(pp)), self.w - pp + int(rng.randint(pp))
            grid, seen = A, seenA
        score_pos, answers, scored = [], [], set()
        for a, x, y in self._walk(rng, grid, x, y, T_test):
            sp = len(tokens)
            tokens += [a + self.action_offset, self.mask_tok]
            revisit += [False, False]
            if (x, y) in seen and (x, y) not in scored:
                scored.add((x, y)); score_pos.append(sp)
                answers.append(int(grid[x, y]) + self.obs_offset)

        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(revisit, dtype=torch.bool),
                score_pos, answers,
                {"confound": confound, "n_scored": len(score_pos)})

    def generate_batch(self, batch_size: int, T_a=96, T_b=96, T_test=24, rng=None):
        eps = [self.generate_episode(T_a, T_b, T_test, rng) for _ in range(batch_size)]
        n = max(e[0].shape[0] for e in eps)
        toks = torch.full((batch_size, n), self.mask_tok, dtype=torch.long)
        rev = torch.zeros(batch_size, n, dtype=torch.bool)
        for i, e in enumerate(eps):
            toks[i, :e[0].shape[0]] = e[0]; rev[i, :e[1].shape[0]] = e[1]
        return toks, rev, [e[2] for e in eps], [e[3] for e in eps], [e[4] for e in eps]
