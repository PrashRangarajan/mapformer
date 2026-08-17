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
B's cells) or in the CONFOUNDING patch (identical observations, but it sits in
A's INTERIOR, so continuing stays inside A). Both patches have identical local
geometry -- 4 valid actions at every cell -- so the two conditions cannot be
separated from the action stream. `validate_cscg_tasks.py` measures this
(condition-identifiability gate); an earlier version placed the confounder at a
corner and was separable at 0.762 balanced accuracy without any map.

Scored at cells already seen, deduped per cell.

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

    def _build(self, rng):
        """Stitched coordinate space: A at (0,0), B offset so B's top-left P x P
        COINCIDES with A's bottom-right P x P. Walking out of that overlap really
        does enter B's cells -- which is the whole point, and which an earlier
        version of this file got wrong by keeping both phases inside room A."""
        P, h, w = self.patch, self.h, self.w
        H, W = 2 * h - P, 2 * w - P
        grid = np.full((H, W), -1, dtype=np.int64)          # -1 = wall
        A = rng.randint(0, self.n_obs_types, size=(h, w))
        B = rng.randint(0, self.n_obs_types, size=(h, w))
        shared = rng.randint(0, self.n_obs_types, size=(P, P))
        A[h - P:, w - P:] = shared
        B[:P, :P] = shared
        # CONFOUNDER: identical patch in A's INTERIOR (rows 1..3, cols 1..3).
        # It must have the SAME local geometry as the join, or the two conditions
        # are separable without any map. Placing it at A's top-left corner (an
        # earlier version) gave valid-action counts [2,3,3,3,4,4,3,4,4] against the
        # join's [4,4,4,4,4,4,4,4,4], and the cue "did I move >2 cells up or left
        # of where I started" -- computable from the TEST-PHASE ACTION STREAM
        # ALONE -- fired on 52.5% of shared episodes and 0.0% of confound ones:
        # balanced accuracy 0.762 against a 0.500 chance, via relative
        # displacement, i.e. exactly the mechanism under test.
        self._cf_origin = (1, 1)
        A[1:1 + P, 1:1 + P] = shared
        grid[:h, :w] = A
        grid[h - P:, w - P:] = B
        regA = {(i, j) for i in range(h) for j in range(w)}
        regB = {(i + h - P, j + w - P) for i in range(h) for j in range(w)}
        return grid, regA, regB, (h - P, w - P)

    def _walk(self, rng, region, x, y, n):
        """Random walk confined to `region`; invalid moves resampled, not recorded."""
        for _ in range(n):
            valid = [a for a, (dx, dy) in ACTION_DELTAS.items()
                     if (x + dx, y + dy) in region]
            if not valid:
                return
            a = int(valid[rng.randint(len(valid))])
            dx, dy = ACTION_DELTAS[a]
            x, y = x + dx, y + dy
            yield a, x, y

    def generate_episode(self, T_a: int = 96, T_b: int = 96, T_test: int = 24,
                         rng=None, force_confound=None):
        if rng is None:
            rng = np.random
        grid, regA, regB, (oy, ox) = self._build(rng)
        confound = bool(rng.randint(2)) if force_confound is None else force_confound
        P = self.patch
        tokens, revisit, seen = [], [], set()

        def phase(region, n):
            cells = sorted(region)
            x, y = cells[rng.randint(len(cells))]
            for a, x, y in self._walk(rng, region, x, y, n):
                tokens.extend([a + self.action_offset, int(grid[x, y]) + self.obs_offset])
                revisit.extend([False, (x, y) in seen]); seen.add((x, y))

        phase(regA, T_a)          # room A alone
        phase(regB, T_b)          # room B alone -- DISJOINT experiences

        # phase T: start in one of the two identical patches, walk the FULL space
        if confound:              # A's interior copy; walking out stays in A
            cy, cx = self._cf_origin
            x, y = cy + int(rng.randint(P)), cx + int(rng.randint(P))
        else:                     # the true join; walking out enters B
            x, y = oy + int(rng.randint(P)), ox + int(rng.randint(P))
        both = regA | regB
        score_pos, answers, scored = [], [], set()
        for a, x, y in self._walk(rng, both, x, y, T_test):
            sp = len(tokens)
            tokens.extend([a + self.action_offset, self.mask_tok])
            revisit.extend([False, False])
            if (x, y) in seen and (x, y) not in scored:
                scored.add((x, y)); score_pos.append(sp)
                answers.append(int(grid[x, y]) + self.obs_offset)

        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(revisit, dtype=torch.bool), score_pos, answers,
                {"confound": confound, "n_scored": len(score_pos)})

    # ------------------------------------------------------------------
    # Training episodes for the ATTENTION-PROBE evaluation.
    #
    # These carry NO blind test phase and NO scored answers. The objective is
    # just MapFormer's own -- predict the observation at revisited cells --
    # which is the analogue of CSCG running EM over the two room walks. The
    # evaluation then INSPECTS the trained model (probe_stitch_attention.py),
    # exactly as CSCG inspects its learned transition matrix rather than
    # scoring a held-out accuracy. CSCG reports this experiment qualitatively
    # ("Predictive performance on the stitching of the two rooms is perfect"),
    # with no accuracy table, no baseline and no chance rate, so there is no
    # scored metric of theirs to port.
    #
    # Phase T (a walk over the UNION of both rooms) is what creates pressure to
    # stitch: cells whose only earlier visit was in the other room's walk are
    # only predictable if the two experiences have been joined.
    # ------------------------------------------------------------------
    def generate_train_episode(self, T_a: int = 256, T_b: int = 256,
                               T_t: int = 128, rng=None):
        if rng is None:
            rng = np.random
        grid, regA, regB, _join = self._build(rng)
        tokens, revisit, seen = [], [], set()

        def run(region, start, n):
            x, y = start
            for a, x, y in self._walk(rng, region, x, y, n):
                tokens.extend([a + self.action_offset,
                               int(grid[x, y]) + self.obs_offset])
                revisit.extend([False, (x, y) in seen])
                seen.add((x, y))
            return x, y

        cellsA = sorted(regA)
        pos = cellsA[rng.randint(len(cellsA))]
        pos = run(regA, pos, T_a)                 # room A alone
        cellsB = sorted(regB)
        pos = cellsB[rng.randint(len(cellsB))]
        pos = run(regB, pos, T_b)                 # room B alone -- DISJOINT
        run(regA | regB, pos, T_t)                # the union: needs the join

        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(revisit, dtype=torch.bool))

    def generate_train_batch(self, batch_size: int, T_a=256, T_b=256, T_t=128,
                             rng=None):
        eps = [self.generate_train_episode(T_a, T_b, T_t, rng)
               for _ in range(batch_size)]
        n = max(e[0].shape[0] for e in eps)
        toks = torch.full((batch_size, n), self.mask_tok, dtype=torch.long)
        rev = torch.zeros(batch_size, n, dtype=torch.bool)
        for i, e in enumerate(eps):
            toks[i, :e[0].shape[0]] = e[0]
            rev[i, :e[1].shape[0]] = e[1]
        return toks, rev

    def generate_batch(self, batch_size: int, T_a=96, T_b=96, T_test=24, rng=None):
        eps = [self.generate_episode(T_a, T_b, T_test, rng) for _ in range(batch_size)]
        n = max(e[0].shape[0] for e in eps)
        toks = torch.full((batch_size, n), self.mask_tok, dtype=torch.long)
        rev = torch.zeros(batch_size, n, dtype=torch.bool)
        for i, e in enumerate(eps):
            toks[i, :e[0].shape[0]] = e[0]; rev[i, :e[1].shape[0]] = e[1]
        return toks, rev, [e[2] for e in eps], [e[3] for e in eps], [e[4] for e in eps]
