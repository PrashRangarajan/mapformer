"""Schema transfer: shortcut travel through NEVER-VISITED locations. From CSCG.

Source (George et al., bioRxiv 10.1101/864421v4), verbatim:

    "we first trained the CSCG on Room 1 based on aliased observations from a
     random walk... Next, we placed the agent in Room 2 which is unfamiliar. We
     kept the transition matrix of the CSCG fixed, and re-initialized the emission
     matrix to random values... Even without visiting all the locations in the new
     room, the CSCG is able to make shortcut travels between visited locations
     through locations that have never been visited... After a short traversal
     along the periphery... we queried to find the shortest path from the end state
     to the start state. The CSCG returned the correct sequence of actions, even
     though it obviously cannot predict the observations along the path."

WHAT IS AND IS NOT PORTED, stated rather than buried
----------------------------------------------------
CSCG's test is PLANNING -- query the shortest path, check the action sequence.
That is not portable here for two reasons:

  1. MapFormer has no planner. There is no transition graph to run Dijkstra on.
  2. Scoring a planner's action sequence is EXACTLY the failure that voided five
     tasks in this repo on 2026-08-09: a shortest path is self-predictable, with
     an n-gram on the action stream alone scoring 0.650-0.969 against a chance of
     0.250 (`PLANNER_TASK_AUDIT.md`).

So this ports the CAPABILITY the paper states underneath the planning demo --
"shortcut travels between visited locations through locations that have never
been visited" -- in a form that is measurable without a planner.

Episode
-------
    phase P   traverse ONLY the room PERIPHERY, observations REVEALED.
              The interior is never seen.
    phase S   walk a SHORTCUT: leave the periphery, cross the unvisited interior,
              and re-emerge. Observations WITHHELD (MASK) throughout.
              Score ONLY where the agent lands back on a periphery cell it saw in
              phase P.

To answer, the model must path-integrate ACROSS cells it has no observations for,
then recognise where it came out. That is the schema claim -- structure carries
you over unknown territory -- with no planning and no demonstration to imitate.

Interior cells are never scored (their observations are genuinely unknowable), so
the metric cannot be inflated by guessing there.

Rooms are BOUNDED (walls), matching CSCG's rooms rather than our torus.
"""
import numpy as np
import torch

ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class SchemaWorld:
    """Periphery-only exploration, then shortcuts through the unvisited interior."""

    N_ACTIONS = 4

    def __init__(self, h: int = 8, w: int = 8, n_obs_types: int = 20, seed: int = 0):
        self.h, self.w, self.n_obs_types = h, w, n_obs_types
        self.seed = seed
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.mask_tok = self.N_ACTIONS + n_obs_types
        self.unified_vocab_size = self.mask_tok + 1

    def _is_periphery(self, x, y):
        return x == 0 or y == 0 or x == self.h - 1 or y == self.w - 1

    def _periphery_ring(self):
        """Cells of the border, in walk order (clockwise from the origin)."""
        top = [(0, y) for y in range(self.w)]
        right = [(x, self.w - 1) for x in range(1, self.h)]
        bottom = [(self.h - 1, y) for y in range(self.w - 2, -1, -1)]
        left = [(x, 0) for x in range(self.h - 2, 0, -1)]
        return top + right + bottom + left

    @staticmethod
    def _action_between(a_cell, b_cell):
        dx, dy = b_cell[0] - a_cell[0], b_cell[1] - a_cell[1]
        for a, d in ACTION_DELTAS.items():
            if d == (dx, dy):
                return a
        return None

    def generate_episode(self, n_laps: int = 2, T_short: int = 40, rng=None):
        if rng is None:
            rng = np.random
        grid = rng.randint(0, self.n_obs_types, size=(self.h, self.w))
        tokens, revisit = [], []
        seen = set()

        # ---- phase P: periphery only; the interior is NEVER observed ----
        ring = self._periphery_ring()
        start = int(rng.randint(len(ring)))
        cur = ring[start]
        seen.add(cur)
        for step in range(1, n_laps * len(ring) + 1):
            nxt = ring[(start + step) % len(ring)]
            a = self._action_between(cur, nxt)
            if a is None:            # ring wraps at the seam; skip the non-adjacent hop
                cur = nxt; continue
            cur = nxt
            tokens += [a + self.action_offset, int(grid[cur]) + self.obs_offset]
            revisit += [False, False]; seen.add(cur)

        # ---- phase S: shortcuts across the unvisited interior, blind ----
        score_pos, answers, scored = [], [], set()
        x, y = cur
        for _ in range(T_short):
            valid = [a for a, (dx, dy) in ACTION_DELTAS.items()
                     if 0 <= x + dx < self.h and 0 <= y + dy < self.w]
            a = int(valid[rng.randint(len(valid))])
            dx, dy = ACTION_DELTAS[a]
            x, y = x + dx, y + dy
            sp = len(tokens)
            tokens += [a + self.action_offset, self.mask_tok]
            revisit += [False, False]
            # scored ONLY on re-emergence at a periphery cell already seen.
            # Interior cells are unknowable and are never scored.
            if self._is_periphery(x, y) and (x, y) in seen and (x, y) not in scored:
                scored.add((x, y)); score_pos.append(sp)
                answers.append(int(grid[x, y]) + self.obs_offset)

        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(revisit, dtype=torch.bool),
                score_pos, answers,
                {"n_scored": len(score_pos), "n_seen": len(seen)})

    def generate_batch(self, batch_size: int, n_laps=2, T_short=40, rng=None):
        eps = [self.generate_episode(n_laps, T_short, rng) for _ in range(batch_size)]
        n = max(e[0].shape[0] for e in eps)
        toks = torch.full((batch_size, n), self.mask_tok, dtype=torch.long)
        rev = torch.zeros(batch_size, n, dtype=torch.bool)
        for i, e in enumerate(eps):
            toks[i, :e[0].shape[0]] = e[0]; rev[i, :e[1].shape[0]] = e[1]
        return toks, rev, [e[2] for e in eps], [e[3] for e in eps], [e[4] for e in eps]
