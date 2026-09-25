"""Match-Query task: blind continuation. Tests position MATCHING, not decoding.

Why this replaces the Map-Query task
------------------------------------
Map-Query asked "which room are you in?", which requires DECODING absolute
position -- a modular sum of ~256 signed steps. Measured outcome at
T_explore=256 after 200 epochs: room accuracy 0.121 (chance 0.016) and the
direction query completely dead at 0.497 (chance 0.50), with its loss flat at
0.70 for every one of 200 epochs. At T_explore=16 the same code reaches 0.994 /
0.969, so the task was correct but asked for a capability these architectures do
not have. MapFormer's position code exists to make positions COMPARABLE, never
readable: the paper's revisit task only ever MATCHES two positions.

Design
------
    [ explore:  (action, observation) x T_explore ]      map is built here
    [ query:    (action, MASK)        x T_query   ]      observations WITHHELD

At each query step the model predicts the observation at the cell it now stands
on. It must (1) integrate the actions to know where it is, and (2) retrieve what
it saw there during explore. Step 2 is attention matching against stored key
positions -- precisely A_P = P.P^T -- so nothing has to be decoded.

Scored only where the current cell was VISITED during explore (otherwise the
answer is unknowable) and is NON-BLANK. Restricting to non-blank puts chance at
1/n_obs_types = 0.0625 rather than ~0.5, since p_empty=0.5 makes blank the
majority class.

Difference from the paper's own task, and why it is a real test: the paper
reveals observations at every step, so the map keeps updating and revisits are
incidental. Here the query phase is BLIND, so the model must rely on the map
built during explore, and the query phase can be pushed arbitrarily far OOD.

Known shortcut risks, all gated in validate_match_query.py:
  - n-gram over the answer stream. If the query walk revisits a cell, the answer
    repeats and an order-1 model catches it. This is the family of bug that
    invalidated hier-goal twice; it must be measured, not assumed.
  - "never moved": predict the observation at the end-of-explore cell every time.
  - marginal: always predict the most common observation.
"""
import numpy as np
import torch

from .environment import GridWorld


class MatchQueryGridWorld(GridWorld):
    """Explore with observations revealed, then continue blind and predict them."""

    def __init__(self, size: int = 64, n_obs_types: int = 16,
                 p_empty: float = 0.5, seed: int = 0, start=(0, 0),
                 mask_tok: int | None = None, vocab_size: int | None = None):
        """mask_tok / vocab_size override the token layout so this env can SHARE
        a vocabulary with LapWorld -- required to train one model on both tasks
        and ask whether learning one degrades the other."""
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                         n_landmarks=0, seed=seed)
        self.start = start
        self.mask_tok = (self.N_ACTIONS + self.obs_vocab_size
                         if mask_tok is None else mask_tok)
        self.unified_vocab_size = (self.mask_tok + 1 if vocab_size is None
                                   else vocab_size)

    def _draw_obs(self, rng):
        obs = np.full((self.size, self.size), self.blank_token, dtype=np.int64)
        occ = rng.random((self.size, self.size)) >= self.p_empty
        obs[occ] = rng.randint(0, self.n_obs_types, occ.sum())
        return obs

    def _walk(self, rng, x, y, n_steps, p_transition_noise: float = 0.0):
        """Directed random walk, paper's run-length 1..10. Yields (action, x, y).

        `p_transition_noise` makes this a STOCHASTIC-TRANSITION MDP: the walk
        RECORDS the commanded action but EXECUTES a resampled one with this
        probability, so the position that generates observations diverges from the
        position implied by the recorded action stream. That divergence is the
        drift an InEKF exists to correct, and it is absent from the clean task --
        which is why testing a correction mechanism here was a design error the
        first time (rule 17: check the premise applies).

        The rng draw is SHORT-CIRCUITED at p=0 so the default consumes exactly the
        stream it consumed before; verified byte-identical.
        """
        t = 0
        while t < n_steps:
            a = int(rng.randint(0, self.N_ACTIONS))
            k = int(rng.randint(1, 11))
            for _ in range(k):
                if t >= n_steps:
                    break
                a_exec = a
                if p_transition_noise > 0.0 and rng.random() < p_transition_noise:
                    a_exec = int(rng.randint(0, self.N_ACTIONS))
                dx, dy = self.ACTION_DELTAS[a_exec]
                x = (x + dx) % self.size
                y = (y + dy) % self.size
                yield a, x, y
                t += 1

    def generate_match_episode(self, T_explore: int = 256, T_query: int = 64,
                               rng=None, p_transition_noise: float = 0.0):
        """`p_transition_noise` is applied to the EXPLORE phase ONLY.

        Not a convenience -- applying it to the query phase makes the task
        UNANSWERABLE rather than harder. Scoring is keyed on the agent's TRUE
        cell, so if query transitions are stochastic the model cannot know which
        cell it is being asked about and no amount of memory helps; the ceiling
        falls to chance. Explore-only is also the honest setting: odometry is
        unreliable while the map is being built, and the map is then queried.
        Gated in validate_match_query_noise.py, which measures both.
        """
        if rng is None:
            rng = np.random
        obs_map = self._draw_obs(rng)
        x, y = self.start
        tokens: list[int] = []
        revisit: list[bool] = []
        seen = set()

        # ---- explore: observations REVEALED, map is built here ----
        for a, x, y in self._walk(rng, x, y, T_explore,
                                  p_transition_noise):
            tokens.append(a + self.action_offset); revisit.append(False)
            tokens.append(int(obs_map[x, y]) + self.obs_offset)
            revisit.append((x, y) in seen)
            seen.add((x, y))

        # ---- query: observations WITHHELD (MASK fed back) ----
        end_explore = (x, y)
        score_pos: list[int] = []
        answers: list[int] = []
        query_cells: list[tuple] = []
        scored_cells: set = set()      # DEDUP: each cell scored at most once
        for a, x, y in self._walk(rng, x, y, T_query):
            score_pos_candidate = len(tokens)     # predict AT the action token
            tokens.append(a + self.action_offset); revisit.append(False)
            tokens.append(self.mask_tok); revisit.append(False)
            o = int(obs_map[x, y])
            # answerable only if seen during explore, and non-blank so that
            # chance is 1/n_obs_types rather than the blank majority rate
            # Dedup is load-bearing, not cosmetic. Without it the query walk
            # revisits cells (run-lengths 1..10 go back and forth), the answer
            # repeats, and both the order-1 n-gram gate (0.114-0.170 vs chance
            # 0.0625) and the never-moved gate (0.213-0.325) FAIL.
            if (x, y) in seen and o != self.blank_token and (x, y) not in scored_cells:
                scored_cells.add((x, y))
                score_pos.append(score_pos_candidate)
                answers.append(o + self.obs_offset)
                query_cells.append((x, y))

        full = torch.tensor(tokens, dtype=torch.long)
        rev = torch.tensor(revisit, dtype=torch.bool)
        info = {"end_explore_pos": end_explore, "query_cells": query_cells,
                "T_explore": T_explore, "T_query": T_query,
                "n_scored": len(score_pos), "obs_map": obs_map, "seen": seen}
        return full, rev, score_pos, answers, info

    def generate_match_batch(self, batch_size: int, T_explore: int = 256,
                             T_query: int = 64, rng=None,
                             p_transition_noise: float = 0.0):
        toks, revs, sps, ans, infos = [], [], [], [], []
        for _ in range(batch_size):
            t, rv, sp, a, info = self.generate_match_episode(
                T_explore, T_query, rng, p_transition_noise)
            toks.append(t); revs.append(rv); sps.append(sp); ans.append(a)
            infos.append(info)
        return torch.stack(toks), torch.stack(revs), sps, ans, infos
