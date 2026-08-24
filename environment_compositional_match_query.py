"""Compositional Match-Query: blind continuation in a repeated-motif world.

Combines two already-validated pieces of this repo:
  - the compositional-motif env (environment_compositional.py): a torus tiled
    into rooms, each room one of `n_templates` motif patterns, templates + layout
    redrawn every episode so nothing is memorised across episodes.
  - the Match-Query blind-continuation protocol (environment_match_query.py):
    explore with observations REVEALED, then continue BLIND (observations
    withheld, MASK fed back) and predict the observation at the current cell.

Why this task (the synergy the flat compositional task could not force)
----------------------------------------------------------------------
On the visible-observation compositional task, cross-instance prediction is pure
CONTENT pattern-completion, which a plain transformer solves without any spatial
code (the separability result). Making the query phase BLIND removes the content
channel, so the model must localise itself by PATH INTEGRATION to answer at all.
Scored two ways:

  exact-blind        the exact cell (x,y) was visited during explore, non-blank.
                     Answerable by path-integration MATCHING alone (this is
                     ordinary Match-Query).

  cross-instance     the exact cell was NOT visited, but (a) the current ROOM was
  -blind             visited during explore (so its template is identifiable by
                     matching) and (b) the motif-cell (template, local) was seen
                     during explore in a DIFFERENT copy (so the content is
                     known). Answerable ONLY by path-integration AND motif
                     abstraction TOGETHER: localise -> identify this room's
                     template -> reuse the pattern learned from another copy.

Neither axis alone suffices for the cross-instance-blind column: a plain
transformer cannot localise when blind; a flat MapFormer can localise but has
not bound motif->position, so it can only hit cells it exactly visited. Both
columns have oracle accuracy 1.0 by construction (true position + true map).

Scoring restricted to NON-BLANK cells so chance is 1/n_obs_types (=0.0625) not
the blank-majority rate. Each absolute cell scored at most once per query phase
(dedup) -- load-bearing: without it the query walk revisits cells, the answer
repeats, and the order-1 answer n-gram gate fires (the bug that invalidated
hier-goal twice). Gates in validate_compositional_match_query.py.
"""
import numpy as np
import torch

from .environment_compositional import CompositionalGridWorld


class CompositionalMatchQueryGridWorld(CompositionalGridWorld):
    def __init__(self, size: int = 64, room_size: int = 8, n_obs_types: int = 16,
                 n_templates: int = 4, p_empty: float = 0.5, seed: int = 0,
                 start=(0, 0)):
        super().__init__(size=size, room_size=room_size, n_obs_types=n_obs_types,
                         n_templates=n_templates, p_empty=p_empty,
                         fresh_per_episode=True, seed=seed)
        self.start = start
        # MASK token sits one past the compositional vocab (actions|obs|blank)
        self.mask_tok = self.unified_vocab_size
        self.unified_vocab_size = self.mask_tok + 1

    def _walk(self, rng, x, y, n_steps):
        """Directed random walk, paper run-length 1..10. Yields (action, x, y)."""
        t = 0
        while t < n_steps:
            a = int(rng.randint(0, self.N_ACTIONS))
            k = int(rng.randint(1, 11))
            dx, dy = self.ACTION_DELTAS[a]
            for _ in range(k):
                if t >= n_steps:
                    break
                x = (x + dx) % self.size
                y = (y + dy) % self.size
                yield a, x, y
                t += 1

    def generate_cmq_episode(self, T_explore: int = 256, T_query: int = 64, rng=None):
        if rng is None:
            rng = self._rng
        S = self.room_size
        templates, room_tmpl = self._draw_world(rng)
        obs_map = self._obs_map_from(templates, room_tmpl)

        x, y = self.start
        tokens: list[int] = []
        revisit: list[bool] = []
        seen_cells: set = set()                 # exact (x,y) seen in explore
        seen_rooms: set = set()                 # room ids explored (template known)
        seen_motif: dict = {}                   # (tmpl, lx, ly) -> obs value

        # ---- explore: observations REVEALED ----
        for a, x, y in self._walk(rng, x, y, T_explore):
            tokens.append(a + self.action_offset); revisit.append(False)
            o = int(obs_map[x, y])
            tokens.append(o + self.obs_offset)
            revisit.append((x, y) in seen_cells)
            ri, rj = self.room_of(x, y)
            tmpl = int(room_tmpl[ri, rj])
            seen_cells.add((x, y))
            seen_rooms.add((ri, rj))
            seen_motif[(tmpl, x % S, y % S)] = o

        end_explore = (x, y)

        # ---- query: observations WITHHELD (MASK fed back) ----
        score_pos: list[int] = []
        answers: list[int] = []
        categories: list[str] = []              # "exact" | "cross"
        scored_cells: set = set()               # dedup per cell
        consistency_fail = 0
        for a, x, y in self._walk(rng, x, y, T_query):
            pos = len(tokens)                   # predict AT the action token
            tokens.append(a + self.action_offset); revisit.append(False)
            tokens.append(self.mask_tok); revisit.append(False)

            o = int(obs_map[x, y])
            if o == self.blank_token or (x, y) in scored_cells:
                continue
            ri, rj = self.room_of(x, y)
            tmpl = int(room_tmpl[ri, rj])
            mkey = (tmpl, x % S, y % S)
            exact = (x, y) in seen_cells
            room_known = (ri, rj) in seen_rooms
            motif_known = mkey in seen_motif

            if exact:
                cat = "exact"
            elif room_known and motif_known:
                cat = "cross"
                # consistency: the remembered motif value must equal the answer
                if seen_motif[mkey] != o:
                    consistency_fail += 1
            else:
                continue                        # novel / room-unknown -> unanswerable

            scored_cells.add((x, y))
            score_pos.append(pos)
            answers.append(o + self.obs_offset)
            categories.append(cat)

        full = torch.tensor(tokens, dtype=torch.long)
        rev = torch.tensor(revisit, dtype=torch.bool)
        info = {"end_explore_pos": end_explore, "T_explore": T_explore,
                "T_query": T_query, "n_scored": len(score_pos),
                "n_exact": categories.count("exact"),
                "n_cross": categories.count("cross"),
                "consistency_fail": consistency_fail, "obs_map": obs_map,
                "seen_cells": seen_cells, "categories": categories}
        return full, rev, score_pos, answers, categories, info

    def generate_cmq_batch(self, batch_size: int, T_explore: int = 256,
                           T_query: int = 64, rng=None):
        toks, revs, sps, ans, cats, infos = [], [], [], [], [], []
        for _ in range(batch_size):
            t, rv, sp, a, c, info = self.generate_cmq_episode(T_explore, T_query, rng)
            toks.append(t); revs.append(rv); sps.append(sp)
            ans.append(a); cats.append(c); infos.append(info)
        return torch.stack(toks), torch.stack(revs), sps, ans, cats, infos
