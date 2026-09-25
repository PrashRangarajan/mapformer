"""
Compositional grid environment: repeatable room MOTIFS at different locations.

Purpose
-------
Test the hypothesis from the "same vs different point at the higher level"
discussion: a hierarchy earns its keep only when a lossy summary is a
SUFFICIENT STATISTIC. A compositional environment with repeated motifs is the
regime where that can hold — IF the task queries motif-level (structural)
information that is invariant to which copy of a motif you are in.

Environment
-----------
- Torus grid of size `size`, tiled into (size/room_size)^2 rooms.
- `n_templates` distinct S x S motif TEMPLATES (S = room_size). Each template
  is a fixed pattern of obs types. Each room is assigned one template, WITH
  repetition, so every template appears at several DIFFERENT locations.
- Observation at global cell (x, y) is fully determined by
  (template_of_room(x,y), x % S, y % S). So two different rooms that share a
  template emit the SAME observation pattern — they are "the same thing"
  locally, at different absolute locations.

`fresh_per_episode=True` (default): templates AND room assignment are redrawn
for every trajectory. This forbids cross-episode memorisation of a fixed
template set (the "fixed maze -> memorisation" trap) and forces the model to
infer motif structure IN CONTEXT within each episode. This is the honest
setting; set False only for debugging.

Two prediction targets (the dissociation)
-----------------------------------------
At each observation position we expose three masks:

- revisit_mask         : the EXACT absolute cell (x,y) was visited before.
                         Fine-sufficient — needs absolute location. (The paper's
                         standard target.)
- motif_revisit_mask   : the motif-cell (template, x%S, y%S) was visited before,
                         in ANY room copy.
- cross_instance_mask  : motif-cell seen before BUT the exact cell was NOT
                         (i.e. same motif, DIFFERENT room copy). This is the
                         compositional-transfer target: predictable ONLY if the
                         model recognises the two rooms share a template and
                         REUSES the pattern. A model that collapses motif
                         instances gets these "for free"; a model that keeps
                         instances distinct by absolute position does not.

Also exposed per step (via attributes collected by generate_batch), for oracle
segmentation and analysis:
    self.step_room_id, self.step_template_id, self.step_local,
    self.step_new_room  (True when the room changed vs the previous step)
"""

import torch
import numpy as np
from typing import Optional


class CompositionalGridWorld:
    N_ACTIONS = 4
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(
        self,
        size: int = 64,
        room_size: int = 8,
        n_obs_types: int = 16,
        n_templates: int = 4,
        p_empty: float = 0.5,
        fresh_per_episode: bool = True,
        seed: Optional[int] = None,
    ):
        assert size % room_size == 0
        self.size = size
        self.room_size = room_size
        self.rooms_per_side = size // room_size
        self.n_obs_types = n_obs_types
        self.n_templates = n_templates
        self.p_empty = p_empty
        self.fresh_per_episode = fresh_per_episode

        # Unified vocab: [0..3] actions, [4..4+K-1] obs, [4+K] blank
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.blank_token = n_obs_types
        self.unified_blank = self.N_ACTIONS + n_obs_types
        self.obs_vocab_size = n_obs_types + 1
        self.unified_vocab_size = self.N_ACTIONS + self.obs_vocab_size

        self._rng = np.random.RandomState(seed)
        if not fresh_per_episode:
            self._templates, self._room_tmpl = self._draw_world(self._rng)

    # -- world construction ------------------------------------------------
    def _draw_world(self, rng):
        """Return (templates, room_template_assignment).

        templates: (n_templates, S, S) obs-type array (blank_token allowed).
        room_tmpl: (rooms_per_side, rooms_per_side) template id per room.
        """
        S = self.room_size
        M = self.n_templates
        templates = np.full((M, S, S), self.blank_token, dtype=np.int64)
        occ = rng.random((M, S, S)) >= self.p_empty
        templates[occ] = rng.randint(0, self.n_obs_types, occ.sum())
        # ensure every template appears at least once, then fill the rest at
        # random so each template lands at SEVERAL different room locations
        R = self.rooms_per_side
        n_rooms = R * R
        assign = np.empty(n_rooms, dtype=np.int64)
        assign[:M] = np.arange(M)
        assign[M:] = rng.randint(0, M, n_rooms - M)
        rng.shuffle(assign)
        room_tmpl = assign.reshape(R, R)
        return templates, room_tmpl

    def _obs_map_from(self, templates, room_tmpl):
        """Materialise the full size x size obs map from templates+assignment."""
        S = self.room_size
        R = self.rooms_per_side
        obs = np.empty((self.size, self.size), dtype=np.int64)
        for ri in range(R):
            for rj in range(R):
                t = room_tmpl[ri, rj]
                obs[ri * S:(ri + 1) * S, rj * S:(rj + 1) * S] = templates[t]
        return obs

    def room_of(self, x, y):
        S = self.room_size
        return (x // S, y // S)

    # -- trajectory --------------------------------------------------------
    def generate_trajectory(self, n_steps=128, start=None, p_transition_noise=0.0):
        S = self.room_size
        if self.fresh_per_episode:
            templates, room_tmpl = self._draw_world(self._rng)
        else:
            templates, room_tmpl = self._templates, self._room_tmpl
        obs_map = self._obs_map_from(templates, room_tmpl)

        if start is not None:
            x, y = start
        else:
            x = self._rng.randint(0, self.size)
            y = self._rng.randint(0, self.size)

        tokens = []
        visited = []
        seen_cells = set()          # exact (x,y)
        seen_motif = set()          # (template_id, lx, ly)
        is_revisit, is_motif_rev, is_cross = [], [], []
        room_ids, tmpl_ids, locals_, new_room = [], [], [], []
        prev_room = None

        t = 0
        while t < n_steps:
            a = self._rng.randint(0, self.N_ACTIONS)
            k = self._rng.randint(1, 11)
            for _ in range(k):
                if t >= n_steps:
                    break
                a_exec = a
                if p_transition_noise > 0.0 and self._rng.random() < p_transition_noise:
                    a_exec = self._rng.randint(0, self.N_ACTIONS)
                dx, dy = self.ACTION_DELTAS[a_exec]
                x = (x + dx) % self.size
                y = (y + dy) % self.size

                ri, rj = self.room_of(x, y)
                tmpl = int(room_tmpl[ri, rj])
                lx, ly = x % S, y % S
                motif_key = (tmpl, lx, ly)

                exact_rev = (x, y) in seen_cells
                motif_rev = motif_key in seen_motif
                cross = motif_rev and not exact_rev

                tokens.append(a + self.action_offset)
                tokens.append(int(obs_map[x, y]) + self.obs_offset)

                visited.append((x, y))
                is_revisit.append(exact_rev)
                is_motif_rev.append(motif_rev)
                is_cross.append(cross)
                room_ids.append(ri * self.rooms_per_side + rj)
                tmpl_ids.append(tmpl)
                locals_.append((lx, ly))
                new_room.append(prev_room is None or (ri, rj) != prev_room)
                prev_room = (ri, rj)

                seen_cells.add((x, y))
                seen_motif.add(motif_key)
                t += 1

        self.visited_locations = visited
        self.step_room_id = room_ids
        self.step_template_id = tmpl_ids
        self.step_local = locals_
        self.step_new_room = new_room

        tokens = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        obs_mask[1::2] = True

        def _to_mask(flags):
            m = torch.zeros(2 * n_steps, dtype=torch.bool)
            for i, f in enumerate(flags):
                if f:
                    m[2 * i + 1] = True
            return m

        return (tokens, obs_mask, _to_mask(is_revisit),
                _to_mask(is_motif_rev), _to_mask(is_cross))

    def generate_batch(self, batch_size, n_steps=128, p_transition_noise=0.0):
        toks, oms, revs, motifs, crosses = [], [], [], [], []
        locs, rooms, tmpls, newrooms = [], [], [], []
        for _ in range(batch_size):
            tok, om, rev, motif, cross = self.generate_trajectory(
                n_steps, p_transition_noise=p_transition_noise)
            toks.append(tok); oms.append(om); revs.append(rev)
            motifs.append(motif); crosses.append(cross)
            locs.append(list(self.visited_locations))
            rooms.append(list(self.step_room_id))
            tmpls.append(list(self.step_template_id))
            newrooms.append(list(self.step_new_room))
        meta = {"locations": locs, "room_id": rooms,
                "template_id": tmpls, "new_room": newrooms}
        return (torch.stack(toks), torch.stack(oms), torch.stack(revs),
                torch.stack(motifs), torch.stack(crosses), meta)
