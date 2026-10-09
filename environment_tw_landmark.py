"""Landmarks vs path integration in words (TW_LANDMARK_PREREG.md).

The text world (environment_textworld.TextWorld: 64x64 torus walk told in English, 58 words) plus PLACE NAMES.
Per walk, every distinct cell is a landmark with probability `name_rate` (one uniform draw per cell); a landmark gets
a place name drawn without replacement from a pool of N_NAMES single-token names, fresh per walk, so a name cannot be
memorised across sequences and must be bound in context. EVERY arrival at a landmark cell carries the clause
`reached <name>` just before the seeing phrase:

    walked north then paused reached n417 and saw lamp .        (landmark cell)
    went slowly east and found nothing .                         (unnamed cell)

The name sits exactly 3 tokens before the object slot (name, 2 seeing words, object), the easiest layout for an
index model's previous-token / induction circuit (the index arm gets its best shot at name lookup).

Draw discipline (what makes the eval conditions PAIRED and r = 0 the text world byte for byte):
  * the walk and every rendering word are drawn from the global numpy RNG in exactly TextWorld's order and number
    (the step loop stops where TextWorld's stops, i.e. where the NAME-FREE rendering fills n_tokens), whatever the
    name rate or mode; names are then inserted and the result truncated to n_tokens (names only shorten the walk);
  * names come from a private RandomState seeded by a CRC32 of the walk itself (actions, observations, start), so
    they consume no global draws. At name_rate 0 the token stream equals TextWorld's (checked in
    docs/audits/2026-10-08/tw_landmark/tw_landmark_checks.py), and every eval condition of one walk is the same
    walk with only the names changed.
  * landmark draws are coupled across rates (a cell is a landmark iff u_cell < rate), so the rate-0.5 landmarks are
    a subset of the rate-1 landmarks with the same names.

Modes (eval only; training is always 'consistent'):
  consistent  a landmark's every arrival shows its one name.
  fresh       every arrival at a landmark shows a NEW name never used before in the walk: the rendering has the
              training form (same clause positions) but names carry no information (the 'names uninformative' test).
  conflict    as consistent, except that each REVISIT to a landmark, with probability 0.5, shows the name of a
              different landmark visited earlier in the rendering (cue conflict: path says one cell, name another).

Per object slot, `self.slots` records (token index, move index, cell, landmark?, revisit?, conflict?, alt object
token or -1): the eval's target bookkeeping.
"""
import zlib

import numpy as np
import torch

from mapformer.environment_textworld import TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE, OBJECTS, ASIDE

N_NAMES = 512
MARK = "reached"
NAMES = [f"n{i:03d}" for i in range(N_NAMES)]


class TextWorldLandmark(TextWorld):
    def __init__(self, size=64, n_obs_types=16, p_empty=0.5, seed=None, name_rate=0.0, mode="consistent",
                 p_conflict=0.5, **kw):
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty, seed=seed, **kw)
        assert MARK not in self.idx and not set(NAMES) & set(self.idx)
        self.base_vocab_size = len(self.vocab)                  # 58: TextWorld's ids are unchanged
        self.vocab = self.vocab + [MARK] + NAMES                # same vocabulary at every rate (same init per seed)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.unified_vocab_size = len(self.vocab)
        self.mark_id = self.idx[MARK]
        self.name_ids = [self.idx[w] for w in NAMES]
        self.name_rate, self.mode, self.p_conflict = float(name_rate), mode, float(p_conflict)
        self.slots = []

    # ------------------------------------------------------------------ rendering
    def _draw(self, n_tokens):
        """TextWorld's draws, in TextWorld's order and number. -> (clauses, locs, rev, obs_tok); a clause is
        (words, obj_at) with words before name insertion."""
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        locs = list(self.grid.visited_locations)
        clauses, used = [], 0
        for s in range(steps):
            a, o = int(tok[2 * s]), int(tok[2 * s + 1])
            words = [VERBS[r.randint(len(VERBS))]]
            if r.random() < self.p_adverb:
                words.append(ADVERBS[r.randint(len(ADVERBS))])
            words.append(DIRS[a][r.randint(3)])
            if r.random() < self.p_filler:
                words += [FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))]
            words += SEE[r.randint(len(SEE))]
            obj_at = len(words)
            words += [self._obj_word(o), "."]
            if r.random() < self.p_aside:
                words += ASIDE[r.randint(len(ASIDE))] + [OBJECTS[r.randint(self.n_obs_types)], "."]
            if n_tokens - used <= 0:                            # TextWorld's break: after the draws of this step
                break
            clauses.append((words, obj_at, bool(rev[2 * s + 1]), tuple(locs[s])))
            used += len(words)
        else:
            raise RuntimeError(f"{steps} steps did not fill n_tokens={n_tokens}")
        return clauses, tok

    def _names(self, clauses, walk_tok, rate, mode):
        """Per clause: the name id shown (or None) and, in conflict mode, the cell whose name was shown."""
        h = zlib.crc32(np.ascontiguousarray(walk_tok, dtype=np.int64).tobytes())
        h = zlib.crc32(np.asarray(clauses[0][3], dtype=np.int64).tobytes(), h)
        rs = np.random.RandomState(h)
        cells = list(dict.fromkeys(c[3] for c in clauses))
        assert len(cells) <= N_NAMES, len(cells)
        u = rs.random_sample(len(cells)); perm = rs.permutation(N_NAMES)
        fresh = rs.permutation(N_NAMES); conf_u = rs.random_sample(len(clauses)); conf_pick = rs.random_sample(len(clauses))
        land = {c: (u[j] < rate) for j, c in enumerate(cells)}
        name = {c: self.name_ids[perm[j]] for j, c in enumerate(cells)}
        out, nf, seen_land = [], 0, []
        for k, (_w, _o, revisit, cell) in enumerate(clauses):
            if not land[cell]:
                out.append((None, None))
                continue
            if mode == "fresh":
                assert nf < N_NAMES, "fresh-name pool exhausted"
                out.append((self.name_ids[fresh[nf]], None)); nf += 1
            elif mode == "conflict" and revisit and conf_u[k] < self.p_conflict and \
                    [c for c in seen_land if c != cell]:
                others = [c for c in seen_land if c != cell]
                alt = others[int(conf_pick[k] * len(others))]
                out.append((name[alt], alt))
            else:
                assert mode in ("consistent", "conflict"), mode
                out.append((name[cell], None))
            if cell not in seen_land:
                seen_land.append(cell)
        return out, land

    def generate_trajectory(self, n_tokens=1024, name_rate=None, mode=None):
        """-> (tokens, obs_mask, revisit_mask), each of length n_tokens (TextWorld's contract). Bookkeeping per
        rendered object slot in self.slots; self.visited_locations as TextWorld."""
        rate = self.name_rate if name_rate is None else float(name_rate)
        mode = self.mode if mode is None else mode
        clauses, walk_tok = self._draw(n_tokens)
        shown, land = self._names(clauses, walk_tok, rate, mode) if rate > 0 else ([(None, None)] * len(clauses), {})
        obj_of = {c[3]: c[0][c[1]] for c in clauses}           # the object word at each cell
        out, obs, rv = [], [], []
        self.visited_locations, self.slots = [], []
        for k, ((words, obj_at, revisit, cell), (nm, alt)) in enumerate(zip(clauses, shown)):
            room = n_tokens - len(out)
            if room <= 0:
                break
            ids = [self.idx[w] for w in words]
            if nm is not None:                                  # insert 'reached <name>' before the seeing phrase
                ids = ids[:obj_at - 2] + [self.mark_id, nm] + ids[obj_at - 2:]
                obj_at += 2
            flags = [False] * len(ids); rflags = [False] * len(ids)
            if obj_at < room:                                   # the object slot fits: a real target
                flags[obj_at] = True; rflags[obj_at] = revisit
                self.visited_locations.append(cell)
                alt_obj = self.idx[obj_of[alt]] if alt is not None else -1
                self.slots.append((len(out) + obj_at, k, cell, bool(land.get(cell, False)), revisit,
                                   alt is not None, alt_obj))
            out += ids[:room]; obs += flags[:room]; rv += rflags[:room]
        assert len(out) == n_tokens, (len(out), n_tokens)
        return (torch.tensor(out, dtype=torch.long), torch.tensor(obs, dtype=torch.bool),
                torch.tensor(rv, dtype=torch.bool))


def render_conditions(env, n_tokens, conds):
    """Render ONE walk under several (rate, mode) conditions from the same global RNG state; every condition consumes
    exactly the same draws (asserted). -> {cond: (tokens, obs, rev, slots)}; the RNG is left after the walk."""
    st = np.random.get_state(); res = {}; end = None
    for c in conds:
        np.random.set_state(st)
        t, o, r = env.generate_trajectory(n_tokens, name_rate=c[0], mode=c[1])
        res[c] = (t, o, r, list(env.slots))
        e = np.random.get_state()
        if end is None:
            end = e
        else:
            assert e[2] == end[2] and np.array_equal(e[1], end[1]), "conditions consumed different draws"
    np.random.set_state(end)
    return res
