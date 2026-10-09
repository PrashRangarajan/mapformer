"""Navigation told in words with STATE-CHANGE clauses (TW_STATECHANGE_PREREG.md).

The text world (environment_textworld.TextWorld: 64x64 torus walk rendered as English, K=16 objects, synonyms,
fillers, asides) plus actions that change the world's state WITHOUT moving the agent:

    walked north and saw a lamp . she took the lamp .        (TAKE: the cell is now empty, the agent holds the lamp)
    went left and found nothing . she dropped the lamp .     (DROP: the cell now holds the lamp, hands empty)

A state clause always follows the step's core clause, at the agent's current cell, before any aside. Rules: the agent
holds at most one object; TAKE is eligible when the cell currently holds an object and the hands are empty; DROP when
the cell is currently empty and the agent holds something; every cell changes state AT MOST ONCE per sequence (so the
current content of a cell is decided by its latest event, and no recency rule among several changes is needed: a scope
statement, not a property of narratives). Take verbs {took, grabbed, lifted}, drop verbs {dropped, placed, released},
determiner "the" -- none is a direction word or a movement verb, and no direction word appears outside a movement
clause (MapFormer's step is context-free; "picked UP" / "put DOWN" would move the phase by construction).

Targets are unchanged in form: the OBJECT word of each step at a revisited cell, which now reports the cell's CURRENT
content. Strata (per scored slot, `slot_info`): 'T1' revisit, the cell never changed before this slot (pure location);
'T2take' / 'T2drop' the FIRST return after the cell's change (the answer differs from the last 'saw' at that cell: it
needs the location AND the state clause); 'T3' later returns (the last 'saw' already shows the new state).

With p_take = p_drop = 0 the token stream is byte-identical to TextWorld's at the same numpy state (no extra random
draw is made unless a state event is eligible and its probability is > 0); with state_vocab=False the vocabulary is
TextWorld's too (used to calibrate readouts on stored text-world checkpoints). New words are appended AFTER
TextWorld's vocabulary, so every shared word keeps its id.

Per-position classes (`pos_class`, int8) for the readouts: see CLS below. Optional positions (present a variable number
of times per move): adverb, filler, aside, take clause, drop clause. Core positions (exactly once per move): verb,
direction, see words, object slot, core '.'.
"""
import numpy as np
import torch

from mapformer.environment_textworld import TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE, OBJECTS, ASIDE

TAKE = ["took", "grabbed", "lifted"]
DROP = ["dropped", "placed", "released"]
DET = "the"
CLS = {"verb": 0, "adverb": 1, "dir": 2, "filler": 3, "see": 4, "obj": 5, "period": 6, "aside": 7, "take": 8, "drop": 9}
OPTIONAL = (1, 3, 7, 8, 9)
STRATA = ("first", "T1", "T2take", "T2drop", "T3")


class StateChangeWorld(TextWorld):
    def __init__(self, size=64, n_obs_types=16, p_empty=0.5, seed=None, p_adverb=0.3, p_filler=0.3, p_aside=0.15,
                 max_steps=400, p_take=0.4, p_drop=0.4, state_vocab=True):
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty, seed=seed, p_adverb=p_adverb,
                         p_filler=p_filler, p_aside=p_aside, max_steps=max_steps)
        self.p_take, self.p_drop = p_take, p_drop
        assert state_vocab or (p_take == 0 and p_drop == 0), "state clauses need the state vocabulary"
        if state_vocab:
            for w in TAKE + DROP + [DET]:
                assert w not in self.idx, w
                self.idx[w] = len(self.vocab); self.vocab.append(w)
            self.unified_vocab_size = len(self.vocab)
            self.take_ids = [self.idx[w] for w in TAKE]; self.drop_ids = [self.idx[w] for w in DROP]
        dirwords = {w for d in DIRS.values() for w in d}
        assert not (set(TAKE + DROP + [DET]) & (dirwords | set(VERBS))), "state words must not be moves"
        self.blank_k = n_obs_types
        self.pos_class = None; self.slot_info = []; self.clauses = []

    def _word(self, k):
        return "nothing" if k >= self.n_obs_types else OBJECTS[k]

    def generate_trajectory(self, n_tokens=1024):
        """-> (tokens, obs_mask, revisit_mask), each of length n_tokens (TextWorld's contract). Side outputs for
        readouts: self.visited_locations (cell of each fitting object slot), self.pos_class (np.int8 per token),
        self.slot_info (one dict per fitting slot: pos, cell, rev, stratum, answer word id, last_saw id, first_saw id,
        straddle_sc / straddle_aside = a state clause / aside lies between the cell's first observed visit and this
        slot, aside_here = an aside was told at this cell before this slot, sc_kind of the cell's change or None),
        self.clauses (one tuple per complete-or-cut optional sentence: (kind, start, end_exclusive, cell)), and
        self.events (per fitting slot k: list of state events (kind, object k) made right after it)."""
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        locs = list(self.grid.visited_locations)
        self.visited_locations = []
        out, obs, rv, pc = [], [], [], []
        content, changed, hands = {}, {}, None          # cell -> current k; cell -> (kind, step); held object k
        first_pos, last_saw, first_saw, n_visits_since = {}, {}, {}, {}
        sc_count, aside_count, aside_cells = 0, 0, set()
        sc_before, aside_before = {}, {}                 # cell -> counts at the cell's first observed visit
        info, clauses, events = [], [], []
        for s in range(steps):
            a, o = int(tok[2 * s]), int(tok[2 * s + 1])
            cell = tuple(locs[s])
            k_orig = o - self.grid.obs_offset
            k = content.get(cell, k_orig)
            words = [VERBS[r.randint(len(VERBS))]]; cls = [CLS["verb"]]
            if r.random() < self.p_adverb:
                words.append(ADVERBS[r.randint(len(ADVERBS))]); cls.append(CLS["adverb"])
            words.append(DIRS[a][r.randint(3)]); cls.append(CLS["dir"])
            if r.random() < self.p_filler:
                f = [FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))]
                words += f; cls += [CLS["filler"]] * len(f)
            see = SEE[r.randint(len(SEE))]; words += see; cls += [CLS["see"]] * len(see)
            obj_at = len(words)
            words += [self._word(k), "."]; cls += [CLS["obj"], CLS["period"]]
            # ---- state clause (no random draw unless eligible and p > 0: p = 0 reproduces TextWorld exactly)
            ev = None
            if cell not in changed:
                if hands is None and k < self.n_obs_types and self.p_take > 0 and r.random() < self.p_take:
                    v = TAKE[r.randint(len(TAKE))]; ev = ("take", k)
                elif hands is not None and k >= self.n_obs_types and self.p_drop > 0 and r.random() < self.p_drop:
                    v = DROP[r.randint(len(DROP))]; ev = ("drop", hands)
            sc_span = None
            if ev is not None:
                sc_span = (len(words), len(words) + 5)
                words += ["she", v, DET, self._word(ev[1]), "."]; cls += [CLS[ev[0]]] * 5
            as_span = None
            if r.random() < self.p_aside:
                as_words = ASIDE[r.randint(len(ASIDE))] + [OBJECTS[r.randint(self.n_obs_types)], "."]
                as_span = (len(words), len(words) + len(as_words))
                words += as_words; cls += [CLS["aside"]] * len(as_words)
            room = n_tokens - len(out)
            if room <= 0:
                break
            base = len(out)
            fits = obj_at < room
            flags = [False] * len(words); rflags = [False] * len(words)
            if fits:
                flags[obj_at] = True; rflags[obj_at] = bool(rev[2 * s + 1])
                self.visited_locations.append(locs[s])
                pos = base + obj_at
                is_rev = bool(rev[2 * s + 1])
                if not is_rev:
                    st = "first"
                elif cell not in changed:
                    st = "T1"
                elif n_visits_since[cell] == 0:
                    st = "T2" + changed[cell][0]
                else:
                    st = "T3"
                if cell not in first_pos:
                    first_pos[cell] = pos; first_saw[cell] = k; sc_before[cell] = sc_count; aside_before[cell] = aside_count
                info.append({"pos": pos, "cell": cell, "rev": is_rev, "stratum": st,
                             "answer": self.idx[self._word(k)],
                             "last_saw": self.idx[self._word(last_saw[cell])] if cell in last_saw else None,
                             "first_saw": self.idx[self._word(first_saw[cell])],
                             "straddle_sc": sc_count > sc_before[cell], "straddle_aside": aside_count > aside_before[cell],
                             "aside_here": cell in aside_cells, "sc_kind": changed[cell][0] if cell in changed else None})
                last_saw[cell] = k
                if cell in changed:
                    n_visits_since[cell] += 1
            # apply the state event (it is told after the slot, at the same cell)
            if ev is not None:
                changed[cell] = (ev[0], s); n_visits_since[cell] = 0
                if ev[0] == "take":
                    content[cell] = self.blank_k; hands = ev[1]
                else:
                    content[cell] = ev[1]; hands = None
                if base + sc_span[1] <= n_tokens:
                    clauses.append((ev[0], base + sc_span[0], base + sc_span[1], cell)); sc_count += 1
            if as_span is not None:
                aside_cells.add(cell)
                if base + as_span[1] <= n_tokens:
                    clauses.append(("aside", base + as_span[0], base + as_span[1], cell)); aside_count += 1
            if fits:
                events.append(ev)
            out += [self.idx[w] for w in words][:room]; obs += flags[:room]; rv += rflags[:room]; pc += cls[:room]
        else:
            raise RuntimeError(f"{steps} steps did not fill n_tokens={n_tokens}")
        self.pos_class = np.array(pc, dtype=np.int8); self.slot_info = info; self.clauses = clauses
        self.events = events
        return (torch.tensor(out, dtype=torch.long), torch.tensor(obs, dtype=torch.bool),
                torch.tensor(rv, dtype=torch.bool))
