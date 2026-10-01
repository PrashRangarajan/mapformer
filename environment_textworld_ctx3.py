"""The text world with decoys at a controlled CUE DISTANCE (CONTEXT_STEP_DESIGN.md, second revision).

Walk, map and revisit targets are TextWorld's. Every step's movement clause, and every decoy, carries a
movement-free PAD phrase (", after a long pause ,", ...). A single cue word says whether the direction
word is a real move or a decoy. `dist` decides where the PAD goes relative to the cue, so the tokens per
step are the same in both distances and only the cue's distance to the direction word changes:

  cue="lead"   near:  she PAD <cue> <verb> [adv] <dir> ...        cue 2-3 tokens before <dir>
               far:   she <cue> PAD <verb> [adv] <dir> ...        cue 7-12 tokens before <dir>
               move cue in {then, soon, finally}; decoy cue in {never, nearly, almost}.
               After <dir>: [filler] <seeing phrase> <object> . for both (a decoy's object is the
               CURRENT cell's, correct because the walker did not move; not scored).
  cue="trail"  near:  <verb> [adv] <dir> <cue> PAD <rest>          cue 1 token after <dir>
               far:   <verb> [adv] <dir> PAD <cue> <rest>          cue 6-10 tokens after <dir>
               move: <cue> in {and, so, then}, <rest> = <saw|found|noticed> <object> .
               decoy: <cue> in {but, yet, though}, <rest> = she stayed .

A causal window of 4 tokens (the context gate's and the Selective-RoPE conv's) contains the cue only in
"near". `self.ctx` lists (token index, class) for every direction word, class in {"move", "decoy"}.
"""
import numpy as np
import torch

from mapformer.environment_textworld import TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE, OBJECTS, ASIDE

PADS = [[",", "after", "a", "long", "pause", ","], [",", "after", "some", "thought", ","],
        [",", "after", "a", "long", "pause", "and", "some", "thought", ","],
        [",", "with", "a", "sigh", "and", "a", "shrug", ","], [",", "for", "a", "while", ","]]
LEAD_MOVE, LEAD_DECOY = ["then", "soon", "finally"], ["never", "nearly", "almost"]
TRAIL_MOVE, TRAIL_DECOY = ["and", "so", "then"], ["but", "yet", "though"]
SEE1 = ["saw", "found", "noticed"]


class TextWorldCtx3(TextWorld):
    def __init__(self, *a, cue="lead", dist="near", p_decoy=0.3, **kw):
        super().__init__(*a, **kw)
        assert cue in ("lead", "trail") and dist in ("near", "far"), (cue, dist)
        self.cue, self.dist, self.p_decoy = cue, dist, p_decoy
        extra = [w for p in PADS for w in p] + LEAD_MOVE + LEAD_DECOY + TRAIL_MOVE + TRAIL_DECOY + SEE1 + ["stayed"]
        for w in extra:
            if w not in self.idx:
                self.idx[w] = len(self.vocab); self.vocab.append(w)
        self.unified_vocab_size = len(self.vocab)
        self.all_dir = [w for ws in DIRS.values() for w in ws]
        self.ctx = []

    def _clause(self, r, d, decoy, obj_word):
        """-> (words, index of the direction word, index of the object slot or None)."""
        pad = PADS[r.randint(len(PADS))]
        verb = [VERBS[r.randint(len(VERBS))]] + ([ADVERBS[r.randint(len(ADVERBS))]] if r.random() < self.p_adverb else [])
        if self.cue == "lead":
            c = (LEAD_DECOY if decoy else LEAD_MOVE)[r.randint(3)]
            head = ["she"] + (pad + [c] if self.dist == "near" else [c] + pad) + verb
            tail = ([FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))] if r.random() < self.p_filler else [])
            tail += SEE[r.randint(len(SEE))]
            words = head + [d] + tail
            obj_at = len(words)
            return words + [obj_word, "."], len(head), obj_at
        c = (TRAIL_DECOY if decoy else TRAIL_MOVE)[r.randint(3)]
        mid = [c] + pad if self.dist == "near" else pad + [c]
        rest = ["she", "stayed", "."] if decoy else [SEE1[r.randint(3)], obj_word, "."]
        words = verb + [d] + mid + rest
        return words, len(verb), (None if decoy else len(verb) + 1 + len(mid) + 1)

    def generate_trajectory(self, n_tokens=1024):
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        locs = list(self.grid.visited_locations)
        out, obs, rv, self.visited_locations, self.ctx = [], [], [], [], []
        for s in range(steps):
            a, o = int(tok[2 * s]), int(tok[2 * s + 1])
            ow = self._obj_word(o)
            words, di, obj_at = self._clause(r, DIRS[a][r.randint(3)], False, ow)
            dir_at = [(di, "move")]
            if r.random() < self.p_aside:
                words += ASIDE[r.randint(len(ASIDE))] + [OBJECTS[r.randint(self.n_obs_types)], "."]
            if r.random() < self.p_decoy:
                dw, ddi, _ = self._clause(r, self.all_dir[r.randint(len(self.all_dir))], True, ow)
                dir_at.append((len(words) + ddi, "decoy")); words += dw
            room = n_tokens - len(out)
            if room <= 0:
                break
            flags = [False] * len(words); rflags = [False] * len(words)
            if obj_at < room:
                self.visited_locations.append(locs[s])
                flags[obj_at] = True; rflags[obj_at] = bool(rev[2 * s + 1])
            base = len(out)
            self.ctx += [(base + i, k) for i, k in dir_at if i < room]
            out += [self.idx[w] for w in words][:room]; obs += flags[:room]; rv += rflags[:room]
        else:
            raise RuntimeError(f"{steps} steps did not fill n_tokens={n_tokens}")
        return (torch.tensor(out, dtype=torch.long), torch.tensor(obs, dtype=torch.bool),
                torch.tensor(rv, dtype=torch.bool))
