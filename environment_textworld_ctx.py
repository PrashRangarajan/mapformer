"""The text world with DECOYS: direction words used without moving (CONTEXT_STEP_DESIGN.md).

Identical to environment_textworld.TextWorld (same walk, map, clause grammar and revisit targets),
except that after each movement clause, with probability `p_decoy`, a decoy sentence is inserted
that uses a direction word WITHOUT moving the walker:

    she thought about the <dir> .      (cue 3 tokens before the direction word)
    a sign pointed <dir> .             (cue 1 token before)
    she did not go <dir> .             (cue 2 tokens before: negation)

<dir> is any of the 12 direction synonyms, uniformly. Decoys carry no object slot, so no target.
Every cue sits BEFORE the direction word, so a causal window of 4 tokens sees it. At p_decoy = 0 the
token stream is identical to TextWorld's (no extra RNG draws are made).

`self.ctx` after generate_trajectory lists (token index, class) for every direction word, class in
{"move", "about", "sign", "not"}, for the step probe.
"""
import numpy as np
import torch

from mapformer.environment_textworld import (TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE,
                                             OBJECTS, ASIDE)

DECOYS = {"about": ["she", "thought", "about", "the"], "sign": ["a", "sign", "pointed"],
          "not": ["she", "did", "not", "go"]}


class TextWorldCtx(TextWorld):
    def __init__(self, *a, p_decoy=0.3, **kw):
        super().__init__(*a, **kw)
        self.p_decoy = p_decoy
        for w in [w for ws in DECOYS.values() for w in ws]:
            if w not in self.idx:                    # append: existing ids are unchanged
                self.idx[w] = len(self.vocab); self.vocab.append(w)
        self.unified_vocab_size = len(self.vocab)
        self.all_dir = [w for ws in DIRS.values() for w in ws]
        self.ctx = []

    def generate_trajectory(self, n_tokens=1024):
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        locs = list(self.grid.visited_locations)
        out, obs, rv, self.visited_locations, self.ctx = [], [], [], [], []
        for s in range(steps):
            a, o = int(tok[2 * s]), int(tok[2 * s + 1])
            words = [VERBS[r.randint(len(VERBS))]]
            if r.random() < self.p_adverb:
                words.append(ADVERBS[r.randint(len(ADVERBS))])
            dir_at = [(len(words), "move")]
            words.append(DIRS[a][r.randint(3)])
            if r.random() < self.p_filler:
                words += [FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))]
            words += SEE[r.randint(len(SEE))]
            obj_at = len(words)
            words += [self._obj_word(o), "."]
            if r.random() < self.p_aside:
                words += ASIDE[r.randint(len(ASIDE))] + [OBJECTS[r.randint(self.n_obs_types)], "."]
            if self.p_decoy > 0 and r.random() < self.p_decoy:
                kind = list(DECOYS)[r.randint(len(DECOYS))]
                words += DECOYS[kind]
                dir_at.append((len(words), kind))
                words += [self.all_dir[r.randint(len(self.all_dir))], "."]
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
