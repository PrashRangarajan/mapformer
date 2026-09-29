"""The text world with decoys, two cue conditions (CONTEXT_STEP_DESIGN.md, "Revision").

Walk, map, movement-clause grammar and revisit targets are TextWorld's. After each movement clause,
with probability `p_decoy`, a decoy uses a direction word WITHOUT moving the walker:

  cue="lead"   she did not go <dir> [filler] <seeing phrase> <object> .
               The cue comes BEFORE the direction word; everything AFTER it is drawn exactly as after a
               real move (same filler and seeing-phrase draws). The object is the CURRENT cell's, which
               is correct because the walker did not move; it is not scored.
  cue="trail"  <verb> [adverb] <dir> no , she stayed .
               Everything BEFORE the direction word is drawn exactly as before a real move; the cue
               (a retraction) comes right AFTER it. No observation.

<dir> is any of the 12 direction synonyms. `self.ctx` lists (token index, class) for every direction
word, class in {"move", "lead", "trail"}.
"""
import numpy as np
import torch

from mapformer.environment_textworld import (TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE,
                                             OBJECTS, ASIDE)

LEAD = [["she", "did", "not", "go"], ["she", "refused", "to", "go"], ["she", "thought", "about", "going"]]
TRAIL = [["no", ",", "she", "stayed", "."], ["but", "turned", "back", "at", "once", "."],
         ["--", "or", "rather", ",", "did", "not", "."]]


class TextWorldCtx2(TextWorld):
    def __init__(self, *a, cue="lead", p_decoy=0.3, **kw):
        super().__init__(*a, **kw)
        assert cue in ("lead", "trail"), cue
        self.cue, self.p_decoy = cue, p_decoy
        for w in [w for c in LEAD + TRAIL for w in c]:
            if w not in self.idx:
                self.idx[w] = len(self.vocab); self.vocab.append(w)
        self.unified_vocab_size = len(self.vocab)
        self.all_dir = [w for ws in DIRS.values() for w in ws]
        self.ctx = []

    def _after_dir(self, r):
        """The continuation after a direction word in a movement clause, up to the object slot."""
        words = []
        if r.random() < self.p_filler:
            words += [FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))]
        return words + SEE[r.randint(len(SEE))]

    def _before_dir(self, r):
        words = [VERBS[r.randint(len(VERBS))]]
        if r.random() < self.p_adverb:
            words.append(ADVERBS[r.randint(len(ADVERBS))])
        return words

    def generate_trajectory(self, n_tokens=1024):
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        locs = list(self.grid.visited_locations)
        out, obs, rv, self.visited_locations, self.ctx = [], [], [], [], []
        for s in range(steps):
            a, o = int(tok[2 * s]), int(tok[2 * s + 1])
            words = self._before_dir(r)
            dir_at = [(len(words), "move")]
            words.append(DIRS[a][r.randint(3)])
            words += self._after_dir(r)
            obj_at = len(words)
            words += [self._obj_word(o), "."]
            if r.random() < self.p_aside:
                words += ASIDE[r.randint(len(ASIDE))] + [OBJECTS[r.randint(self.n_obs_types)], "."]
            if r.random() < self.p_decoy:
                d = self.all_dir[r.randint(len(self.all_dir))]
                if self.cue == "lead":
                    words += LEAD[r.randint(len(LEAD))]
                    dir_at.append((len(words), "lead"))
                    words += [d] + self._after_dir(r) + [self._obj_word(o), "."]   # current cell, not scored
                else:
                    words += self._before_dir(r)
                    dir_at.append((len(words), "trail"))
                    words += [d] + TRAIL[r.randint(len(TRAIL))]
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
