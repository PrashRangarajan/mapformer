"""Navigation told in words: the torus walk rendered as English (TEXTWORLD_PREREG.md).

The walk, the map and the revisit bookkeeping are GridWorld's (64x64 torus, K observation types,
p_empty blanking, the directed walk). Each step is rendered as a short clause of WORD tokens:

    [verb] [adverb?] [direction] [filler clause?] [seeing phrase] [object] [.]
    e.g.  walked slowly north then paused and saw a lamp .
          headed left and found nothing .

Direction words carry the action, with synonyms: north = {north, up, northward}, and so on for
south / west / east. Everything else (verbs, adverbs, fillers, seeing phrases, the period) is
movement-free text. With probability p_aside an extra sentence is inserted between steps that
mentions an object noun WITHOUT a location ("she thought about a cat ."). No direction word ever
appears outside a movement clause: MapFormer's step Delta is a function of the token alone
(model.py: delta = action_to_lie(token_emb(x))), so a direction word in a non-movement sense would
move the phase by construction. That case is a separate question.

The loss / readout target is the OBJECT token of each step taken at a revisited cell ("nothing" for
a blank cell), as on the torus. Sequences are exactly `n_tokens` long: steps are rendered until the
budget is filled and the last one is cut; a cut step contributes no target.
"""
import numpy as np
import torch

from mapformer.environment import GridWorld

DIRS = {0: ["north", "up", "northward"], 1: ["south", "down", "southward"],
        2: ["west", "left", "westward"], 3: ["east", "right", "eastward"]}   # GridWorld action ids
VERBS = ["walked", "went", "moved", "stepped", "headed", "wandered"]
ADVERBS = ["slowly", "quickly", "carefully", "quietly"]
FILLERS = ["then", "paused", "looked", "around", "for", "a", "moment", "and", "she"]
SEE = [["and", "saw"], ["and", "found"], ["and", "noticed"], ["there", "was"]]
OBJECTS = ["lamp", "cat", "chair", "tree", "key", "book", "cup", "bell",
           "coin", "shoe", "rope", "stone", "hat", "clock", "box", "pen"]
ASIDE = [["she", "thought", "about", "a"], ["she", "remembered", "a"], ["a", "story", "about", "a"]]


class TextWorld:
    def __init__(self, size=64, n_obs_types=16, p_empty=0.5, seed=None,
                 p_adverb=0.3, p_filler=0.3, p_aside=0.15, max_steps=400):
        assert n_obs_types <= len(OBJECTS), n_obs_types
        self.grid = GridWorld(size=size, n_obs_types=n_obs_types, p_empty=p_empty, seed=seed)
        self.p_adverb, self.p_filler, self.p_aside, self.max_steps = p_adverb, p_filler, p_aside, max_steps
        words = ["."] + VERBS + ADVERBS + FILLERS + [w for p in SEE for w in p] + ["nothing"] \
            + OBJECTS[:n_obs_types] + [w for p in ASIDE for w in p] + [w for d in DIRS.values() for w in d]
        self.vocab = list(dict.fromkeys(words))            # ordered, de-duplicated ("and", "a", "she")
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.unified_vocab_size = len(self.vocab)
        self.N_ACTIONS = 4                                   # train.py reads it (action noise unused here)
        self.n_obs_types = n_obs_types
        self.dir_ids = {a: [self.idx[w] for w in ws] for a, ws in DIRS.items()}
        self.obj_ids = [self.idx["nothing"]] + [self.idx[w] for w in OBJECTS[:n_obs_types]]
        self.visited_locations = []

    def _obj_word(self, obs_tok):
        k = obs_tok - self.grid.obs_offset                   # 0..K-1 an object, K = blank
        return "nothing" if k >= self.n_obs_types else OBJECTS[k]

    def generate_trajectory(self, n_tokens=1024):
        """-> (tokens, obs_mask, revisit_mask), each of length n_tokens. obs_mask marks every object
        slot, revisit_mask the object slots at revisited cells."""
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)             # a step renders to >= 5 tokens
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        out, obs, rv = [], [], []
        locs = list(self.grid.visited_locations)
        self.visited_locations = []
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
            flags = [False] * len(words); flags[obj_at] = True
            room = n_tokens - len(out)
            if room <= 0:
                break
            if obj_at < room:                                # the object slot fits: a real target
                self.visited_locations.append(locs[s])
                rflags = [False] * len(words); rflags[obj_at] = bool(rev[2 * s + 1])
            else:
                flags = [False] * len(words); rflags = [False] * len(words)
            out += [self.idx[w] for w in words][:room]; obs += flags[:room]; rv += rflags[:room]
        else:
            raise RuntimeError(f"{steps} steps did not fill n_tokens={n_tokens}")
        return (torch.tensor(out, dtype=torch.long), torch.tensor(obs, dtype=torch.bool),
                torch.tensor(rv, dtype=torch.bool))

    def generate_batch(self, batch_size, n_steps=1024, p_transition_noise=0.0):
        """train.py's contract; `n_steps` is the TOKEN length here."""
        assert p_transition_noise == 0.0
        T, M, R, L = [], [], [], []
        for _ in range(batch_size):
            t, m, rv = self.generate_trajectory(n_steps)
            T.append(t); M.append(m); R.append(rv); L.append(list(self.visited_locations))
        return torch.stack(T), torch.stack(M), torch.stack(R), L
