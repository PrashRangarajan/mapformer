"""The text world where the SAME direction words are sometimes actions and sometimes observed content
(TW_AMBIG_PREREG.md). Walk, map, movement-clause grammar, asides and revisit targets are TextWorld's.

Two additions, both drawn per step and only when p_nm > 0 (at p_nm = 0 no extra random number is drawn, no word
is added, and the token stream is byte-identical to TextWorld's; checked in docs/audits/2026-10-08/tw_ambig_checks.py):

1. Movement clauses in two extra FRAMES (each with probability p_nm/6 per step; base clause otherwise), so
   that a non-movement use in the same frame has the same local words around its direction word:
     LEAD   near:  she PAD <mcue> <verb> [adv] <dir> [fillers] <and saw> <obj> .     cue 2-3 tokens before <dir>
            far:   she <mcue> PAD <verb> [adv] <dir> [fillers] <and saw> <obj> .     cue 7-12 before
            <mcue> in {next, soon, finally}
     TRAIL  near:  later , she <verb> [adv] <dir> <mcue> PAD <saw> <obj> .            cue 1 token after <dir>
            far:   later , she <verb> [adv] <dir> PAD <mcue> <saw> <obj> .            cue 6-10 after
            <mcue> in {so, thus, hence}
   The tokens are the same in near and far; only the order of PAD and cue changes (environment_textworld_ctx3).

2. After the step (and its optional aside), with probability p_nm, ONE NON-MOVEMENT sentence that uses a
   direction word (any of the 12 synonyms, uniformly) without moving the walker. Classes (shares of p_nm):
     nat (1/3)        observation content, cue 1-2 tokens away:  she saw a sign pointing <dir> .
                      / she heard a sound from the <dir> .  / a cold <dir> wind blew .
     lead_near (1/6)  she PAD <ncue> <verb> [adv] <dir> [fillers] <and saw> <obj> .   <ncue> in {dreamt, imagined, recalled}
     lead_far  (1/6)  she <ncue> PAD <verb> [adv] <dir> [fillers] <and saw> <obj> .
     trail_near (1/6) later , she <verb> [adv] <dir> <ncue> PAD the sign .           <ncue> in {said, read, showed}
     trail_far (1/6)  later , she <verb> [adv] <dir> PAD <ncue> the sign .
   In the lead forms the reported clause's object is the CURRENT cell's (the walker did not move, so it is
   true; never scored). In the far forms the 4 tokens before and the 3 after the direction word are drawn
   exactly as in the movement clause of the same frame: only a cue 6-12 tokens away decides the role.

`self.ctx` lists (token index, role, form) for every direction word (role "move" / "nm"; form "base",
"lead_near", ..., "nat"); `self.nm_mask` marks every position of a non-movement sentence.

tag_roles=True emits a non-movement direction word as id V + k (V = unified_vocab_size, k its index in
ALL_DIR) instead of its word id; the random draws are identical, so the stream equals the untagged one after
mapping V + k -> ALL_DIR[k]. Only the oracle arms (model_tw_ambig.RoleTag / DirOnlyRole) read tagged streams.
"""
import numpy as np
import torch

from mapformer.environment_textworld import TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE, OBJECTS, ASIDE
from mapformer.environment_textworld_ctx3 import PADS, SEE1

ALL_DIR = [w for a in range(4) for w in DIRS[a]]
LEAD_MOVE, LEAD_NM = ["next", "soon", "finally"], ["dreamt", "imagined", "recalled"]
TRAIL_MOVE, TRAIL_NM = ["so", "thus", "hence"], ["said", "read", "showed"]
NAT = [(["she", "saw", "a", "sign", "pointing"], []), (["she", "heard", "a", "sound", "from", "the"], []),
       (["a", "cold"], ["wind", "blew"])]
FORMS = ["lead_near", "lead_far", "trail_near", "trail_far"]
NM_CLASSES = ["nat"] + FORMS
NM_SHARE = {"nat": 1 / 3, "lead_near": 1 / 6, "lead_far": 1 / 6, "trail_near": 1 / 6, "trail_far": 1 / 6}


class TextWorldAmbig(TextWorld):
    def __init__(self, *a, p_nm=0.3, tag_roles=False, **kw):
        super().__init__(*a, **kw)
        self.p_nm, self.tag_roles = p_nm, tag_roles
        extra = [w for p in PADS for w in p] + LEAD_MOVE + LEAD_NM + TRAIL_MOVE + TRAIL_NM + SEE1 \
            + ["later", ",", "the", "sign"] + [w for pre, post in NAT for w in pre + post]
        for w in (extra if p_nm > 0 else []):          # at p_nm = 0 the task, vocabulary included, IS TextWorld
            if w not in self.idx:
                self.idx[w] = len(self.vocab); self.vocab.append(w)
        self.unified_vocab_size = len(self.vocab)
        self.dir_word_ids = [self.idx[w] for w in ALL_DIR]
        self.tag_offset = self.unified_vocab_size          # tagged id = tag_offset + k
        self.ctx, self.nm_mask = [], []
        self._cum = np.cumsum([NM_SHARE[c] for c in NM_CLASSES])

    # ------------------------------------------------------------------ clause builders
    def _verb(self, r):
        return [VERBS[r.randint(len(VERBS))]] + ([ADVERBS[r.randint(len(ADVERBS))]] if r.random() < self.p_adverb else [])

    def _tail(self, r):
        fill = [FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))] if r.random() < self.p_filler else []
        return fill + SEE[r.randint(len(SEE))]

    def _framed(self, r, form, nm, d, obj_word):
        """A LEAD / TRAIL frame clause -> (words, index of <dir>, index of the object slot or None)."""
        pad = PADS[r.randint(len(PADS))]
        if form.startswith("lead"):
            c = (LEAD_NM if nm else LEAD_MOVE)[r.randint(3)]
            head = ["she"] + (pad + [c] if form == "lead_near" else [c] + pad) + self._verb(r)
            words = head + [d] + self._tail(r)
            obj_at = len(words)
            return words + [obj_word, "."], len(head), obj_at
        c = (TRAIL_NM if nm else TRAIL_MOVE)[r.randint(3)]
        head = ["later", ",", "she"] + self._verb(r)
        mid = [c] + pad if form == "trail_near" else pad + [c]
        if nm:
            return head + [d] + mid + ["the", "sign", "."], len(head), None
        words = head + [d] + mid + [SEE1[r.randint(3)]]
        return words + [obj_word, "."], len(head), len(words)

    def _base(self, r, a, obj_word):
        """TextWorld's movement clause, drawing exactly TextWorld.generate_trajectory's random numbers in its
        order (verb, adverb, direction synonym, fillers, seeing phrase)."""
        words = [VERBS[r.randint(len(VERBS))]]
        if r.random() < self.p_adverb:
            words.append(ADVERBS[r.randint(len(ADVERBS))])
        di = len(words); words.append(DIRS[a][r.randint(3)])
        if r.random() < self.p_filler:
            words += [FILLERS[r.randint(len(FILLERS))] for _ in range(r.randint(1, 4))]
        words += SEE[r.randint(len(SEE))]
        obj_at = len(words)
        return words + [obj_word, "."], di, obj_at

    def _nm(self, r, obj_word):
        """One non-movement sentence -> (words, index of <dir>, class)."""
        cls = NM_CLASSES[int(np.searchsorted(self._cum, r.random() * self._cum[-1], side="right"))]
        d = ALL_DIR[r.randint(len(ALL_DIR))]
        if cls == "nat":
            pre, post = NAT[r.randint(len(NAT))]
            return pre + [d] + post + ["."], len(pre), cls
        words, di, _ = self._framed(r, cls, True, d, obj_word)
        return words, di, cls

    # ------------------------------------------------------------------ trajectory
    def generate_trajectory(self, n_tokens=1024):
        r = np.random
        steps = max(self.max_steps, n_tokens // 3)
        tok, _om, rev = self.grid.generate_trajectory(steps)
        tok = tok.numpy(); rev = rev.numpy()
        locs = list(self.grid.visited_locations)
        out, obs, rv, nmm, self.visited_locations, self.ctx = [], [], [], [], [], []
        q = self.p_nm / 6
        for s in range(steps):
            a, o = int(tok[2 * s]), int(tok[2 * s + 1])
            ow = self._obj_word(o)
            form = "base"
            if self.p_nm > 0:
                u = r.random()
                if u < 4 * q:
                    form = FORMS[int(u // q)]
            if form == "base":
                words, di, obj_at = self._base(r, a, ow)
            else:
                words, di, obj_at = self._framed(r, form, False, DIRS[a][r.randint(3)], ow)
            dirs = [(di, "move", form)]
            nm = [False] * len(words)
            if r.random() < self.p_aside:
                add = ASIDE[r.randint(len(ASIDE))] + [OBJECTS[r.randint(self.n_obs_types)], "."]
                words += add; nm += [False] * len(add)
            if self.p_nm > 0 and r.random() < self.p_nm:
                nw, ndi, cls = self._nm(r, ow)
                dirs.append((len(words) + ndi, "nm", cls)); words += nw; nm += [True] * len(nw)
            room = n_tokens - len(out)
            if room <= 0:
                break
            flags = [False] * len(words); rflags = [False] * len(words)
            if obj_at < room:
                self.visited_locations.append(locs[s])
                flags[obj_at] = True; rflags[obj_at] = bool(rev[2 * s + 1])
            base = len(out)
            ids = [self.idx[w] for w in words]
            for i, role, f in dirs:
                if role == "nm" and self.tag_roles:
                    ids[i] = self.tag_offset + ALL_DIR.index(words[i])
                if i < room:
                    self.ctx.append((base + i, role, f))
            out += ids[:room]; obs += flags[:room]; rv += rflags[:room]; nmm += nm[:room]
        else:
            raise RuntimeError(f"{steps} steps did not fill n_tokens={n_tokens}")
        self.nm_mask = nmm
        return (torch.tensor(out, dtype=torch.long), torch.tensor(obs, dtype=torch.bool),
                torch.tensor(rv, dtype=torch.bool))

    def untag(self, tokens):
        """Map tagged ids back to word ids (identity on untagged streams)."""
        t = tokens.clone(); m = t >= self.tag_offset
        t[m] = torch.tensor(self.dir_word_ids)[t[m] - self.tag_offset]
        return t
