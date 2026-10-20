"""Shared helpers for the language-side post hoc analyses (docs/paper/expansion/LANGUAGE.md). CPU only, eval only,
no training. Run any script in this directory from /home/prashr with PYTHONPATH=/home/prashr.

- load(ck): any text-world checkpoint (runs/textworld/p0 Vanilla_r4, runs/tw_normstep/p0 MapWM/NormStep/NormStepNB/
  DirOnly). Returns (model in eval mode, arm name).
- gen(te, T): TextWorld.generate_trajectory re-implemented with a ROLE per token (verb, adv, dir, fill, see, obj, dot,
  aside, anoun, adot) and the clause index; consumes np.random exactly as the original (checked by check_gen()).
- forward(m, tok, delta, scale, want_attn, qk_mask): the one-layer forward of MapFormerWM / _StepOverride with an
  explicit per-position step tensor; reproduces m(tok) exactly when delta = step_of(m, tok) and scale = 1 (checked by
  check_forward()). scale multiplies the attention output (the 1/(1-p) dropout re-score, TW_NORMSTEP CORRECTED block).
  qk_mask (H, n_blocks) in {0,1} removes rotary channels from the score (q and k components zeroed).
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prashr")
from mapformer.environment_textworld import TextWorld, VERBS, ADVERBS, FILLERS, SEE, OBJECTS, ASIDE, DIRS  # noqa
from mapformer.model import _apply_rope  # noqa
from mapformer.model_textstep import step_of  # noqa

REPO = "/home/prashr/mapformer"
TW = f"{REPO}/runs/textworld/p0"
NS = f"{REPO}/runs/tw_normstep/p0"
ROLES = ["verb", "adv", "dir", "fill", "see", "obj", "dot", "aside", "anoun", "adot"]
SCALE = 1.0 / 0.9


def load(ck):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    if "arm" in blob:
        from mapformer.tw_normstep_readouts import load as nsload
        m, arm = nsload(ck)
        return m.eval(), arm
    from mapformer.analyze_textworld_secondary import load as twload
    return twload(ck, "cpu").eval(), "MapWM(tw)"


def ck_tw(s):
    return f"{TW}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt"


def ck_ns(arm, s):
    return f"{NS}/{arm}_s{s}/{arm}.pt"


def gen(te, n_tokens=1024):
    """Copy of TextWorld.generate_trajectory that also returns per-token roles, clause index and the clause's
    direction (0..3) and cell. Returns dict of numpy arrays."""
    r = np.random
    steps = max(te.max_steps, n_tokens // 3)
    tok, _om, rev = te.grid.generate_trajectory(steps)
    tok = tok.numpy(); rev = rev.numpy()
    locs = list(te.grid.visited_locations)
    out, obs, rv, role, cl = [], [], [], [], []
    te.visited_locations = []
    dirs, cells = [], []
    for s in range(steps):
        a, o = int(tok[2 * s]), int(tok[2 * s + 1])
        words = [VERBS[r.randint(len(VERBS))]]; ro = ["verb"]
        if r.random() < te.p_adverb:
            words.append(ADVERBS[r.randint(len(ADVERBS))]); ro.append("adv")
        words.append(DIRS[a][r.randint(3)]); ro.append("dir")
        if r.random() < te.p_filler:
            k = r.randint(1, 4)
            words += [FILLERS[r.randint(len(FILLERS))] for _ in range(k)]; ro += ["fill"] * k
        sp = SEE[r.randint(len(SEE))]
        words += sp; ro += ["see"] * len(sp)
        obj_at = len(words)
        words += [te._obj_word(o), "."]; ro += ["obj", "dot"]
        if r.random() < te.p_aside:
            asd = ASIDE[r.randint(len(ASIDE))]
            words += asd + [OBJECTS[r.randint(te.n_obs_types)], "."]
            ro += ["aside"] * len(asd) + ["anoun", "adot"]
        flags = [False] * len(words); flags[obj_at] = True
        room = n_tokens - len(out)
        if room <= 0:
            break
        if obj_at < room:
            te.visited_locations.append(locs[s])
            rflags = [False] * len(words); rflags[obj_at] = bool(rev[2 * s + 1])
        else:
            flags = [False] * len(words); rflags = [False] * len(words)
        out += [te.idx[w] for w in words][:room]; obs += flags[:room]; rv += rflags[:room]
        role += ro[:room]; cl += [s] * min(len(words), room)
        dirs.append(a); cells.append(tuple(locs[s]))
    else:
        raise RuntimeError("did not fill")
    return {"tok": np.array(out), "obs": np.array(obs), "rev": np.array(rv),
            "role": np.array([ROLES.index(x) for x in role]), "clause": np.array(cl),
            "dirs": np.array(dirs), "cells": cells, "locs": list(te.visited_locations)}


def check_gen(n=5):
    for seed in range(n):
        te = TextWorld(size=64, seed=10000); np.random.seed(seed)
        t1, o1, r1 = te.generate_trajectory(1024)
        te2 = TextWorld(size=64, seed=10000); np.random.seed(seed)
        g = gen(te2, 1024)
        assert (t1.numpy() == g["tok"]).all() and (o1.numpy() == g["obs"]).all() and (r1.numpy() == g["rev"]).all()
    return True


def omega(m):
    return m.path_integrator.omega.detach()          # (H, nb)


@torch.no_grad()
def delta_of(m, tok):
    """(L, H, nb) raw step per position (omega not applied)."""
    t = torch.as_tensor(tok)[None]
    return step_of(m, t)[0]


@torch.no_grad()
def step_table(m, V):
    t = torch.arange(V)[None]
    return step_of(m, t)[0]                           # (V, H, nb)


@torch.no_grad()
def forward(m, tok, delta, scale=1.0, want_attn=False, qk_mask=None):
    t = torch.as_tensor(tok)[None]
    x = m.token_emb(t)
    cos_a, sin_a = m.path_integrator(delta[None])
    assert len(m.layers) == 1
    ly = m.layers[0]; B, T, _ = x.shape
    h = ly.norm1(x)
    Q = ly.q_proj(h).view(B, T, ly.n_heads, ly.d_head).transpose(1, 2)
    K = ly.k_proj(h).view(B, T, ly.n_heads, ly.d_head).transpose(1, 2)
    Vv = ly.v_proj(h).view(B, T, ly.n_heads, ly.d_head).transpose(1, 2)
    Q = _apply_rope(Q, cos_a, sin_a); K = _apply_rope(K, cos_a, sin_a)
    if qk_mask is not None:
        mk = qk_mask.repeat_interleave(2, dim=-1)[None, :, None, :]       # (1, H, 1, d_head)
        Q = Q * mk; K = K * mk
    sc = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(ly.d_head)
    cm = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    sc = sc.masked_fill(cm, float("-inf"))
    A = F.softmax(sc, dim=-1)
    o = torch.matmul(A, Vv).transpose(1, 2).reshape(B, T, ly.d_model) * scale
    x = x + ly.o_proj(o)
    x = x + ly.ffn(ly.norm2(x))
    logits = m.out_proj(m.out_norm(x))[0]
    return (logits, A[0]) if want_attn else logits


def check_forward(m, seed=0):
    te = TextWorld(size=64, seed=10000); np.random.seed(seed)
    g = gen(te, 1024); tok = g["tok"]
    with torch.no_grad():
        ref = m(torch.as_tensor(tok)[None])[0]
    got = forward(m, tok, delta_of(m, tok))
    return float((ref - got).abs().max())


def acc_on(logits, g):
    """Revisit accuracy: the prediction at position i-1 of each revisited object slot i."""
    pred = logits.argmax(-1).numpy(); idx = np.nonzero(g["rev"])[0]; idx = idx[idx > 0]
    return (pred[idx - 1] == g["tok"][idx]), idx


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi
