"""Shared helpers for the depth-exchange-rate analyses (CPU, inference only, stored checkpoints; post hoc).

forward_capture() re-implements WMTransformerLayer.forward (manual attention path, eval mode, dropout off) for the
index-RoPE models (1..4 layers, or one block looped n_loops times) and for the path model (MapFormerWM / _r4), so it can
return every pass's residual stream and attention matrix. check_logits() asserts the re-implementation reproduces the
model's own logits.
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prashr")
from mapformer.model import _apply_rope  # noqa: E402

REPO = "/home/prashr/mapformer"
OUTDIR = f"{REPO}/docs/paper/expansion/scripts/depth/out"


def cos_sin(m, tokens):
    """(cos, sin) of shape (B, H, L, n_blocks) and the list of blocks to apply, in order."""
    B, L = tokens.shape
    if hasattr(m, "_rope_cos_sin"):                       # index RoPE (plain or looped)
        x = m.token_emb(tokens)
        c, s = m._rope_cos_sin(L, tokens.device, x.dtype)
        c = c.expand(B, -1, -1, -1); s = s.expand(B, -1, -1, -1)
    else:                                                  # path-integrated MapWM
        c, s = m.get_position_state(tokens)
    n_loops = getattr(m, "n_loops", None)
    blocks = [m.layers[0]] * n_loops if (n_loops and len(m.layers) == 1 and type(m).__name__.endswith("Looped")) \
        else list(m.layers)
    return c, s, blocks


@torch.no_grad()
def forward_capture(m, tokens, keep_attn=True):
    """Returns logits, residuals [x_0 (embedding), x_1, ..., x_P] and attentions [(B,H,L,L) per pass]."""
    m.eval()
    B, L = tokens.shape
    c, s, blocks = cos_sin(m, tokens)
    x = m.token_emb(tokens)
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    res, att = [x], []
    for blk in blocks:
        h = blk.norm1(x)
        Q = blk.q_proj(h).view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
        K = blk.k_proj(h).view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
        V = blk.v_proj(h).view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
        Q = _apply_rope(Q, c, s); K = _apply_rope(K, c, s)
        sc = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(blk.d_head)
        sc = sc.masked_fill(mask, float("-inf"))
        A = F.softmax(sc, dim=-1)
        o = torch.matmul(A, V).transpose(1, 2).reshape(B, L, blk.d_model)
        x = x + blk.o_proj(o)
        x = x + blk.ffn(blk.norm2(x))
        res.append(x)
        if keep_attn:
            att.append(A)
    logits = m.out_proj(m.out_norm(x))
    return logits, res, att


@torch.no_grad()
def check_logits(m, tokens, tol=2e-4):
    m.eval()
    a = m(tokens); b = forward_capture(m, tokens, keep_attn=False)[0]
    d = float((a - b).abs().max())
    assert d < tol, d
    return d


def runs_info(moves, opposite):
    """Per move t: cur (length of the current run up to t), prev_run (length of the previous run if the current run
    reverses it, else 0), run_id. opposite(a, b) -> bool."""
    T = len(moves); cur = np.zeros(T, int); prev = np.zeros(T, int); rid = np.zeros(T, int)
    c = p = 0; r = 0; last_len = 0
    for t in range(T):
        if t > 0 and moves[t] == moves[t - 1]:
            c += 1
        else:
            if t > 0:
                last_len = c; r += 1
            p = last_len if (t > 0 and opposite(moves[t], moves[t - 1])) else 0
            c = 1
        cur[t] = c; prev[t] = p; rid[t] = r
    return cur, prev, rid


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-2):
    """Ridge regression (standardised X, intercept), out-of-sample R^2 per target column."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = Ytr.mean(0)
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ (Ytr - ym))
    P = Xte @ W + ym
    ss = ((Yte - P) ** 2).sum(0); st = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return 1 - ss / st
