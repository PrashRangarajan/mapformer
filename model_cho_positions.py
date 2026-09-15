"""SAMEBLOCK_PREREG.md: every position mechanism inside Cho et al.'s block (`model_cho_coupled.py`), so the only
difference between arms is how position enters attention.

  coupled   learned absolute embeddings over coupled position IDs (the oracle; Cho et al.)
  rope      index RoPE on queries and keys (base 10000)
  nope      no position information
  signed    MapFormer path integration: Delta = W_out W_in x (rank 4) per head and block, theta = omega * cumsum(Delta),
            omega learned with MapFormer's geometric init; queries and keys rotated by theta (MapWM style)
  abs       the same with |Delta| (monotone increments)

The increment reads the token embedding, as in MapFormer. Block otherwise identical for all arms: GEGLU FFN,
RMSNorm pre and post (sandwich reading), no dropout.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model_cho_coupled import RMSNorm


def rotate(x, cos, sin):
    """x: (B, H, L, dh) with dh = 2 nb; cos/sin: (B or 1, H or 1, L, nb). Pairs (even, odd)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class RotBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, d_ff, bias=False); self.v = nn.Linear(d, d_ff, bias=False); self.w2 = nn.Linear(d_ff, d, bias=False)
        self.pre1, self.post1, self.pre2, self.post2 = RMSNorm(d), RMSNorm(d), RMSNorm(d), RMSNorm(d)

    def forward(self, x, cos=None, sin=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.pre1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, L, self.h, D // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if cos is not None:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.post1(self.o(a.transpose(1, 2).reshape(B, L, D)))
        y = self.pre2(x)
        return x + self.post2(self.w2(F.gelu(self.w1(y)) * self.v(y)))


class ChoPositions(nn.Module):
    MODE = "nope"

    def __init__(self, vocab_size, d_model=512, n_heads=4, n_layers=1, grid_size=64, max_pos=202, d_ff=2048,
                 bottleneck_r=4, **kw):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.nb = self.dh // 2
        self.max_pos = max_pos
        self.wants_pos_ids = self.MODE == "coupled"
        self.tok = nn.Embedding(vocab_size, d_model)
        if self.MODE == "coupled":
            self.pos = nn.Embedding(max_pos + 2, d_model)
        if self.MODE == "rope":
            self.register_buffer("inv_freq", 10000.0 ** (-torch.arange(self.nb, dtype=torch.float32) / self.nb))
        if self.MODE in ("signed", "abs"):
            self.w_in = nn.Linear(d_model, bottleneck_r, bias=False)
            self.w_out = nn.Linear(bottleneck_r, n_heads * self.nb, bias=False)
            frac = torch.arange(self.nb, dtype=torch.float32) / (self.nb - 1)
            self.omega = nn.Parameter((2 * math.pi * (1.0 / grid_size) ** frac).repeat(n_heads, 1))   # MapFormer init
        self.blocks = nn.ModuleList([RotBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model); self.head = nn.Linear(d_model, vocab_size, bias=False)

    def angles(self, tokens, e):
        B, L = tokens.shape
        if self.MODE == "rope":
            ang = torch.arange(L, device=tokens.device, dtype=e.dtype)[:, None] * self.inv_freq.to(e.dtype)
            return ang.cos()[None, None], ang.sin()[None, None]
        if self.MODE in ("signed", "abs"):
            d = self.w_out(self.w_in(e)).view(B, L, self.h, self.nb)
            if self.MODE == "abs":
                d = d.abs()
            ang = (torch.cumsum(d, dim=1) * self.omega).transpose(1, 2)            # (B, H, L, nb)
            return ang.cos(), ang.sin()
        return None, None

    def forward(self, tokens, pos_ids=None):
        B, L = tokens.shape
        e = self.tok(tokens)
        x = e + self.pos(pos_ids[:, :L].clamp(max=self.max_pos + 1)) if self.MODE == "coupled" else e
        cos, sin = self.angles(tokens, e)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))


def _mk(mode):
    return type(f"ChoPos_{mode}", (ChoPositions,), {"MODE": mode})


ChoPos_coupled, ChoPos_rope, ChoPos_nope, ChoPos_signed, ChoPos_abs = (_mk(m) for m in ("coupled", "rope", "nope", "signed", "abs"))
