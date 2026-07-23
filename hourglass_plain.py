"""
Plain-layer Hourglass scaffold, for the SCAFFOLD SANITY-CHECK only.

Purpose: before reading anything into a MapFormer-in-Hourglass result, confirm
that MY shorten/upsample scaffold reproduces Hourglass's own published
efficiency property on the task it is confirmed to work on (enwik8 char-level
LM, Nawrot et al. 2021). This file uses ordinary causal transformer layers
(RoPE attention) so the ONLY thing under test is the hierarchy scaffold, not
MapFormer.

It reuses the exact `_causal_shorten` / `_upsample` operators from
model_hourglass.py (the ones whose causality is numerically verified in
test_hourglass_causal.py), so a positive result here transfers directly to the
MapFormer variant that shares those ops.

Reference config (lucidrains train.py, enwik8):
    num_tokens=256, dim=512, depth=(4,2,4), shorten_factor=2,
    heads=8, seq_len=512, causal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_hourglass import _causal_shorten, _upsample


def _rope_cos_sin(seq_len, d_head, device, base=10000.0):
    """Standard RoPE angles for positions 0..seq_len-1 (per-resolution)."""
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    ang = torch.outer(pos, inv_freq)                    # (seq_len, half)
    return ang.cos(), ang.sin()                         # (seq_len, half)


def _apply_rope(x, cos, sin):
    """x: (B, H, T, d_head); cos/sin: (T, d_head//2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).reshape_as(x)


class PlainLayer(nn.Module):
    """Pre-norm causal self-attention (RoPE) + FFN. Ordinary transformer."""

    def __init__(self, dim, heads, dropout=0.0, ff_mult=4):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(),
                                nn.Linear(ff_mult * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = _rope_cos_sin(T, self.dh, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float('-inf'))
        a = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(a, v).transpose(1, 2).reshape(B, T, self.dim)
        x = x + self.drop(self.o(out))
        x = x + self.ff(self.norm2(x))
        return x


class HourglassPlainLM(nn.Module):
    """Single-level plain Hourglass LM. depth=(pre, valley, post)."""

    def __init__(self, num_tokens=256, dim=512, depth=(4, 2, 4),
                 shorten_factor=2, heads=8, dropout=0.0):
        super().__init__()
        self.k = shorten_factor
        n_pre, n_val, n_post = depth
        self.emb = nn.Embedding(num_tokens, dim)
        self.pre = nn.ModuleList([PlainLayer(dim, heads, dropout) for _ in range(n_pre)])
        self.valley = nn.ModuleList([PlainLayer(dim, heads, dropout) for _ in range(n_val)])
        self.post = nn.ModuleList([PlainLayer(dim, heads, dropout) for _ in range(n_post)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens)

    def forward(self, x):
        B, L = x.shape
        k = self.k
        x = self.emb(x)
        for layer in self.pre:
            x = layer(x)
        skip = x
        pad = (-L) % k
        if pad:
            skip_p = F.pad(skip, (0, 0, 0, pad))
        else:
            skip_p = skip
        xc = _causal_shorten(skip_p, k)
        for layer in self.valley:
            xc = layer(xc)
        up = _upsample(xc, k)[:, :L]
        x = skip + up
        for layer in self.post:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    def flops_proxy(self, L):
        """Relative attention-FLOP proxy: sum of T^2 over layers (dominant
        term). Naive pool/upsample add ~0 FLOPs."""
        k = self.k
        Lc = (L + (-L) % k) // k
        return (len(self.pre) + len(self.post)) * L * L + len(self.valley) * Lc * Lc


class FlatPlainLM(nn.Module):
    """Flat transformer baseline (all layers full resolution)."""

    def __init__(self, num_tokens=256, dim=512, n_layers=9, heads=8, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([PlainLayer(dim, heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    def flops_proxy(self, L):
        return len(self.layers) * L * L
