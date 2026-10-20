"""DELTA_FREE (DELTA_FREE_PREREG.md): PoPE's phase bias delta, unpinned.

The faithful PoPE layer (`model_pope.WMTransformerLayer_PoPE`) zero-initialises delta and clamps it to
[DELTA_MIN, DELTA_MAX] = [-2 pi, 0] in the forward pass. Measured on every PoPE-Flat run of PAPER2X2, 98-100% of
entries sit in the clamp's dead zone (delta >= 0: clamped to 0, zero gradient), so in practice beta = 0 and every
head's kernel peaks at zero phase difference (docs/paper/expansion/MAPPOPE.md, Proposition 1: "peak pinning").
These classes remove the pin, and nothing else:

  PoPE-FreeDelta       index phase (= PoPE-Flat), delta ~ U(-2 pi, 0), NO clamp, learnable
  MapPoPE-FreeDelta    path phase (= MapPoPE-Flat, r=2), delta ~ U(-2 pi, 0), NO clamp, learnable
  MapPoPE-FrozenDelta  path phase, delta ~ U(-2 pi, 0), NO clamp, requires_grad False (no route back to 0;
                       AdamW skips a parameter whose grad is None, so weight decay does not touch it either)

Pairing by construction (checked in analyze_delta_free.py --selftest):
  * every parameter other than delta is BITWISE the pinned twin's at the same torch seed: the parent class is built
    first, unchanged; the layer is re-classed in place (no new module is constructed, so the global torch RNG is not
    consumed); delta is drawn from a PRIVATE generator seeded from torch.initial_seed() (= --seed after
    train_variant's torch.manual_seed), so the global stream -- and with it dropout -- is untouched at the start;
  * the delta draw depends only on the seed and the shape (n_heads, d_head), so PoPE-FreeDelta, MapPoPE-FreeDelta
    and MapPoPE-FrozenDelta start from the same delta values at a seed;
  * the initial draw is kept as a buffer `pope_delta_init` (in the state dict) so a checkpoint records how far
    delta moved; the frozen arm must end with pope_delta == pope_delta_init exactly.
Unclamped delta is harmless numerically: it enters only through cos / sin, which are 2 pi periodic.

The existing classes (PoPE-Flat, MapPoPE-Flat) are NOT modified; their training reproduces HEAD bitwise
(DELTA_FREE_PREREG.md, "Bitwise check").
"""
import math

import torch
import torch.nn.functional as F

from .model_pope import (WMTransformerLayer_PoPE, MapFormerWM_PoPE, MapFormerWM_RoPEIndex_PoPE,
                         DELTA_MIN, DELTA_MAX)

DELTA_GEN_OFFSET = 1_000_000_007          # private generator seed = torch.initial_seed() + this (mod 2**63)


class WMTransformerLayer_PoPE_FreeDelta(WMTransformerLayer_PoPE):
    """WMTransformerLayer_PoPE.forward with the clamp on delta removed; nothing else differs.
    No __init__ of its own: instances are made by re-classing a constructed WMTransformerLayer_PoPE."""

    def delta_eff(self):
        """The delta the forward pass uses (raw). The pinned parent's is the clamped value."""
        return self.pope_delta

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        mq, mk = F.softplus(Q), F.softplus(K)
        d = self.pope_delta.view(1, self.n_heads, 1, -1)          # the ONLY change: no clamp
        cd, sd = torch.cos(d), torch.sin(d)
        cosK = cos_a * cd - sin_a * sd
        sinK = sin_a * cd + cos_a * sd

        scores = (torch.matmul(mq * cos_a, (mk * cosK).transpose(-1, -2))
                  + torch.matmul(mq * sin_a, (mk * sinK).transpose(-1, -2))) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


def delta_eff(layer):
    """Effective delta of any PoPE layer: raw for the free classes, the clamp for the faithful one."""
    if hasattr(layer, "delta_eff"):
        return layer.delta_eff()
    return layer.pope_delta.clamp(DELTA_MIN, DELTA_MAX)


def _unpin(model, frozen):
    """Re-class every PoPE layer to the unclamped forward and draw delta ~ U(-2 pi, 0) from a private generator."""
    gen = torch.Generator().manual_seed((torch.initial_seed() + DELTA_GEN_OFFSET) % (2 ** 63))
    n = 0
    for layer in model.layers:
        assert type(layer) is WMTransformerLayer_PoPE, type(layer)
        layer.__class__ = WMTransformerLayer_PoPE_FreeDelta
        with torch.no_grad():
            init = torch.empty_like(layer.pope_delta).uniform_(DELTA_MIN, DELTA_MAX, generator=gen)
            layer.pope_delta.copy_(init)
        layer.register_buffer("pope_delta_init", init.clone())
        if frozen:
            layer.pope_delta.requires_grad_(False)
        n += 1
    assert n > 0, "no PoPE layer to unpin"
    return model


class MapFormerWM_RoPEIndex_PoPE_FreeDelta(MapFormerWM_RoPEIndex_PoPE):
    """PoPE-Flat (index phase) with delta ~ U(-2 pi, 0), unclamped, learnable."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _unpin(self, frozen=False)


class MapFormerWM_PoPE_FreeDelta(MapFormerWM_PoPE):
    """MapPoPE-Flat (path phase, r=2) with delta ~ U(-2 pi, 0), unclamped, learnable."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _unpin(self, frozen=False)


class MapFormerWM_PoPE_FrozenDelta(MapFormerWM_PoPE):
    """MapPoPE-Flat (path phase, r=2) with delta ~ U(-2 pi, 0), unclamped, FROZEN (no gradient, no weight decay)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _unpin(self, frozen=True)
