"""PoPE's own component ablation (their Table 5), plus the extrapolation test it cannot do.

The paper ablates sigma() and delta and reports OpenWebText perplexity:

    PoPE without sigma()      21.57 / 18.93     (124M / 253M)
    PoPE with ReLU for sigma  21.55 / 18.90
    PoPE without delta        21.42 / 18.57
    Full PoPE                 21.33 / 18.55

That table is IN DISTRIBUTION. The account this repo needs to test is about OUT of
distribution: PoPE's magnitudes are non-negative, so

    |score(Delta)| = |sum_c mu_q mu_k cos(w_c Delta - d_c)| <= sum_c mu_q mu_k = score(0)

for EVERY Delta, seen or unseen. RoPE's signed Q.K has no such bound. If that is why
the PoPE-encoding arms survive past their training context, then:

  - NoSigma (signed magnitudes) should LOSE the extrapolation advantage; and
  - **ReLU should KEEP it while staying worse in distribution.**

ReLU is the discriminating arm: it is equally non-negative, hence equally bounded, but
the paper shows it behaves like NoSigma in distribution. If bounding is what matters
out of distribution, ReLU extrapolates like full PoPE despite that. If ReLU
extrapolates badly, the bound account is dead.

NoDelta freezes the per-head phase bias at zero -- their smaller ablation, included so
the replication of Table 5 is complete rather than selective.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_pope import (WMTransformerLayer_PoPE, MapFormerWM_RoPEIndex_PoPE,
                         DELTA_MIN, DELTA_MAX)

MAGS = {"softplus": F.softplus, "identity": lambda z: z, "relu": F.relu}


class WMTransformerLayer_PoPE_Ablate(WMTransformerLayer_PoPE):
    """PoPE with a swappable magnitude function and an optionally frozen delta."""

    def __init__(self, d_model, n_heads, dropout, mag="softplus", use_delta=True):
        super().__init__(d_model, n_heads, dropout)
        self.mag_name, self.use_delta = mag, use_delta
        if not use_delta:
            self.pope_delta.requires_grad_(False)     # frozen at its zero init

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        sh = lambda z: z.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = sh(self.q_proj(h)), sh(self.k_proj(h)), sh(self.v_proj(h))
        f = MAGS[self.mag_name]
        mq, mk = f(Q), f(K)
        if self.use_delta:
            d = self.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, self.n_heads, 1, -1)
            cd, sd = torch.cos(d), torch.sin(d)
            cosK, sinK = cos_a * cd - sin_a * sd, sin_a * cd + cos_a * sd
        else:
            cosK, sinK = cos_a, sin_a                 # delta == 0 exactly
        scores = (torch.matmul(mq * cos_a, (mk * cosK).transpose(-1, -2))
                  + torch.matmul(mq * sin_a, (mk * sinK).transpose(-1, -2))) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), V)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, self.d_model))
        x = x + self.dropout(out)
        return x + self.ffn(self.norm2(x))


class _PoPEAblate(MapFormerWM_RoPEIndex_PoPE):
    MAG, USE_DELTA = "softplus", True

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, base=10000.0, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, base)
        drop = self.layers[0].dropout.p if len(self.layers) else dropout
        self.layers = nn.ModuleList([
            WMTransformerLayer_PoPE_Ablate(d_model, n_heads, drop,
                                           mag=self.MAG, use_delta=self.USE_DELTA)
            for _ in self.layers])


class PoPE_Full(_PoPEAblate):
    """Control: identical to MapFormerWM_RoPEIndex_PoPE, built through this class."""
    MAG, USE_DELTA = "softplus", True


class PoPE_NoSigma(_PoPEAblate):
    """magnitude = content, signed -- removes the bound."""
    MAG, USE_DELTA = "identity", True


class PoPE_ReLU(_PoPEAblate):
    """magnitude = relu(content): still non-negative, hence still bounded."""
    MAG, USE_DELTA = "relu", True


class PoPE_NoDelta(_PoPEAblate):
    """delta frozen at zero."""
    MAG, USE_DELTA = "softplus", False
