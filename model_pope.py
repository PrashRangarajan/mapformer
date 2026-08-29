"""PoPE (Polar Coordinate Positional Embeddings, arXiv:2509.10534) for MapFormer.

Faithful to the paper, verbatim references:

  eq.3  mu_k = sigma(k),  mu_q = sigma(q),  sigma(x) = ln(1+e^x)   [softplus]
  eq.4  phi_k = s*theta_c,  phi_q = t*theta_c,
        "theta_c is a component-specific frequency, i.e. theta_c = theta^{(c-1)/d}"
  eq.6  a_ts = sum_{c=1}^{d} mu_q_tc mu_k_sc cos((s-t) theta_c + delta_c)
  eq.8  x_k = mu_k cos(phi_k + delta_c),  y_k = mu_k sin(phi_k + delta_c)

Key property (paper, sec.3): "c is an index over individual elements of the key
and query and not over pairs of elements, thereby DOUBLING THE NUMBER OF
FREQUENCIES FROM d/2 TO d". So PoPE has one frequency, one magnitude and one
phase PER ELEMENT -- not per 2-D pair as in RoPE.

delta_c: "a learnable bias that tunes the optimal relative offset for each
frequency c". Init "either with delta_c = 0 or delta_c ~ Uniform(-2pi, 0)";
"we bound delta_c so that it always lies in the interval [-2pi, 0] ... and found
this improves stability"; "the zero initialization is important for length
generalization". We use zero-init + the [-2pi, 0] clamp.

DOCUMENTED INTERPRETATION (raised with the user, not silently resolved):
the paper prints theta_c = theta^{(c-1)/d} with a POSITIVE exponent, which would
make frequencies increase to ~1e4 rad/position and alias at every step. RoPE's,
given in the same paper, is theta^{-2(c-1)/d} (negative). Since the stated intent
is to double the frequency COUNT over the same range, we read this as a sign typo
and use the decreasing schedule -- which is also what this repo's PathIntegrator
already does (its docstring notes the identical typo in the MapFormer paper).

Earlier versions of this file used d/2 frequencies (reusing MapFormer's block
structure) and an unclamped delta. Both are corrected here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, MapFormerWM, ActionToLieAlgebra, PathIntegrator
from .model_hourglass import MapFormerWM_Hourglass_k2, MapFormerWM_Hourglass_CoarseIdx

DELTA_MIN, DELTA_MAX = -2.0 * math.pi, 0.0     # paper: bound delta_c to [-2pi, 0]


class WMTransformerLayer_PoPE(WMTransformerLayer):
    """PoPE attention. cos_a/sin_a carry ONE angle PER ELEMENT: (B, H, T, d_head)."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__(d_model, n_heads, dropout)
        # one learnable bias per (head, frequency); eq.6 delta_c, zero-init
        self.pope_delta = nn.Parameter(torch.zeros(n_heads, self.d_head))

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # magnitude = softplus(content) (eq.3); phase = position only (eq.4)
        mq, mk = F.softplus(Q), F.softplus(K)
        d = self.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, self.n_heads, 1, -1)
        cd, sd = torch.cos(d), torch.sin(d)
        # key phase shifted by +delta_c (eq.8)
        cosK = cos_a * cd - sin_a * sd
        sinK = sin_a * cd + cos_a * sd

        # a_ts = sum_c mu_q mu_k cos((s-t)theta_c + delta_c), via cos(A-B) expansion
        scores = (torch.matmul(mq * cos_a, (mk * cosK).transpose(-1, -2))
                  + torch.matmul(mq * sin_a, (mk * sinK).transpose(-1, -2))) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


def _widen_to_d(model, grid_size=64, bottleneck_r=2):
    """Rebuild the position machinery with n_blocks = d_head (paper: d, not d/2)."""
    model.n_blocks = model.d_head
    model.action_to_lie = ActionToLieAlgebra(model.d_model, model.n_heads,
                                             model.n_blocks, bottleneck_r)
    model.path_integrator = PathIntegrator(model.n_heads, model.n_blocks, grid_size)
    return model


def _swap(layers, d_model, n_heads):
    drop = layers[0].dropout.p if len(layers) else 0.1
    return nn.ModuleList([WMTransformerLayer_PoPE(d_model, n_heads, drop) for _ in layers])


class MapFormerWM_PoPE(MapFormerWM):
    """COMBO: MapFormer path-integration position + PoPE decoupling. Flat."""
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        _widen_to_d(self, grid_size, bottleneck_r)
        self.layers = _swap(self.layers, d_model, n_heads)


class MapFormerWM_RoPEIndex_PoPE(MapFormerWM):
    """PoPE ALONE: index position + PoPE decoupling (the paper's own setting).

    theta_c = base^{-(c-1)/d} for c = 1..d  (see the interpretation note above),
    phases phi_t = t * theta_c -- no path integration.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, base=10000.0, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size)
        self.n_blocks = self.d_head
        del self.action_to_lie, self.path_integrator
        c = torch.arange(self.n_blocks, dtype=torch.float32)         # c-1 = 0..d-1
        self.register_buffer("theta_c", base ** (-c / self.n_blocks))
        self.layers = _swap(self.layers, d_model, n_heads)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        ang = torch.outer(torch.arange(L, device=tokens.device, dtype=x.dtype), self.theta_c)
        cos_a = ang.cos()[None, None].expand(B, self.n_heads, L, -1)
        sin_a = ang.sin()[None, None].expand(B, self.n_heads, L, -1)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_Hourglass_PoPE(MapFormerWM_Hourglass_k2):
    """MapFormer path-integration + PoPE + single-level hourglass."""
    # BUGFIX 2026-08-28: this took (*a, **kw) and called _widen_to_d WITHOUT
    # bottleneck_r, so the rebuild there silently reset action_to_lie to the
    # default rank 2 -- r=2 and r=4 gave identical param counts. Every
    # hier-vs-flat comparison spanning the PoPE and MapWM families was
    # rank-confounded, unlogged. `kw.get("grid_size", 64)` was a second latent
    # bug in the same line: passed positionally, grid_size silently became 64.
    # Both are now named parameters, which also makes inspect.signature work.
    def __init__(self, *a, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(*a, grid_size=grid_size, bottleneck_r=bottleneck_r, **kw)
        _widen_to_d(self, grid_size, bottleneck_r)
        d, h = self.d_model, self.n_heads
        self.pre_layers = _swap(self.pre_layers, d, h)
        self.coarse_layers = _swap(self.coarse_layers, d, h)
        self.post_layers = _swap(self.post_layers, d, h)


class MapFormerWM_Hourglass_PoPE_CoarseIdx(MapFormerWM_Hourglass_CoarseIdx):
    """PoPE + ordinal index coarse position + hierarchy (best-of-both)."""
    def __init__(self, *a, grid_size=64, bottleneck_r=2, **kw):   # see BUGFIX above
        super().__init__(*a, grid_size=grid_size, bottleneck_r=bottleneck_r, **kw)
        _widen_to_d(self, grid_size, bottleneck_r)
        d, h = self.d_model, self.n_heads
        self.pre_layers = _swap(self.pre_layers, d, h)
        self.coarse_layers = _swap(self.coarse_layers, d, h)
        self.post_layers = _swap(self.post_layers, d, h)
