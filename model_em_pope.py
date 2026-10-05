"""Gain granularity between MapEM and MapPoPE (GAIN_GRAIN_PREREG.md).

Every score in this family is

    score_ts = sum_c G_c(x_t, x_s) * A_c * cos(dtheta_c + delta_c),    dtheta_c = theta_t,c - theta_s,c

with the SAME path machinery in every arm (MapWM's ActionToLieAlgebra + PathIntegrator: 32 angles per head, omega
2pi .. 2pi/64, rank 2 by default). The two built ends:
  MapEM (`model.MapFormerEM`)          G_c = A_X = q_t . k_s / sqrt(d_head), ONE SIGNED scalar for all c;
                                       A_c = |q0_c||k0_c| / sqrt(d_head), delta_c = arg q0_c - arg k0_c (learned).
  MapPoPE-Pair (`model_pope_pair`)     G_c = softplus(q_t,e) softplus(k_s,e) per ELEMENT e (two elements per angle),
                                       A_c = 1, delta_e learned in [-2pi, 0] (sits at 0 on 89-100% of channels,
                                       docs/theory/2026-10-05/neuro_design.md 2.3).
The middle points built here (delta = 0 throughout):
  GainKernelLayer(n_modules=M)         G_c = mu^q_t,m(c) mu^k_s,m(c), mu = softplus(W LN(x) + b) >= 0, one gain per
                                       token per head per MODULE; module m = channels [m*32/M, (m+1)*32/M) in omega order
                                       (channel 0 = 2pi, the finest scale; channel 31 = 2pi/64, the coarsest);
                                       A_c = softplus(a_c) >= 0 learned per head per channel, init 1.
      M = 1   `GainScalar`             one non-negative scalar gain per token: content scales ONE fixed kernel whose
                                       maximum is at dtheta = 0 (pure gain field / pure rate remapping).
      M = 4   `GainMod4`               content chooses which band of spatial scales answers.
      M = 32                           one gain per angle: with A_c = 1 this IS MapPoPE-Pair with its two element
                                       gains per angle tied and delta = 0 (checked to float rounding,
                                       docs/audits/2026-10-05/gain_grain_equiv.py). Not trained; used by the checks.
  `MapFormerEM_NonNeg`                 MapEM with its content factor made non-negative: G = softplus(A_X). Same
                                       parameters, same initial weights as `VanillaEM` at a seed (no new draws); the
                                       only change is the sign constraint.
Score normalisation of GainKernelLayer: 2 / sqrt(d_head), so that M = 32, A_c = 1 equals MapPoPE-Pair's sum over 64
elements / sqrt(d_head) exactly (each angle drives two elements there).

Initialisation: GainKernelLayer draws q_proj, k_proj, v_proj, o_proj, the norms and the FFN exactly as
WMTransformerLayer (hence as WMTransformerLayer_PoPE) does, then deletes q_proj / k_proj and draws q_gain / k_gain. So
at a seed, `GainScalar` / `GainMod4` share every initial weight except the score's content projections with
`MapPoPE-Pair` (embeddings, rank-2 bottleneck, omega, v/o/FFN, readout) -- checked in the equivalence script.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import MapFormerWM, MapFormerEM, EMTransformerLayer, WMTransformerLayer


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class GainKernelLayer(WMTransformerLayer):
    """score_ts = (2/sqrt(d_head)) sum_m mu^q_t,m mu^k_s,m sum_{c in m} A_c cos(theta_t,c - theta_s,c).
    cos_a / sin_a: (B, H, T, n_blocks), one angle per channel (no element duplication)."""

    def __init__(self, d_model, n_heads, dropout, n_blocks, n_modules):
        super().__init__(d_model, n_heads, dropout)              # draws q/k/v/o, norms, FFN as WMTransformerLayer
        assert n_blocks % n_modules == 0, (n_blocks, n_modules)
        del self.q_proj, self.k_proj                             # the score's content enters only as gains
        self.n_blocks, self.n_modules = n_blocks, n_modules
        self.q_gain = nn.Linear(d_model, n_heads * n_modules)
        self.k_gain = nn.Linear(d_model, n_heads * n_modules)
        self.amp_raw = nn.Parameter(torch.full((n_heads, n_blocks), _inv_softplus(1.0)))   # A_c = softplus -> 1

    def gains(self, h):
        """(mu_q, mu_k) per channel: (B, H, T, n_blocks), module gains repeated over their channels."""
        B, T, _ = h.shape; H, M = self.n_heads, self.n_modules; rep = self.n_blocks // M
        gq = F.softplus(self.q_gain(h)).view(B, T, H, M).transpose(1, 2).repeat_interleave(rep, dim=-1)
        gk = F.softplus(self.k_gain(h)).view(B, T, H, M).transpose(1, 2).repeat_interleave(rep, dim=-1)
        return gq, gk

    def amplitude(self):
        return F.softplus(self.amp_raw)                          # (H, n_blocks), A_c >= 0

    def score(self, hq, hk, cq, sq, ck, sk):
        gq, _ = self.gains(hq); _, gk = self.gains(hk)
        qa = gq * self.amplitude().view(1, self.n_heads, 1, -1)
        return (torch.matmul(qa * cq, (gk * ck).transpose(-1, -2))
                + torch.matmul(qa * sq, (gk * sk).transpose(-1, -2))) * (2.0 / math.sqrt(self.d_head))

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = self.score(h, h, cos_a, sin_a, cos_a, sin_a)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerWM_GainKernel(MapFormerWM):
    """MapWM's path machinery (32 angles per head, same omega range, same rank) with GainKernelLayer attention."""
    N_MODULES = None

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2,
                 n_modules=None, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        M = n_modules if n_modules is not None else self.N_MODULES
        assert self.n_blocks * 2 == self.d_head, (self.n_blocks, self.d_head)
        drop = self.layers[0].dropout.p if len(self.layers) else dropout
        self.layers = nn.ModuleList([GainKernelLayer(d_model, n_heads, drop, self.n_blocks, M) for _ in self.layers])
        self.n_modules = M

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.action_to_lie(x))          # (B, H, T, n_blocks)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_GainScalar(MapFormerWM_GainKernel):
    """M = 1: one non-negative gain per token per head; content scales one fixed kernel (pure gain field)."""
    N_MODULES = 1


class MapFormerWM_GainMod4(MapFormerWM_GainKernel):
    """M = 4 modules of 8 contiguous channels in omega order (fine -> coarse scale bands)."""
    N_MODULES = 4


class MapFormerWM_GainMod32(MapFormerWM_GainKernel):
    """M = 32: one gain per angle (checks only; = MapPoPE-Pair with tied element gains, delta 0, at A_c = 1)."""
    N_MODULES = 32


class EMTransformerLayer_NonNeg(EMTransformerLayer):
    """MapEM's layer with the content factor passed through softplus: scores = softplus(A_X) * A_P. No new parameters.
    `content_map` is a class attribute so the identity check (content_map = identity -> MapEM bit for bit) can set it."""
    content_map = staticmethod(F.softplus)

    def forward(self, x, q_pos, k_pos, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        Q_c = self.q_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K_c = self.k_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        A_X = torch.matmul(Q_c, K_c.transpose(-1, -2)) / scale
        A_P = torch.matmul(q_pos, k_pos.transpose(-1, -2)) / scale
        scores = self.content_map(A_X) * A_P
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerEM_NonNeg(MapFormerEM):
    """MapEM (separate q0_pos / k0_pos, 32 angles per head, rank 2) with G = softplus(A_X) >= 0. The layers are
    MapFormerEM's own, re-classed (no RNG draw), so the initial weights equal VanillaEM's at the same seed."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        for layer in self.layers:
            assert type(layer) is EMTransformerLayer
            layer.__class__ = EMTransformerLayer_NonNeg


ARMS = {"GainScalar": MapFormerWM_GainScalar, "GainMod4": MapFormerWM_GainMod4, "VanillaEM_NonNeg": MapFormerEM_NonNeg}


def register(vmap):
    """Add the new arms (and MapPoPE-Pair, unchanged) to a VARIANT_MAP without editing train_variant.py."""
    from mapformer.model_pope_pair import MapFormerWM_PoPEPair
    vmap["MapPoPE-Pair"] = MapFormerWM_PoPEPair
    for k, v in ARMS.items():
        vmap[k] = v
    return vmap
