"""Tier 1, item 2: warm-start single-p0 EM at the constructed recency solution.

AUDIT_2026-09-10.md finding 2 shows at the KERNEL level that a single-p0 position
kernel (rho = 1, zero phase freedom) selects the recency answer exactly, once the
query token q_k carries Delta = -(k-1) on a count axis and rewinds it (MASK carries
+1 on a cycle axis to separate older cycles). That is an argmax with an idealised
content gate -- not a demonstration that the FULL one-layer model can reach or hold
the solution. This module installs the construction's POSITION PATHWAY inside the
unmodified VanillaEM_P0_r4 architecture and leaves the content branch random:

    token_emb  latent rewind code in coords 0-1 (scaled by EPS = 1/64)
    w_in       reads coords 0-1 at gain 1/EPS -> h = z(token)
    w_out      random 2-vector per (head, block), divided by 2*pi so that with the
               default omega = 2*pi*(1/64)^frac the effective frequencies are
               W_SCALE*(1/64)^frac -- sharpened from the verified x1 schedule so the
               answer beats its nearest competitor by ~58% of the kernel peak, not 3%
    p0         rescaled to the per-head norm VanillaEM_P0_r4 LEARNS on recency
               (1.957, mean of 8 trained seeds), so the kernel amplitude is the
               learned one rather than the 100x-smaller init

What the content branch must still learn: attend to SYMBOL-type keys only. Filler
keys between the answer and the next symbol, and the query token's own key, tie the
answer on A_P (filler has Delta = 0; the query key sits at dS = 0), so a token-type
gate in A_X is required -- the construction never claimed otherwise.

Two variants:
    EMWarm_freeze  position pathway FROZEN (w_in, w_out, omega, p0, latent code)
    EMWarm_train   position pathway installed, everything trainable

Embedding: token_emb(t) = base(t) + latent(t), with base's coords 0-1 held at zero
by a gradient mask (decay of zero is zero). That is the same function class as one
table with two columns held at the latent values -- NOT a separate embedding -- so
the architecture is unchanged. In the trainable variant the latent is a Parameter,
so each coordinate still has exactly one parameter.

Token layout is the default recency config (16 symbols, 8 filler, k_max 64);
construction asserts vocab 89.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model_em_fixed import MapFormerEM_SingleP0_r4

N_SYM, N_FILL, K_MAX = 16, 8, 64
SYM0, FILL0, Q0 = 0, N_SYM, N_SYM + N_FILL
MASK = N_SYM + N_FILL + K_MAX
VOCAB = MASK + 1
EPS = 1.0 / 64
P0_NORM = 1.957
N_LAT = 2
# Chosen by a margin sweep BEFORE any training (argmax exact at every scale x1..x32 on
# 3 seeds; min margin / kappa_max 3% at x1, ~12% x2, ~29% x4, ~44% x8, ~58% x16, ~62% x32).
# It changes only how hard the content gate's job is, not whether the kernel is correct.
W_SCALE = 16.0


def rewind_latent():
    z = torch.zeros(VOCAB, N_LAT)
    z[SYM0:SYM0 + N_SYM, 0] = 1.0                 # symbol: +1 on the count axis
    for k in range(1, K_MAX + 1):
        z[Q0 + k - 1, 0] = -(k - 1.0)             # q_k: rewind the count by k-1
    z[MASK, 1] = 1.0                              # MASK: +1 on the cycle axis
    return z                                      # filler: 0


class _LatentEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, latent: torch.Tensor, trainable: bool):
        super().__init__()
        self.base = base
        with torch.no_grad():
            self.base.weight[:, :N_LAT] = 0.0
        m = torch.ones_like(self.base.weight)
        m[:, :N_LAT] = 0.0
        self.register_buffer("grad_mask", m)
        self.base.weight.register_hook(lambda g: g * self.grad_mask)
        if trainable:
            self.latent = nn.Parameter(latent.clone())
        else:
            self.register_buffer("latent", latent.clone())

    def forward(self, t):
        e = self.base(t)
        return e + F.pad(self.latent[t], (0, e.shape[-1] - N_LAT))


def _warm(freeze: bool):
    class _W(MapFormerEM_SingleP0_r4):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, bottleneck_r)
            assert vocab_size == VOCAB, f"warm start built for vocab {VOCAB}, got {vocab_size}"
            H, nb = self.n_heads, self.n_blocks
            self.token_emb = _LatentEmbedding(self.token_emb, EPS * rewind_latent(),
                                              trainable=not freeze)
            with torch.no_grad():
                wi = self.action_to_lie.w_in.weight          # (r, d_model)
                wi.zero_()
                wi[0, 0] = 1.0 / EPS
                wi[1, 1] = 1.0 / EPS
                wo = self.action_to_lie.w_out.weight         # (H*nb, r)
                wo.zero_()
                wo[:, :N_LAT] = torch.randn(H * nb, N_LAT) * W_SCALE / (2 * math.pi)
                p = self.p0_pos.data
                self.p0_pos.data.copy_(p / p.norm(dim=-1, keepdim=True) * P0_NORM)
            if freeze:
                for prm in (self.action_to_lie.w_in.weight, self.action_to_lie.w_out.weight,
                            self.path_integrator.omega, self.p0_pos):
                    prm.requires_grad_(False)
            self.warm_frozen = freeze
    _W.__name__ = f"MapFormerEM_Warm_{'freeze' if freeze else 'train'}_r4"
    return _W


MapFormerEM_Warm_freeze_r4 = _warm(True)
MapFormerEM_Warm_train_r4 = _warm(False)
