"""T3 (THEORY_MAPPOPE.md): give PoPE's phase offset a PER-TOKEN component.

PoPE's score is  a_ts = sum_c mu_q,tc mu_k,sc cos(phi_t,c - phi_s,c - delta_c)  with `delta_c` a
per-(head, channel) CONSTANT, so the kernel's peak is fixed and content can only scale channels.
MapWM instead rotates the content vectors, which puts `angle(q_p) - angle(k_p)` inside the cosine:
the current query and key shift the peak. The account says that pairwise freedom is what absorbs a
mis-scaled accumulator, and that PoPE's lack of it is why MapPoPE collapses out of distribution.

Here `delta` becomes `delta_c + d^q_c(x_t)` on queries and `delta_c + d^k_c(x_s)` on keys, from two
linear heads, ZERO-INITIALISED so the model starts as exactly PoPE and can only move away from it.
No raw angle is needed: cos(phi + d) = cos(phi)cos(d) - sin(phi)sin(d).

`phase_gate=0.0` gives the inert twin required by rule 8: identical parameter count, the phase heads
present and receiving no gradient, function-identical to plain MapPoPE.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_pope import WMTransformerLayer_PoPE, MapFormerWM_PoPE, DELTA_MIN, DELTA_MAX, _widen_to_d


class WMTransformerLayer_PoPE_T3(WMTransformerLayer_PoPE):
    def __init__(self, d_model, n_heads, dropout, phase_gate=1.0):
        super().__init__(d_model, n_heads, dropout)
        self.phase_gate = phase_gate
        self.dq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.dk = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        nn.init.zeros_(self.dq.weight); nn.init.zeros_(self.dk.weight)

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        sh = lambda z: z.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = sh(self.q_proj(h)), sh(self.k_proj(h)), sh(self.v_proj(h))
        mq, mk = F.softplus(Q), F.softplus(K)

        dq = self.phase_gate * sh(self.dq(h))                      # per-token query phase
        dk = self.phase_gate * sh(self.dk(h))                      # per-token key phase
        d0 = self.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, self.n_heads, 1, -1)
        cq, sq = torch.cos(dq), torch.sin(dq)
        ck, sk = torch.cos(dk + d0), torch.sin(dk + d0)

        Qc, Qs = cos_a * cq - sin_a * sq, sin_a * cq + cos_a * sq
        Kc, Ks = cos_a * ck - sin_a * sk, sin_a * ck + cos_a * sk
        scores = (torch.matmul(mq * Qc, (mk * Kc).transpose(-1, -2))
                  + torch.matmul(mq * Qs, (mk * Ks).transpose(-1, -2))) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), V)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, self.d_model))
        x = x + self.dropout(out)
        return x + self.ffn(self.norm2(x))


class MapFormerWM_PoPE_T3(MapFormerWM_PoPE):
    """MapPoPE with a per-token phase offset (phase_gate=1) or its inert twin (phase_gate=0)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, bottleneck_r=2, phase_gate=1.0):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer_PoPE_T3(d_model, n_heads, dropout, phase_gate)
            for _ in self.layers])


class MapFormerWM_PoPE_T3_Inert(MapFormerWM_PoPE_T3):
    def __init__(self, *a, **kw):
        kw["phase_gate"] = 0.0
        super().__init__(*a, **kw)


class MapFormerWM_PoPE_T3_PI01(MapFormerWM_PoPE_T3):
    """T3 with the phase heads initialised at std 0.1 rather than zero.

    T3GEN_RESULTS.md G3: on Bach the zero initialisation is a BAD PRIOR -- forcing the phase to
    start away from zero improves both in-distribution NLL (-0.041, 5/5) and extrapolation
    (-0.116, 5/5). The Dyck-2 and torus verdicts in that same file used the zero start, so they
    are re-run with this class.
    """

    def __init__(self, *a, phase_init=0.1, **kw):
        super().__init__(*a, **kw)
        for mod in self.modules():
            for nm in ("dq", "dk"):
                h = getattr(mod, nm, None)
                if h is not None:
                    nn.init.normal_(h.weight, 0.0, phase_init)
