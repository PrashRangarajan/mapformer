"""A decay envelope on PoPE: suppress the pairs whose position kernel is unreliable.

THEORY_MAPPOPE.md / T3GEN_RESULTS.md leave one failure mode: PoPE's kernel
`sum_c mu_q mu_k cos(omega_c (S_t - S_s) - delta_c)` has non-negative amplitudes and constant phases,
so once the argument leaves the range training calibrated it produces CONFIDENT WRONG values rather
than decaying to nothing. xPos (arXiv:2212.10554) fixes the analogous failure in RoPE with a decay
envelope, and ALiBi (arXiv:2108.12409) shows a linear-in-distance additive bias is enough.

Here: `scores += -softplus(lambda_h) * distance`, one learnable scalar per head, initialised on
ALiBi's geometric spread 2^-1 .. 2^-n_heads (so the parameter cost is n_heads, not a projection). Two distances, matching the two position mechanisms:

  index version        distance = |t - s|                       (the classical ALiBi/xPos distance)
  path-integrated      distance = |S_t - S_s| / scale           (the model's own accumulated distance)

For the path-integrated form the accumulator is averaged over frequency blocks and divided by its own
per-sequence mean absolute step, so `lambda` means the same thing -- decay per token-equivalent --
whatever rate the model has learned, and the initialisation is comparable across arms.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_pope import (WMTransformerLayer_PoPE, MapFormerWM_PoPE,
                         MapFormerWM_RoPEIndex_PoPE, DELTA_MIN, DELTA_MAX)


class WMTransformerLayer_PoPE_Decay(WMTransformerLayer_PoPE):
    """PoPE attention plus an additive, distance-proportional log-decay."""

    def __init__(self, d_model, n_heads, dropout, lam_init=None):
        super().__init__(d_model, n_heads, dropout)
        # ALiBi's geometric spread across heads (2^-1 .. 2^-n_heads): some heads local, some
        # global. A single constant would make every head local -- at lambda 0.1 a pair 512
        # tokens apart is penalised by 51 logits, i.e. no long-range attention at all.
        lam = (torch.full((n_heads,), float(lam_init)) if lam_init is not None
               else 2.0 ** -torch.arange(1, n_heads + 1, dtype=torch.float32))
        self.lam_raw = nn.Parameter(torch.log(torch.expm1(lam)))
        self._dist = None                      # (B, 1 or H, T, T), set by the model each forward

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        sh = lambda z: z.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = sh(self.q_proj(h)), sh(self.k_proj(h)), sh(self.v_proj(h))
        mq, mk = F.softplus(Q), F.softplus(K)
        d = self.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, self.n_heads, 1, -1)
        cd, sd = torch.cos(d), torch.sin(d)
        cosK, sinK = cos_a * cd - sin_a * sd, sin_a * cd + cos_a * sd
        scores = (torch.matmul(mq * cos_a, (mk * cosK).transpose(-1, -2))
                  + torch.matmul(mq * sin_a, (mk * sinK).transpose(-1, -2))) / math.sqrt(self.d_head)
        lam = F.softplus(self.lam_raw).view(1, -1, 1, 1)
        scores = scores - lam * self._dist                       # the envelope
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), V)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, self.d_model))
        x = x + self.dropout(out)
        return x + self.ffn(self.norm2(x))


def _swap_decay(layers, d_model, n_heads, lam_init=None):
    drop = layers[0].dropout.p if len(layers) else 0.1
    return nn.ModuleList([WMTransformerLayer_PoPE_Decay(d_model, n_heads, drop, lam_init)
                          for _ in layers])


class MapFormerWM_PoPE_Decay(MapFormerWM_PoPE):
    """Path integration + PoPE + a decay envelope in the ACCUMULATED distance."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, bottleneck_r=2, lam_init=None):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        self.layers = _swap_decay(self.layers, d_model, n_heads, lam_init)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)                                  # (B, L, H, n_blocks)
        cos_a, sin_a = self.path_integrator(delta)
        S = delta.mean(-1).cumsum(1).transpose(1, 2)                   # (B, H, L)
        step = delta.mean(-1).abs().mean(dim=(1,), keepdim=True).transpose(1, 2).clamp_min(1e-6)
        dist = (S.unsqueeze(-1) - S.unsqueeze(-2)).abs() / step.unsqueeze(-1)   # (B, H, L, L)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            layer._dist = dist
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_RoPEIndex_PoPE_Decay(MapFormerWM_RoPEIndex_PoPE):
    """Index PoPE + the classical |t - s| decay envelope (the ALiBi/xPos distance)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, base=10000.0, lam_init=None, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, base)
        self.layers = _swap_decay(self.layers, d_model, n_heads, lam_init)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        ang = torch.outer(torch.arange(L, device=tokens.device, dtype=x.dtype), self.theta_c)
        cos_a = ang.cos()[None, None].expand(B, self.n_heads, L, -1)
        sin_a = ang.sin()[None, None].expand(B, self.n_heads, L, -1)
        t = torch.arange(L, device=tokens.device, dtype=x.dtype)
        dist = (t.view(-1, 1) - t.view(1, -1)).abs()[None, None]        # (1, 1, L, L)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            layer._dist = dist
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


# ---------------------------------------------------------------------------
# The crossed arms (2026-09-20): position mechanism x decay METRIC.
# DYCK_DECAY_RESULTS.md observed that the same envelope helps the path-integrated row and guts the
# index row, and proposed that a decay envelope is a proximity prior in whatever metric the position
# variable defines. That contrast varied position AND metric together. These two arms cross them.
# ---------------------------------------------------------------------------

class MapFormerWM_PoPE_Decay_IdxMetric(MapFormerWM_PoPE_Decay):
    """Path-integrated PHASE, but the envelope decays over TOKEN distance |t - s|."""

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)
        t = torch.arange(L, device=tokens.device, dtype=x.dtype)
        dist = (t.view(-1, 1) - t.view(1, -1)).abs()[None, None]
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            layer._dist = dist
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_RoPEIndex_PoPE_Decay_StateMetric(MapFormerWM_RoPEIndex_PoPE_Decay):
    """INDEX phase, but the envelope decays over a learned state distance |S_t - S_s|.

    The index PoPE class deletes the increment map, so one is added back here and used ONLY to
    supply the decay metric -- the attention phase stays `t * theta_c`. Its increments are learned
    through the envelope alone.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, base=10000.0, lam_init=None, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, base, lam_init)
        from .model import ActionToLieAlgebra
        self.metric_map = ActionToLieAlgebra(d_model, n_heads, self.d_head, bottleneck_r)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        ang = torch.outer(torch.arange(L, device=tokens.device, dtype=x.dtype), self.theta_c)
        cos_a = ang.cos()[None, None].expand(B, self.n_heads, L, -1)
        sin_a = ang.sin()[None, None].expand(B, self.n_heads, L, -1)
        d = self.metric_map(x)                                        # (B, L, H, d_head)
        S = d.mean(-1).cumsum(1).transpose(1, 2)                      # (B, H, L)
        step = d.mean(-1).abs().mean(dim=(1,), keepdim=True).transpose(1, 2).clamp_min(1e-6)
        dist = (S.unsqueeze(-1) - S.unsqueeze(-2)).abs() / step.unsqueeze(-1)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            layer._dist = dist
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_RoPEIndex_PoPE_Decay_FrozenMetric(MapFormerWM_RoPEIndex_PoPE_Decay_StateMetric):
    """The control for the crossed index arm (audit, 2026-09-20).

    Identical parameters to the learned-state arm, and the envelope still decays over a STATE
    distance -- but the increment map is frozen at initialisation, so the metric is a fixed random
    projection rather than something training shapes. Separates three things the crossed arm bundles:
    the extra 256 parameters, having any non-token metric at all, and LEARNING that metric.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for p in self.metric_map.parameters():
            p.requires_grad_(False)
