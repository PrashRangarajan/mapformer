"""Level 1.5 InEKF with learnable softmax temperature β in attention.

Tests whether MapFormer's existing attention (already a modern-Hopfield-
equivalent memory) can recover TEMFaithful's landmark-retrieval sharpness
just by sharpening the softmax — without any architectural change to the
attention's keys / values / storage rule.

Hypothesis: TEMFaithful wins on lm200 partly because its softmax
retrieval is sharper (learnable β) and its memory is sparser (obs-only,
half the size). MapFormer's attention has a fixed `1/sqrt(d_head)`
temperature; can it benefit from a learnable scalar that the model can
push up to sharpen retrieval at landmarks?

Crucially this is the cleanest test of the dilution hypothesis that
*preserves MapFormer's design philosophy*: no hardcoded action/obs
mask, no domain knowledge injected, no new memory. Just one extra
scalar parameter per attention layer.

If accuracy on lm200 OOD T=512 stays at Level15's ~0.82, the
dilution-from-action-tokens hypothesis is wrong and TEMFaithful's
landmark advantage is from something else (most likely the orthogonal
W_a's exact loop closure, which MapFormer's learned ω·Δ approximates
but doesn't enforce). If accuracy moves toward TEMFaithful's ~0.97,
the dilution hypothesis is right and MapFormer can recover most of
the gap with this trivial change.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, _apply_rope
from .model_inekf_level15 import InEKFLevel15


class WMTransformerLayer_Beta(nn.Module):
    """MapFormer-WM transformer layer with a learnable softmax temperature.

    Identical to the standard WMTransformerLayer except the attention
    score is multiplied by a learnable scalar β before the softmax.
    Initialised at 1/sqrt(d_head) (the standard transformer default) so
    untrained behaviour matches the baseline.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # Learnable softmax temperature. Init at 1/sqrt(d_head) so first-
        # forward behaviour matches the standard transformer scaling.
        # log-parameterised for positivity.
        init_log_beta = math.log(1.0 / math.sqrt(self.d_head))
        self.log_beta = nn.Parameter(torch.tensor(init_log_beta, dtype=torch.float32))

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape

        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)

        # Learnable β: scores = β · Q · K^T (replaces fixed 1/sqrt(d_head))
        beta = torch.exp(self.log_beta)
        scores = beta * torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.o_proj(attn_out)

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerWM_Level15Beta(MapFormerWM):
    """MapFormer-WM with Level 1.5 InEKF and learnable β temperature in attention.

    Differs from MapFormerWM_Level15InEKF in exactly one place: the
    transformer layers use WMTransformerLayer_Beta (learnable softmax
    temperature) instead of the fixed-temperature WMTransformerLayer.
    All other architectural choices unchanged.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer_Beta(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.inekf = InEKFLevel15(d_model, n_heads, self.n_blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        theta_hat, Pi, K, R = self.inekf(theta_path, x)

        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K.detach()
        self.last_R = R.detach()

        theta_for_rope = theta_hat.transpose(1, 2)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
