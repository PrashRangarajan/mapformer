"""Ablation: Level 1.5 InEKF with the post-attention residual dropout REMOVED.

Disambiguation experiment for Level15Beta. Level15Beta differs from Level15
in two places: (a) learnable β temperature, and (b) the post-attn residual
add no longer wraps o_proj(out) in self.dropout. Since the learned β barely
moves (0.15 vs init 0.125), this ablation isolates (b).

If this variant matches Level15Beta on lm200 OOD T=512 (~0.93), the +12pp
jump was driven by the dropout removal, not by β. If it stays at Level15
(~0.82), β was load-bearing.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, _apply_rope
from .model_inekf_level15 import InEKFLevel15


class WMTransformerLayer_NoDrop(nn.Module):
    """Identical to WMTransformerLayer except the post-attn residual add
    no longer applies self.dropout to o_proj(out). Attn-weight dropout
    inside the softmax is unchanged. Fixed scale 1/sqrt(d_head)."""

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

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape

        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        # NO self.dropout wrapping the residual add — the only change vs vanilla.
        x = x + self.o_proj(out)

        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerWM_Level15NoDrop(MapFormerWM):
    """MapFormer-WM with Level 1.5 InEKF and post-attention residual
    dropout removed (β fixed at 1/sqrt(d_head)).
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer_NoDrop(d_model, n_heads, dropout)
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
