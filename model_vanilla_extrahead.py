"""Capacity control for the headline Level15-vs-Vanilla comparison.

Level15 has ~305K params vs Vanilla's ~256K (+19%). Is the Level15 win
(noise / lm200 OOD) from the InEKF correction, or just from ~50K extra
parameters?

`Vanilla_ExtraHead` = Vanilla MapFormer-WM + a generic extra attention
head (content Q/K projections, position-modulated by the SAME cos/sin
the main attention uses, all-positions causal KV), added residually.
This gives Vanilla ~Level15-level parameter count with the capacity
actually usable — unlike L15_NoCorr, whose extra InEKF params are
present but zeroed-out (unused).

Decision:
  - Vanilla_ExtraHead ≈ Vanilla   -> Level15's win is the InEKF correction
                                     (architectural), not capacity.
  - Vanilla_ExtraHead -> Level15  -> the win is partly capacity; report honestly.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, _apply_rope


class MapFormerWM_Vanilla_ExtraHead(MapFormerWM):
    """Vanilla MapFormer-WM + a generic extra attention head (capacity control)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.q_proj_extra = nn.Linear(d_model, d_model)
        self.k_proj_extra = nn.Linear(d_model, d_model)
        self.v_proj_extra = nn.Linear(d_model, d_model)
        self.o_proj_extra = nn.Linear(d_model, d_model)
        self.norm_extra = nn.LayerNorm(d_model)

    def _extra_attention(self, x, cos_a, sin_a):
        B, L, _ = x.shape
        H, d_head = self.n_heads, self.d_head
        h = self.norm_extra(x)
        Q = self.q_proj_extra(h).view(B, L, H, d_head).transpose(1, 2)
        K = self.k_proj_extra(h).view(B, L, H, d_head).transpose(1, 2)
        V = self.v_proj_extra(h).view(B, L, H, d_head).transpose(1, 2)
        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_head)
        causal = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1,
        )
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, L, self.d_model)
        return self.o_proj_extra(out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x_extra = self._extra_attention(x, cos_a, sin_a)
        x = x + x_extra

        x = self.out_norm(x)
        return self.out_proj(x)
