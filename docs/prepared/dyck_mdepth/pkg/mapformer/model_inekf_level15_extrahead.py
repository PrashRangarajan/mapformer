"""Capacity control for Level15_Hopfield.

Level15_Hopfield closes the cross-scale gap (+14pp at size 32). Is that from
the Hopfield head's *structure* (position-only key, obs-restricted KV) or
just from adding an extra attention head's worth of capacity?

This variant adds a GENERIC extra attention head to Level15, the same
residual way the Hopfield head is added (`x = x + x_extra`), but with no
Hopfield restrictions:
  - Q, K from full content projections (not fixed position-only vectors)
  - Q, K position-modulated by θ̂ (same as the main attention)
  - KV over ALL positions (not obs-only)

It has MORE parameters than the Hopfield head (full q_proj/k_proj vs
fixed q0/k0 vectors), so it is a conservative capacity control: if even
this more-capacity generic head fails to close the gap, the Hopfield
*structure* is unambiguously what matters.

Decision:
  - ExtraHead ≈ Level15_Hopfield  -> the +14pp was capacity / extra head.
  - ExtraHead ≈ Level15           -> the Hopfield structure is doing the work.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _apply_rope
from .model_inekf_level15 import MapFormerWM_Level15InEKF


class MapFormerWM_Level15_ExtraHead(MapFormerWM_Level15InEKF):
    """Level15 + a generic extra attention head (capacity control)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        # Generic extra attention head — full content projections.
        self.q_proj_extra = nn.Linear(d_model, d_model)
        self.k_proj_extra = nn.Linear(d_model, d_model)
        self.v_proj_extra = nn.Linear(d_model, d_model)
        self.o_proj_extra = nn.Linear(d_model, d_model)
        self.norm_extra = nn.LayerNorm(d_model)

    def _extra_attention(self, x, theta_hat):
        """A standard position-modulated content attention head.

        Same as a regular WM transformer layer's attention: content Q/K,
        rotated by θ̂, full causal KV. The opposite of the Hopfield head's
        restrictions — this is the 'just another head' control.
        """
        B, L, _ = x.shape
        H, d_head = self.n_heads, self.d_head

        h = self.norm_extra(x)
        Q = self.q_proj_extra(h).view(B, L, H, d_head).transpose(1, 2)
        K = self.k_proj_extra(h).view(B, L, H, d_head).transpose(1, 2)
        V = self.v_proj_extra(h).view(B, L, H, d_head).transpose(1, 2)

        # Position-modulate (same as the main attention)
        theta_for_rope = theta_hat.transpose(1, 2)             # (B, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)
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

        # Generic extra head added the same residual way as the Hopfield head
        x_extra = self._extra_attention(x, theta_hat)
        x = x + x_extra

        x = self.out_norm(x)
        return self.out_proj(x)
