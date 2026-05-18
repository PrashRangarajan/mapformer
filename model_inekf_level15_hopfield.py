"""Level15 + Hopfield-style position-keyed memory bank (Tier 2 cross-scale fix).

The cross-scale residual gap to TEM (even after per-scale ω closes ~half of it)
is structurally about TEM's content-keyed memory being intrinsically less drift-
sensitive than RoPE-style position-encoded attention. This variant bolts a
position-only-keyed retrieval head on top of Level15:

- An auxiliary attention head with KV restricted to OBS positions only (mimics
  TEM's "write at obs token" semantics).
- The key is computed from `k0_pos` rotated by `θ̂` ONLY (no content channel) —
  position-only retrieval, like TEM's `g_t · M_g^T`.
- The value is the obs token embedding.
- The output is added to the standard transformer hidden state at each position.

Compared to a regular extra attention head, two key restrictions:
  1. KV is masked to obs positions (action positions never enter as keys/values).
  2. K is computed *only* from rotated `k0_pos` — no content K.

Both restrictions are what make this Hopfield-flavoured: write-trigger at obs,
position-only retrieval key.

This is parallelizable (just masked attention; one extra head). It should help
specifically at small grids (where TEM's content-keyed memory beat us) and at
sparse-landmark regimes (where TEM's one-shot bind beat us). For other regimes
we expect a small improvement or wash.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _apply_rope
from .model_inekf_level15 import MapFormerWM_Level15InEKF


class MapFormerWM_Level15_Hopfield(MapFormerWM_Level15InEKF):
    """Level15 + position-keyed Hopfield-style retrieval head."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        # Hopfield-style retrieval head: position-only keys, content values.
        self.k0_pos_hop = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.q0_pos_hop = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        # V: project content embedding to (n_heads, d_head)
        self.v_proj_hop = nn.Linear(d_model, d_model)
        # Output projection of the Hopfield head, added to the transformer output
        self.o_proj_hop = nn.Linear(d_model, d_model)
        self.norm_hop = nn.LayerNorm(d_model)
        # Hopfield temperature β (modern Hopfield, Ramsauer 2021)
        self.beta_hop = nn.Parameter(torch.tensor(1.0))

    def _hopfield_retrieve(self, x, theta_hat):
        """Position-keyed Hopfield retrieval.

        x:         (B, L, d_model) -- token embeddings (or transformer outputs).
        theta_hat: (B, L, H, NB)   -- corrected angles for each position.

        Returns:
            retrieved: (B, L, d_model) -- added back to the transformer output
        """
        B, L, _ = x.shape
        H = self.n_heads
        d_head = self.d_head

        # Build the position-only Q and K via RoPE-style rotation of learnable
        # position vectors q0/k0 by the corrected angle θ̂.
        theta_for_rope = theta_hat.transpose(1, 2)         # (B, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)
        q0 = self.q0_pos_hop.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        k0 = self.k0_pos_hop.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        q_pos = _apply_rope(q0, cos_a, sin_a)              # (B, H, L, d_head)
        k_pos = _apply_rope(k0, cos_a, sin_a)              # (B, H, L, d_head)

        # V: content of the token at each position
        v = self.v_proj_hop(x).view(B, L, H, d_head).transpose(1, 2)  # (B, H, L, d_head)

        # Obs mask: True at observation positions (odd indices). Action positions
        # don't write to the bank.
        obs_mask = torch.zeros(L, dtype=torch.bool, device=x.device)
        obs_mask[1::2] = True
        # Key-value mask: keep only obs positions
        kv_mask = obs_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, L)

        # Causal mask: position t can only attend to positions < t
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1,
        )
        # Combined mask: keep KV positions that are (a) obs AND (b) <= current t
        # We expand both to (L, L) and AND them
        kv_obs_only = obs_mask.unsqueeze(0).expand(L, L)               # (L_q, L_kv)
        valid_kv = kv_obs_only & ~causal_mask                          # (L_q, L_kv)
        # Position t cannot attend to anywhere if there's no prior obs — fall back
        # to attending to itself (so logits aren't all -inf).
        no_valid = ~valid_kv.any(dim=-1)
        valid_kv = valid_kv.clone()
        valid_kv[no_valid, no_valid.nonzero(as_tuple=False).squeeze(-1)] = True

        # Hopfield-style attention: β-scaled inner product, softmax, retrieve V
        scores = torch.matmul(q_pos, k_pos.transpose(-1, -2)) * self.beta_hop / math.sqrt(d_head)
        scores = scores.masked_fill(~valid_kv.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)                               # (B, H, L, L)

        out = torch.matmul(attn, v)                                    # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.o_proj_hop(self.norm_hop(out))

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

        # Hopfield-style retrieval added to the transformer output
        x_hop = self._hopfield_retrieve(x, theta_hat)
        x = x + x_hop

        x = self.out_norm(x)
        return self.out_proj(x)
