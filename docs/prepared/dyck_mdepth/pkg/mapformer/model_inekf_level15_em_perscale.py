"""Level15EM + per-scale ω (Tier 1 cross-scale fix).

Combines:
  - Level15EM: MapFormer-EM backbone + Level 1.5 InEKF correction (log_R_init_bias=3
    so the AND-gate isn't destroyed by random θ̂ at init)
  - PerScaleOmega: one learnable ω per training scale, selected at forward time
    by env_sizes kwarg

Tests whether fixing the coupled-ω limitation lets EM's multiplicative AND-gate
fire reliably. If the AND-gate's selectivity helps once A_P is repaired,
Level15EM_PerScaleOmega should beat both Level15EM (worse A_P) and
Level15_PerScaleOmega (no AND-gate selectivity).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import _apply_rope
from .model_inekf_level15_em import MapFormerEM_Level15InEKF


class MapFormerEM_Level15_PerScaleOmega(MapFormerEM_Level15InEKF):
    """Level15EM with per-scale ω heads."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 sizes=(32, 64, 128), log_R_init_bias: float = 3.0):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r, log_R_init_bias=log_R_init_bias)
        self.sizes = tuple(sizes)
        n_scales = len(self.sizes)
        omega0 = self.path_integrator.omega.detach().clone()
        self.omega_per_scale = nn.Parameter(
            omega0.unsqueeze(0).repeat(n_scales, 1, 1)
        )
        self._size_to_idx = {int(s): i for i, s in enumerate(self.sizes)}

    def _omega_for_batch(self, env_sizes, batch_size, device):
        if env_sizes is None:
            idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(env_sizes, (list, tuple)):
            idx = torch.tensor([self._size_to_idx[int(s)] for s in env_sizes],
                                dtype=torch.long, device=device)
        elif isinstance(env_sizes, torch.Tensor):
            idx = torch.zeros(env_sizes.shape[0], dtype=torch.long, device=device)
            for k, v in self._size_to_idx.items():
                idx = torch.where(env_sizes == k,
                                   torch.full_like(idx, v), idx)
        else:
            idx = torch.full((batch_size,), self._size_to_idx[int(env_sizes)],
                              dtype=torch.long, device=device)
        return self.omega_per_scale[idx]                           # (B, H, NB)

    def forward(self, tokens: torch.Tensor, env_sizes=None) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)

        omega = self._omega_for_batch(env_sizes, B, tokens.device)  # (B, H, NB)
        theta_path = cum_delta * omega.unsqueeze(1)

        theta_hat, Pi, K_, R = self.inekf(theta_path, x)

        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K_.detach()
        self.last_R = R.detach()

        theta_for_rope = theta_hat.transpose(1, 2)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        q0 = self.q0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        k0 = self.k0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        q_pos = _apply_rope(q0, cos_a, sin_a)
        k_pos = _apply_rope(k0, cos_a, sin_a)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, q_pos, k_pos, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
