"""Per-scale ω variant of Level 1.5 InEKF.

`TEM_CROSSSCALE_DIAGNOSTIC.md` proposed that Level15's `PathIntegrator.omega`
is shared across grid sizes, so multi-scale training learns a compromise that
hurts the smallest grid the most. `SINGLE_SIZE_CONTROL.md` confirmed it
empirically: Level15 trained single-size at size 32 reaches 0.908 vs
multi-size at size 32 which gets 0.782 (+13pp gap from coupled ω).

This variant gives the model **one ω per scale** (3 scales = 3 ω vectors)
and selects the right one at forward time based on a `env_sizes` arg.

It's a drop-in replacement for `MapFormerWM_Level15InEKF` with an extra
`env_sizes: (B,) long tensor` kwarg in `forward`. The trainer
(`train_multisize.py`) passes per-batch sizes; eval scripts must do the same.
For backward compat, if `env_sizes` is None, the first scale's ω is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_inekf_level15 import MapFormerWM_Level15InEKF, InEKFLevel15


class MapFormerWM_Level15_PerScaleOmega(MapFormerWM_Level15InEKF):
    """Level15 with per-scale ω heads."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 sizes=(32, 64, 128)):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.sizes = tuple(sizes)
        n_scales = len(self.sizes)
        # Initialise each scale's ω as a copy of the parent's (so it can
        # specialise from there).
        omega0 = self.path_integrator.omega.detach().clone()
        self.omega_per_scale = nn.Parameter(
            omega0.unsqueeze(0).repeat(n_scales, 1, 1)
        )
        self._size_to_idx = {int(s): i for i, s in enumerate(self.sizes)}

    def _omega_for_batch(self, env_sizes, batch_size, device):
        """Return ω of shape (B, n_heads, n_blocks) selected by env size."""
        if env_sizes is None:
            idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # `env_sizes` may be a list of Python ints or a tensor
            if isinstance(env_sizes, (list, tuple)):
                idx_list = [self._size_to_idx[int(s)] for s in env_sizes]
                idx = torch.tensor(idx_list, dtype=torch.long, device=device)
            elif isinstance(env_sizes, torch.Tensor):
                # map sizes (e.g. 32, 64, 128) -> indices (0, 1, 2)
                # via a lookup
                idx = torch.zeros(env_sizes.shape[0], dtype=torch.long,
                                   device=device)
                for k, v in self._size_to_idx.items():
                    idx = torch.where(env_sizes == k,
                                       torch.full_like(idx, v), idx)
            else:
                idx_list = [self._size_to_idx[int(env_sizes)]] * batch_size
                idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        return self.omega_per_scale[idx]                           # (B, H, NB)

    def forward(self, tokens: torch.Tensor, env_sizes=None) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)

        omega = self._omega_for_batch(env_sizes, B, tokens.device)  # (B, H, NB)
        # broadcast: cum_delta (B, T, H, NB) * omega (B, 1, H, NB)
        theta_path = cum_delta * omega.unsqueeze(1)

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
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
