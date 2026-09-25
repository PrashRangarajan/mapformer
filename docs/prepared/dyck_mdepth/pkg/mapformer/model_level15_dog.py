"""Level 1.5 InEKF + Sorscher-style DoG auxiliary head (Option A).

Tests whether hex-grid representations emerge in a non-negative bottleneck
layer when supervised with a DoG-of-position place-cell target, on top of
the standard MapFormer-WM categorical CE loss.

Sorscher/Ganguli (2019) prove hex emerges from three conditions:
    1. Path integration (already in MapFormer)
    2. Non-negativity on the spatial code units (ReLU bottleneck below)
    3. Place-cell-like targets with DoG / center-surround structure

The aux head:
    hidden h_t (d_model) -> Linear -> ReLU (the "grid layer") -> Linear -> p̂_t
where p̂_t is a vector of n_place place-cell predictions. The supervision
target is DoG(d(pos_t, c_j)) for each place-cell center c_j. The grid
layer is the candidate site to probe for hex.

The MapFormer categorical CE loss is unchanged — the variant still solves
the original revisit-prediction task. The DoG aux is added with weight
``aux_coef``.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .model_inekf_level15 import MapFormerWM_Level15InEKF


class MapFormerWM_Level15_DoG(MapFormerWM_Level15InEKF):
    """Level15 + DoG-supervised place-cell auxiliary head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        grid_size: int = 64,
        bottleneck_r: int = 2,
        n_grid_units: int = 256,
        n_place_cells: int = 256,
        sigma_E: float = 1.5,
        sigma_I: float = 3.0,
    ):
        super().__init__(
            vocab_size, d_model, n_heads, n_layers, dropout,
            grid_size, bottleneck_r,
        )
        self.n_grid_units = n_grid_units
        self.n_place_cells = n_place_cells
        self.sigma_E = sigma_E
        self.sigma_I = sigma_I
        self.gs = grid_size

        self.grid_proj = nn.Linear(d_model, n_grid_units)
        self.place_proj = nn.Linear(n_grid_units, n_place_cells)

        # Place-cell centers: regular grid over the torus.
        side = int(round(n_place_cells ** 0.5))
        assert side * side == n_place_cells, "n_place_cells must be a perfect square"
        spacing = grid_size / side
        cs = torch.arange(side, dtype=torch.float32) * spacing + spacing / 2.0
        cx, cy = torch.meshgrid(cs, cs, indexing="ij")
        centers = torch.stack([cx.flatten(), cy.flatten()], dim=-1)  # (n_place, 2)
        self.register_buffer("place_centers", centers)

        # Filled by training loop right before forward(). Shape (B, L, 2),
        # where L = input length (2 * n_steps - 1) and entries at obs
        # positions (odd indices) hold the agent's location at that step.
        self._batch_positions: torch.Tensor | None = None
        self.last_grid_activations: torch.Tensor | None = None
        self.last_place_pred: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = (
            cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        )

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

        h = self.out_norm(x)  # (B, L, d_model)

        # DoG aux head — non-negative bottleneck (the candidate hex layer)
        g = torch.relu(self.grid_proj(h))           # (B, L, n_grid)
        p_hat = self.place_proj(g)                  # (B, L, n_place)
        self.last_grid_activations = g
        self.last_place_pred = p_hat

        return self.out_proj(h)

    def dog_targets(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute non-negative DoG place-cell targets for given positions.

        Args:
            positions: (B, L, 2) in cell coordinates [0, grid_size).
        Returns:
            (B, L, n_place) non-negative DoG values at each center.
        """
        c = self.place_centers.unsqueeze(0).unsqueeze(0)   # (1, 1, n_place, 2)
        p = positions.unsqueeze(2).float()                 # (B, L, 1, 2)
        d = p - c                                          # (B, L, n_place, 2)
        # Wrap to torus in [-gs/2, gs/2]
        d = (d + self.gs / 2.0) % self.gs - self.gs / 2.0
        d2 = (d * d).sum(-1)                               # (B, L, n_place)
        # Normalized 2D Gaussians (1/σ² prefactor) — without this prefactor the
        # narrow and wide Gaussians are equal at d=0, the difference is 0, and
        # the ReLU produces a silently all-zero target. Earlier DOG_RESULTS.md
        # (hex score 0.036) was on broken targets and is uninformative.
        sE2 = self.sigma_E * self.sigma_E
        sI2 = self.sigma_I * self.sigma_I
        gE = (1.0 / sE2) * torch.exp(-d2 / (2.0 * sE2))
        gI = (1.0 / sI2) * torch.exp(-d2 / (2.0 * sI2))
        return torch.relu(gE - gI)

    def prediction_error_loss(self) -> torch.Tensor:
        """MSE between predicted place cells and DoG targets at obs positions."""
        if self._batch_positions is None or self.last_place_pred is None:
            # Fallback: zero loss (shouldn't happen during normal training)
            device = (
                self.last_place_pred.device
                if self.last_place_pred is not None
                else next(self.parameters()).device
            )
            return torch.tensor(0.0, device=device)

        positions = self._batch_positions.to(self.last_place_pred.device)
        targets = self.dog_targets(positions)              # (B, L, n_place)

        B, L, _ = targets.shape
        obs_mask = torch.zeros(B, L, device=targets.device, dtype=torch.bool)
        obs_mask[:, 1::2] = True

        diff = (self.last_place_pred - targets) * obs_mask.unsqueeze(-1).float()
        denom = max(1, int(obs_mask.sum().item()) * targets.shape[-1])
        return diff.pow(2).sum() / denom
