"""Ablation: Level15_Hopfield with the main attention's position channel removed.

The clean test of MapFormer's core design choice. `Level15_Hopfield` keeps
BOTH (a) position-modulated main attention (MapFormer's entangled design:
θ̂ rotates the main attention's Q/K) and (b) an explicit position-keyed
Hopfield retrieval head (TEM's factored design).

This variant removes (a): the main transformer attention gets an IDENTITY
rotation (cos=1, sin=0), so it becomes pure content attention. Position
information enters the model EXCLUSIVELY through the explicit Hopfield head.

Tests whether MapFormer's signature position-modulated attention is even
necessary once an explicit position-keyed memory exists. If this ablation
matches the full Level15_Hopfield, the entangled design was redundant —
the factored (TEM-style) design is sufficient.

Path integration + InEKF correction still run (θ̂ is still computed and
still consumed by the Hopfield head). The ONLY thing removed is the
rotation of the main attention's Q/K.
"""

from __future__ import annotations

import torch

from .model_inekf_level15_hopfield import MapFormerWM_Level15_Hopfield


class MapFormerWM_Level15_Hopfield_NoMainAP(MapFormerWM_Level15_Hopfield):
    """Level15_Hopfield with the main attention's position rotation removed.

    Position reaches the model only via the explicit Hopfield head.
    """

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

        # --- ABLATION: main attention gets IDENTITY rotation (pure content) ---
        # theta_hat layout for the layers is (B, H, L, NB). cos=1, sin=0 makes
        # _apply_rope a no-op, so the main attention is content-only.
        theta_for_rope = theta_hat.transpose(1, 2)
        cos_a = torch.ones_like(theta_for_rope)
        sin_a = torch.zeros_like(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        # Hopfield head DOES use the real theta_hat — position enters here only.
        x_hop = self._hopfield_retrieve(x, theta_hat)
        x = x + x_hop

        x = self.out_norm(x)
        return self.out_proj(x)
