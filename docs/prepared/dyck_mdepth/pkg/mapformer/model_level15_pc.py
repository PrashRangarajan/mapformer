"""MapFormer-WM + Level 1.5 InEKF + PC auxiliary loss (NO Grid).

This is the "non-Grid" twin of model_grid_l15_pc.py. It tests whether
combining the inverse-model (Level 1.5 Kalman) and forward-model (PC
aux loss) corrections without the multi-orientation Grid path
integrator buys anything beyond either alone.

Key differences from MapFormerWM_GridL15PC_Free:
  - Standard single-block-per-ω path integrator (one ω per block, no
    modules / orientations)
  - Plain ``ActionToLieAlgebra`` (one scalar Δ per block, not 2D per
    module) — same as Vanilla / Level15 / PC variants
  - Default d_model=128 (no need for d_model=132 since n_blocks=32 isn't
    constrained by orientation count)
  - Total params ~250K (vs GridL15PC_Free's 302K)

Forward pass:

    token → x = embed(token)                                   [parallel]
    delta = ActionToLieAlgebra(x)                # (B,L,H,NB)   [parallel]
    cum_delta = cumsum(delta, dim=L)                            [O(log L)]
    θ_path = ω · cum_delta                                      [parallel]
    θ̂ = θ_path + InEKFLevel15.scan(...)                         [O(log L)]
    pos_feat = [cos θ̂, sin θ̂]                                  [parallel]
    ô = forward_model(pos_feat)                                 [parallel]
    err = (x - ô) · obs_mask     # accumulate for aux_loss      [parallel]
    RoPE(θ̂) on Q, K → standard causal attention                 [standard]

Use::

    python -m mapformer.train_variant --variant Level15PC \\
        --aux-coef 0.1 --epochs 50 ...

This variant tests the hypothesis that L1.5's inverse-model correction
(state-update step) and PC's forward-model aux loss
(representation-pressure step) are *complementary* signals: one tells
the model "this measurement places you at θ" (Kalman), the other tells
it "this position should produce that observation" (PC). Both target
the same θ̂ from different gradient directions.

If complementary, the combined loss should beat both Level15-alone and
PC-alone. If they overlap, the combined model should match the better
of the two. If they conflict, training might destabilize.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_inekf_level15 import MapFormerWM_Level15InEKF


class MapFormerWM_Level15PC(MapFormerWM_Level15InEKF):
    """MapFormer-WM with Level 1.5 InEKF + PC auxiliary loss (no Grid).

    Inherits the path integrator + L1.5 correction from
    ``MapFormerWM_Level15InEKF``; adds a PC forward model on top of the
    *corrected* angle θ̂ and exposes a ``prediction_error_loss()`` aux
    loss for the training loop.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1,
                 grid_size: int = 64, bottleneck_r: int = 2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)

        # PC forward model: predicts observation embedding from the
        # corrected angle's (cos, sin) features. Same shape as the
        # original PC implementation: input is per-block (cos, sin)
        # flattened over heads × blocks.
        pos_feat_dim = 2 * n_heads * self.n_blocks
        self.forward_model = nn.Sequential(
            nn.Linear(pos_feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

        # Buffer to accumulate aux loss for retrieval after forward pass
        self.prediction_error_squared = torch.tensor(0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)                        # (B, L, d_model)

        # Path integration on the standard (single-block-per-ω) backbone
        delta = self.action_to_lie(x)                     # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)            # (B, L, H, NB)
        theta_path = (
            cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        )                                                  # (B, L, H, NB)

        # Level 1.5 InEKF correction
        theta_hat, Pi, K, R = self.inekf(theta_path, x)   # (B, L, H, NB)

        # PC forward model on the *corrected* θ̂ (same as in
        # MapFormerWM_GridL15PC: representation pressure operates on
        # the model's best position estimate).
        pos_feat = torch.cat(
            [torch.cos(theta_hat), torch.sin(theta_hat)], dim=-1
        )                                                  # (B, L, H, 2*NB)
        pos_feat = pos_feat.reshape(B, L, -1)              # (B, L, H*2*NB)
        obs_pred = self.forward_model(pos_feat)            # (B, L, d_model)

        # Aux loss: ‖x - ô‖² at observation positions (odd indices)
        err_raw = x - obs_pred                             # (B, L, d_model)
        obs_mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        obs_mask[:, 1::2] = True
        mask_f = obs_mask.unsqueeze(-1).to(err_raw.dtype)
        err = err_raw * mask_f
        denom = max(1.0, mask_f.sum().item() * err.shape[-1])
        self.prediction_error_squared = err.pow(2).sum() / denom

        # Save for diagnostics
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K.detach()
        self.last_R = R.detach()
        self.last_pred_err = err_raw.detach()

        # RoPE with corrected θ̂
        theta_for_rope = theta_hat.transpose(1, 2)         # (B, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)

    def prediction_error_loss(self) -> torch.Tensor:
        """Aux loss: ‖prediction error‖² at observation positions.

        Add to next-token loss during training, weighted by aux_coef::

            total_loss = next_token_loss + aux_coef * model.prediction_error_loss()
        """
        return self.prediction_error_squared
