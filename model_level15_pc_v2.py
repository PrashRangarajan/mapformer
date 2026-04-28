"""MapFormer-WM + Level 1.5 InEKF + PC aux loss with two architectural fixes:

  Fix 5 (stop-gradient on the InEKF correction inside the PC aux loss):
    PC's aux loss currently flows back through θ̂ = θ_path + d_t. Gradient
    through d_t lets PC drive R_t → 0 (K → 1), saturating the InEKF and
    making θ̂ ≈ z_t (an autoencoder bypass that destroys path integration).
    We block this by using `θ_path + d_t.detach()` for PC's pos_feat, so
    PC can only improve the aux loss by improving path integration itself,
    not by hijacking the InEKF correction.

  Fix 6 (mask aux loss at landmark token positions):
    Landmark tokens are one-shot (each ID appears at one cell, never seen
    before this trajectory). The forward model g(θ̂) → ô cannot predict
    them, so the aux loss at landmark positions is pure noise gradient.
    Masking those positions prevents this noise from polluting the rest
    of the model — and especially from interfering with R_t's learned
    landmark-vs-aliased differentiation.

These two fixes target the R-saturation mechanism diagnosed in
`R_T_DISTRIBUTION.md` (Test 1 of the interference suite). With Level15PC
showing log_R ≈ -3 across all token types (R near the lower clamp),
the InEKF was effectively overwriting θ_path with z_t at every token.
Fix 5 closes the gradient route that drove this; Fix 6 closes the
noise source that motivated it.

If the diagnosis is right, this variant should:
  - Recover Level 1.5's lm200 OOD T=512 performance (~0.821) on lm200
  - Show R_t distribution comparable to Level 1.5 alone (large spread,
    landmark < aliased < blank)
  - Inherit PC's clone-separation benefit on aliased obs (since PC's aux
    loss now improves path integration cleanly)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_inekf_level15 import MapFormerWM_Level15InEKF


class MapFormerWM_Level15PC_NoBypass(MapFormerWM_Level15InEKF):
    """Level15 InEKF + PC aux loss with stop-gradient + landmark masking.

    Architecturally identical to ``MapFormerWM_Level15PC`` except for two
    bug fixes inside the forward pass:

    1. The PC forward model receives ``theta_path + (theta_hat -
       theta_path).detach()`` rather than ``theta_hat`` directly. This
       makes the InEKF correction visible (so the forward model sees the
       model's best position estimate) but prevents PC's gradient from
       flowing into the R / z / Π parameters of the Kalman update.

    2. The aux loss is masked at *landmark* obs positions in addition to
       the existing action-position mask. Landmark tokens have vocab
       IDs >= ``landmark_start_id``; for the paper-default config (4
       actions + 16 obs types + 1 blank), this is 21.
    """

    # Vocab convention for the paper-default config:
    #   N_ACTIONS = 4 → action tokens 0..3
    #   obs_offset = 4
    #   n_obs_types = 16 → aliased obs at vocab IDs 4..19
    #   blank vocab ID = 4 + 16 = 20
    #   first landmark vocab ID = 4 + 16 + 1 = 21
    #
    # We hardcode 21 here to avoid a constructor change; if you alter
    # n_obs_types or N_ACTIONS, override LANDMARK_START_ID accordingly.
    LANDMARK_START_ID = 21

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1,
                 grid_size: int = 64, bottleneck_r: int = 2,
                 landmark_start_id: int = None):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)

        if landmark_start_id is not None:
            self.LANDMARK_START_ID = landmark_start_id

        # PC forward model: same shape as the original PC implementation.
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
        x = self.token_emb(tokens)                         # (B, L, d_model)

        # Standard MapFormer-WM path integration
        delta = self.action_to_lie(x)                      # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = (
            cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        )                                                   # (B, L, H, NB)

        # Level 1.5 InEKF correction
        theta_hat, Pi, K, R = self.inekf(theta_path, x)    # (B, L, H, NB)

        # ===== Fix 5: stop-gradient on the InEKF correction for PC =====
        # PC's aux loss can only push theta_path (via gradient back through
        # the action_to_lie projection); it cannot push the InEKF parameters
        # (R-head, z-head, Π) because the d_t = theta_hat - theta_path
        # contribution is detached.
        d_t_detached = (theta_hat - theta_path).detach()
        theta_for_pc = theta_path + d_t_detached
        # ===============================================================

        pos_feat = torch.cat(
            [torch.cos(theta_for_pc), torch.sin(theta_for_pc)], dim=-1
        )                                                  # (B, L, H, 2*NB)
        pos_feat = pos_feat.reshape(B, L, -1)              # (B, L, H*2*NB)
        obs_pred = self.forward_model(pos_feat)            # (B, L, d_model)

        # ===== Fix 6: mask aux loss at landmark token positions =====
        # Aux loss is computed only at obs positions (odd indices) AND
        # only at non-landmark obs tokens (vocab id < LANDMARK_START_ID).
        err_raw = x - obs_pred                             # (B, L, d_model)
        obs_mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        obs_mask[:, 1::2] = True
        not_landmark = tokens < self.LANDMARK_START_ID     # (B, L)
        mask = obs_mask & not_landmark                     # (B, L)
        # ===============================================================

        mask_f = mask.unsqueeze(-1).to(err_raw.dtype)
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

        # RoPE attention with corrected θ̂ (full grad here, as in vanilla L15)
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
        """Aux loss to be added to next-token loss with weight ``aux_coef``.

        ``total_loss = next_token_loss + aux_coef * model.prediction_error_loss()``
        """
        return self.prediction_error_squared
