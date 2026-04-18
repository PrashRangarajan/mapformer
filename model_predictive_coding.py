"""
MapFormer with Predictive-Coding corrections.

The InEKF we built is mathematically a predictive coding special case
(Gaussian noise + closed-form optimal gain + learned *inverse* model).
This module drops those commitments in favor of a biologically-inspired
forward-model + prediction-error formulation:

  1. Path integration:        θ_path = cumsum(ω·Δ)            [parallel cumsum]
  2. Forward model:           ô_t    = g(cos θ_path, sin θ_path)  [parallel MLP]
                              "given my position, what should I see?"
  3. Prediction error:        ε_t    = x_t - ô_t               [parallel elementwise]
                              (x_t is the actual token embedding)
  4. Error → state:           δθ_t  = f(ε_t)                   [parallel MLP]
  5. Accumulate corrections:  d     = affine_scan(δθ, gate)    [parallel FFT scan]
  6. Corrected state:         θ̂    = θ_path + d                [parallel]
  7. RoPE on θ̂                                                 [parallel]

Differences from the Kalman (inverse-model) variant:
- Forward model g(θ) handles aliased observations gracefully: if many
  positions give the same observation, they all have the same prediction
  and the error is simply zero — no garbled average.
- Error is in embedding space (d_model), not angle space, so the forward
  model can produce richer predictions than a single angle.
- Correction strength is a learned gate (sigmoid of log_gain), not a
  Kalman gain derived from DARE. Keeps learning end-to-end simple.

Error is masked at *action positions* (even indices in the interleaved
stream) since observations cannot be predicted from a rotation alone
when the token is an action symbol. Only observation tokens carry
position-dependent content worth predicting.

Optional auxiliary loss (`prediction_error_loss()`) encourages the forward
model to actually model observations rather than collapse — mirrors the
Friston Free-Energy Principle training signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .model import MapFormerWM, WMTransformerLayer
from .model_inekf_parallel import parallel_affine_scan


class PredictiveCodingCorrector(nn.Module):
    """Forward-model + error-driven state correction on SO(2) Lie algebra."""

    def __init__(self, d_model: int, n_heads: int, n_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_state = n_heads * n_blocks

        pos_feat_dim = 2 * self.n_state  # (cos θ, sin θ) per block per head

        # Forward model: position → predicted embedding
        self.forward_model = nn.Sequential(
            nn.Linear(pos_feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

        # Error → Lie-algebra correction
        self.error_to_state = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )

        # Learnable scan gate (sigmoid → (0,1))
        # Initialized so gate ≈ 0.15 at start (modest corrections)
        self.log_gain = nn.Parameter(torch.full((n_heads, n_blocks), -1.7))

    def forward(
        self,
        theta_path: torch.Tensor,       # (B, T, H, NB)
        content_emb: torch.Tensor,       # (B, T, d_model)
        obs_position_mask: torch.Tensor, # (B, T) — True at observation positions
    ):
        """
        Returns:
            theta_hat:        (B, T, H, NB)  — corrected angles (feed into RoPE)
            pred_err:         (B, T, d_model) — masked prediction error (for aux loss)
            gate:             (H, NB)         — learned correction gain
        """
        B, T, H, NB = theta_path.shape

        # 2π-invariant position features
        pos_feat = torch.cat(
            [torch.cos(theta_path), torch.sin(theta_path)], dim=-1
        )  # (B, T, H, 2*NB)
        pos_feat = pos_feat.reshape(B, T, -1)  # (B, T, H*2*NB)

        # Forward prediction of observation embedding
        obs_pred = self.forward_model(pos_feat)  # (B, T, d_model)

        # Prediction error — masked to observation positions.
        # At action positions the observation is unpredictable from θ alone,
        # so we zero the error to avoid misleading the corrector.
        err_raw = content_emb - obs_pred
        mask_f = obs_position_mask.unsqueeze(-1).to(err_raw.dtype)
        err = err_raw * mask_f  # (B, T, d_model)

        # Error → per-step Lie-algebra correction
        delta_theta = self.error_to_state(err).view(B, T, H, NB)

        # Accumulate corrections via parallel affine scan (same FFT-conv as InEKF).
        # Scan: d_t = (1-α) d_{t-1} + α δθ_t, with α = sigmoid(log_gain) ∈ (0,1).
        gate = torch.sigmoid(self.log_gain)  # (H, NB)
        d = parallel_affine_scan(delta_theta, gate)  # (B, T, H, NB)

        theta_hat = theta_path + d
        return theta_hat, err_raw, gate  # return unmasked err for diagnostic / aux loss


class MapFormerWM_PredictiveCoding(MapFormerWM):
    """MapFormer-WM with predictive-coding state corrections.

    Preserves MapFormer's parallelism: everything is cumsum, elementwise,
    or FFT-convolution; no sequential Python loop.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        grid_size: int = 64,
        bottleneck_r: int = 2,
    ):
        super().__init__(
            vocab_size, d_model, n_heads, n_layers,
            dropout, grid_size, bottleneck_r,
        )
        # Use plain WMTransformerLayer
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.pc = PredictiveCodingCorrector(d_model, n_heads, self.n_blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        # Path integration (parallel cumsum)
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        # Reconstruct observation-position mask from parity.
        # In our interleaved stream, odd indices (1, 3, 5, …) are observations.
        obs_mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        obs_mask[:, 1::2] = True

        # Predictive-coding correction
        theta_hat, pred_err, gate = self.pc(theta_path, x, obs_mask)

        # Save for introspection + auxiliary loss access
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_pred_err = pred_err.detach()
        self.last_gate = gate.detach()
        # Store un-detached for optional auxiliary loss; masked at obs positions
        mask_f = obs_mask.unsqueeze(-1).to(pred_err.dtype)
        self.prediction_error_squared = (pred_err * mask_f).pow(2).sum() / max(1.0, mask_f.sum() * pred_err.shape[-1])

        # RoPE with corrected θ̂
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

    def prediction_error_loss(self) -> torch.Tensor:
        """Auxiliary loss: ||prediction error||² at observation positions.

        Encourages the forward model to actually predict observations rather
        than collapse to a trivial constant. Use in training as:
            total_loss = next_token_loss + aux_coef * model.prediction_error_loss()
        """
        return self.prediction_error_squared
