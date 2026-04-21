"""
Level 1.5 Parallel Invariant EKF — constant Pi, per-token R_t.

Middle ground between Level 1 (constant K*) and Level 2 (full
heteroscedastic with time-varying Pi + R). Specifically:

  - Pi is a LEARNABLE CONSTANT (not steady-state from DARE — we let the
    gradient pick whatever Pi fits the training distribution).
  - R_t is HETEROSCEDASTIC, from a learned head on the content embedding.
  - K_t = Pi / (Pi + R_t) — varies per token, elementwise (not a scan).
  - State correction d_t = (1-K_t) d_{t-1} + K_t nu_t — a scalar affine
    recurrence with time-varying coefficients. One parallel scan.

This drops ONE of Level 2's two scans (the Mobius covariance scan),
replacing the dynamic Pi with a constant. Rationale: the Level 2
diagnostic showed Pi only varies ~4x across tokens; replacing this
with a single learned value should lose little.

Parallelism:
  - One affine scan with time-varying coefficients, implemented as
    a scalar Hillis-Steele scan over tuples (alpha_t, u_t).
  - O(log L) depth; O(L log L) work in pure PyTorch.
  - Forward pass should match Level 1 (~1-2 ms per batch), and the
    backward pass should be much lighter than Level 2 (one scan vs
    two, with simpler scalar updates).

Compared to Level 1: adds per-token K_t adaptation (should help when
observations vary in informativeness, e.g., landmarks).
Compared to Level 2: gives up Pi dynamics, keeps K_t dynamics and is
~2x faster.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, WMTransformerLayer


def assoc_scan_affine_scalar(alpha: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Parallel scan of scalar affine recurrence.

    Computes d_t = alpha_t * d_{t-1} + u_t for all t, with d_{-1} = 0.

    Uses Hillis-Steele on pairs (alpha, v) representing the cumulative
    affine transform x -> alpha * x + v. Composition rule:
       (alpha_1, v_1) composed with (alpha_2, v_2) applied AFTER is
       (alpha_1 * alpha_2, alpha_2 * v_1 + v_2).

    Args:
      alpha: (B, L, ...) time-varying multiplicative coefficient
      u: (B, L, ...) time-varying additive term

    Returns:
      d: (B, L, ...) with d[t] = cumulative affine applied to 0 = v[t]
    """
    L = alpha.shape[1]
    if L == 1:
        return u.clone()

    a = alpha
    v = u

    step = 1
    while step < L:
        # Pad with identity (alpha=1, v=0) at the front
        ones_pad = torch.ones_like(a[:, :step])
        zeros_pad = torch.zeros_like(v[:, :step])
        a_shifted = torch.cat([ones_pad, a[:, :L - step]], dim=1)
        v_shifted = torch.cat([zeros_pad, v[:, :L - step]], dim=1)

        # Update: new_v[t] = a[t] * v_shifted[t] + v[t]
        #         new_a[t] = a[t] * a_shifted[t]
        # Must compute v update before overwriting a (uses current a)
        v = a * v_shifted + v
        a = a * a_shifted
        step *= 2

    return v


class InEKFLevel15(nn.Module):
    """Level 1.5: constant Pi, per-token R_t -> per-token K_t; one scalar scan."""

    def __init__(self, d_model: int, n_heads: int, n_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_state = n_heads * n_blocks

        # Learnable constant Pi (log-parameterized for positivity)
        self.log_Pi = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))

        # Per-token measurement noise R_t from content embedding
        self.log_R_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )
        # Bias toward moderate R_t at init
        with torch.no_grad():
            self.log_R_head[-1].weight.mul_(0.01)
            self.log_R_head[-1].bias.fill_(0.0)

        # Measurement head (same as Levels 1/2)
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )

    def forward(self, theta_path: torch.Tensor, content_emb: torch.Tensor):
        """
        Args:
          theta_path:  (B, L, H, NB) cumulative angles from path integration
          content_emb: (B, L, d_model)
        Returns:
          theta_hat: (B, L, H, NB) corrected angles
          Pi:        (H, NB) constant covariance (same for all tokens)
          K:         (B, L, H, NB) per-token Kalman gain (for analysis)
          R:         (B, L, H, NB) per-token measurement noise (for analysis)
        """
        B, L, H, NB = theta_path.shape

        # Per-token R_t (clamped for stability)
        log_R_raw = self.log_R_head(content_emb).view(B, L, H, NB)
        log_R = log_R_raw.clamp(min=-5.0, max=5.0)
        R = log_R.exp()

        # Measurement z_t in [-pi, pi]
        z = math.pi * torch.tanh(
            self.measure_head(content_emb).view(B, L, H, NB)
        )

        # Wrapped innovation
        diff = z - theta_path
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))

        # Constant Pi -> per-token K_t
        Pi = self.log_Pi.exp()  # (H, NB)
        Pi_b = Pi.unsqueeze(0).unsqueeze(0)  # broadcast over (B, L)
        K = Pi_b / (Pi_b + R).clamp_min(1e-8)  # (B, L, H, NB)

        # Affine scan: d_t = (1-K_t) d_{t-1} + K_t nu_t
        alpha = 1.0 - K
        u = K * nu
        d = assoc_scan_affine_scalar(alpha, u)

        theta_hat = theta_path + d
        return theta_hat, Pi, K, R


class MapFormerWM_Level15InEKF(MapFormerWM):
    """MapFormer-WM with Level 1.5 InEKF (constant Pi, per-token R_t, one scan)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = InEKFLevel15(d_model, n_heads, self.n_blocks)

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
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
