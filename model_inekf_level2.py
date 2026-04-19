"""
Level 2 Parallel Invariant EKF on SO(2) — heteroscedastic measurement noise.

Extends Level 1 (model_inekf_parallel.py) by letting the measurement noise
R_t vary per token, output by a learned head. The Kalman gain K_t then
varies accordingly: large R_t (uninformative observation) -> K_t near 0;
small R_t (landmark) -> K_t near 1. This automatically down-weights blank
tokens and up-weights landmark tokens, without any hand-crafted gating.

Parallelism is preserved via two associative-scan operations over 2x2
matrices (Mamba-style, Hillis-Steele implementation in PyTorch):

  Scan 1 — Covariance (Mobius composition):
    Pi_{t+1} = ((R_t + Q) Pi_t + Q R_t) / (Pi_t + R_t + Q)
    Written as (a, b, c, d) Mobius: apply matrix multiplication.

  Scan 2 — State correction (affine composition with time-varying gain):
    d_t = (1 - K_t) d_{t-1} + K_t nu_t

Both scans are O(log L) depth via Hillis-Steele; O(L log L) work in pure
PyTorch (can be dropped to O(L) with a custom CUDA kernel a la Mamba, but
our sequence lengths are modest so this is not worth it here).

Identity used: if Pi_t is the "prior" covariance before the update at
step t (i.e., after the predict step from t-1), then
    K_t = Pi_t / (Pi_t + R_t)
and Pi_{t+1} = (1 - K_t) Pi_t + Q = Pi_t R_t / (Pi_t + R_t) + Q.
Also K_t R_t = Pi_t - (1 - K_t) Pi_t = K_t Pi_t gives no new info; we
compute K_t directly from Pi_t above.

Initial Pi_0 is a learnable scalar (per head, per block).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, WMTransformerLayer


# ------------------------------------------------------------
# Associative scan over 2x2 matrices (Hillis-Steele)
# ------------------------------------------------------------

def _expand_eye(shape, device, dtype):
    """Build identity matrices broadcastable to `shape[-2:]=(2,2)` over the rest."""
    eye = torch.eye(2, device=device, dtype=dtype)
    return eye.expand(shape).contiguous()


def assoc_scan_matmul(mats: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Cumulative left-multiply scan of 2x2 matrices along `dim`.

    out[..., t, ...] = mats[..., t, ...] @ mats[..., t-1, ...] @ ... @ mats[..., 0, ...]

    Hillis-Steele scan: log(L) iterations, each a batched matmul.
    Output shape matches input shape. Assumes matrix dims are the last two.
    """
    L = mats.shape[dim]
    if L == 1:
        return mats

    step = 1
    while step < L:
        pad_shape = list(mats.shape)
        pad_shape[dim] = step
        eye = _expand_eye(pad_shape, mats.device, mats.dtype)
        padded = torch.cat([eye, mats], dim=dim)
        shifted = padded.narrow(dim, 0, L)
        mats = torch.matmul(mats, shifted)
        # Normalize by max-abs to avoid overflow in long scans.
        # (Mobius matrices are scale-invariant; affine matrices with bottom row
        # [0, 1] aren't, so we only normalize if bottom row is not all [0,1].)
        step *= 2
    return mats


def assoc_scan_matmul_normalized(mats: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Like assoc_scan_matmul but normalizes by max-abs after each step.

    Safe for Mobius matrices (scale-invariant under Mobius action). Do NOT use
    for affine matrices with fixed bottom row [0, 1] — normalization breaks them.
    """
    L = mats.shape[dim]
    if L == 1:
        return mats

    step = 1
    while step < L:
        pad_shape = list(mats.shape)
        pad_shape[dim] = step
        eye = _expand_eye(pad_shape, mats.device, mats.dtype)
        padded = torch.cat([eye, mats], dim=dim)
        shifted = padded.narrow(dim, 0, L)
        mats = torch.matmul(mats, shifted)
        # Per-matrix max-abs normalization
        max_abs = mats.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        mats = mats / max_abs
        step *= 2
    return mats


# ------------------------------------------------------------
# Level 2 InEKF module
# ------------------------------------------------------------

class InEKFLevel2(nn.Module):
    """Heteroscedastic Level 2 InEKF on SO(2).

    Parameters:
      - log_Q:          constant process noise per (head, block)
      - log_P0:         initial covariance Pi_0 per (head, block)
      - log_R_head:     MLP producing log R_t from content embedding
      - measure_head:   MLP producing z_t from content embedding (wrapped to [-pi, pi])
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_state = n_heads * n_blocks

        # Constant process noise Q (positive via exp)
        self.log_Q = nn.Parameter(torch.full((n_heads, n_blocks), -3.0))

        # Initial covariance Pi_0 (positive via exp)
        self.log_P0 = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))

        # Per-token measurement noise R_t (positive via exp, clipped for stability)
        self.log_R_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )

        # Measurement head: same design as Level 1 (content -> z in [-pi, pi])
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )

        # Bias log_R toward a moderate initial value to avoid degenerate
        # extreme gains at init.
        with torch.no_grad():
            self.log_R_head[-1].weight.mul_(0.01)
            self.log_R_head[-1].bias.fill_(0.0)

    def forward(self, theta_path: torch.Tensor, content_emb: torch.Tensor):
        """
        Args:
          theta_path:  (B, L, H, NB) cumulative angles from path integration
          content_emb: (B, L, d_model)
        Returns:
          theta_hat: (B, L, H, NB) Kalman-corrected angles (feed into RoPE)
          Pi:        (B, L, H, NB) prior covariance per step (for analysis)
          K:         (B, L, H, NB) Kalman gain per step (for analysis)
          R:         (B, L, H, NB) measurement noise per step (for analysis)
        """
        B, L, H, NB = theta_path.shape
        device = theta_path.device
        dtype = theta_path.dtype

        # Heteroscedastic R_t, clipped to sane range to prevent numerical explosion
        log_R_raw = self.log_R_head(content_emb).view(B, L, H, NB)
        log_R = log_R_raw.clamp(min=-5.0, max=5.0)
        R = log_R.exp()  # (B, L, H, NB), all positive

        # Measurement z_t in [-pi, pi] (SO(2) valid)
        z_raw = self.measure_head(content_emb).view(B, L, H, NB)
        z = math.pi * torch.tanh(z_raw)

        # Innovations (wrapped)
        diff = z - theta_path
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))  # [-pi, pi]

        Q = self.log_Q.exp()  # (H, NB)
        P0 = self.log_P0.exp()  # (H, NB)

        # --- Scan 1: Covariance via Mobius matrices -------------------
        # Pi_{t+1} = ((R_t + Q) Pi_t + Q R_t) / (1 * Pi_t + (R_t + Q))
        # Mobius matrix M_t = [[R_t + Q,  Q * R_t],
        #                      [1,        R_t + Q]]
        # Note: det = (R_t + Q)^2 - Q R_t = R_t^2 + R_t Q + Q^2 > 0
        # (Using (R_t + Q) both on the diagonal for a symmetric structure)
        # Actually let's rederive carefully:
        #
        #     Pi_{t+1} = (Pi_t R_t) / (Pi_t + R_t) + Q
        #             = (Pi_t R_t + Q (Pi_t + R_t)) / (Pi_t + R_t)
        #             = ((R_t + Q) Pi_t + Q R_t) / (Pi_t + R_t)
        #
        # So M_t = [[R_t + Q, Q R_t],
        #           [1,       R_t + Q ... wait]]
        # Denominator is (Pi_t + R_t), not (Pi_t + R_t + Q). Let me redo.
        #
        # Pi_{t+1} = (Pi_t R_t / (Pi_t + R_t)) + Q = [(Pi_t R_t) + Q (Pi_t + R_t)] / (Pi_t + R_t)
        #         = [(R_t + Q) Pi_t + Q R_t] / [Pi_t + R_t]
        # So M_t entries: a = R_t + Q, b = Q R_t, c = 1, d = R_t
        Q_b = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, H, NB) broadcast
        a = R + Q_b           # (B, L, H, NB)
        b = Q_b * R           # (B, L, H, NB)
        c = torch.ones_like(R)
        d = R.clone()         # (B, L, H, NB)

        # Stack into (B, L, H, NB, 2, 2)
        row0 = torch.stack([a, b], dim=-1)
        row1 = torch.stack([c, d], dim=-1)
        M = torch.stack([row0, row1], dim=-2)  # (B, L, H, NB, 2, 2)

        # We want Pi_t (prior covariance at step t). Convention:
        # Pi_0 = P0.  Pi_t = M_{t-1}(...M_0(P0)) for t >= 1.
        # So we prepend identity to M so that cum[0] applied to P0 gives P0,
        # and cum[t] applied to P0 gives Pi_{t+1}. Then we shift by one.
        eye = _expand_eye((B, 1, H, NB, 2, 2), device, dtype)
        M_ext = torch.cat([eye, M[:, :-1]], dim=1)  # length L
        # Now cum[0] = I, cum[1] = M_0, cum[2] = M_1 @ M_0, ...
        # cum[t] applied to P0 gives Pi_t.
        cum = assoc_scan_matmul_normalized(M_ext, dim=1)

        # Apply Mobius to P0
        a_c = cum[..., 0, 0]
        b_c = cum[..., 0, 1]
        c_c = cum[..., 1, 0]
        d_c = cum[..., 1, 1]
        P0_b = P0.unsqueeze(0).unsqueeze(0)  # broadcast
        Pi = (a_c * P0_b + b_c) / (c_c * P0_b + d_c).clamp_min(1e-8)

        # Safety clamp: Pi must stay positive and not blow up
        Pi = Pi.clamp(min=1e-8, max=1e6)

        # --- Kalman gain K_t = Pi_t / (Pi_t + R_t) -------------------
        K = Pi / (Pi + R).clamp_min(1e-8)

        # --- Scan 2: State correction (affine with time-varying coefficients) -
        # d_t = (1 - K_t) d_{t-1} + K_t nu_t
        # Affine: [d_t; 1] = [[alpha_t, u_t],
        #                    [0,       1   ]] [d_{t-1}; 1]
        # cum_A[t] = A_t A_{t-1} ... A_0, apply to [0; 1], d_t = cum_A[t][0, 1]
        alpha = (1.0 - K)          # (B, L, H, NB)
        u = K * nu                  # (B, L, H, NB)
        zeros = torch.zeros_like(alpha)
        ones = torch.ones_like(alpha)
        row0a = torch.stack([alpha, u], dim=-1)
        row1a = torch.stack([zeros, ones], dim=-1)
        A = torch.stack([row0a, row1a], dim=-2)  # (B, L, H, NB, 2, 2)

        cum_A = assoc_scan_matmul(A, dim=1)  # do not normalize affine matrices
        d = cum_A[..., 0, 1]  # (B, L, H, NB), since initial state is [0; 1]

        theta_hat = theta_path + d
        return theta_hat, Pi, K, R


# ------------------------------------------------------------
# Model: Level 2 InEKF MapFormer
# ------------------------------------------------------------

class MapFormerWM_Level2InEKF(MapFormerWM):
    """MapFormer-WM with Level 2 (heteroscedastic R_t) parallel InEKF.

    Preserves MapFormer's O(log L) depth via two associative scans of 2x2
    matrices. Same structure as Level 1 but lets the Kalman gain vary per
    token based on observation informativeness (learned from the token
    embedding).
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = InEKFLevel2(d_model, n_heads, self.n_blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        # Path integration (same as Level 1)
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        # Level 2 InEKF: Pi_t, K_t, R_t all vary per token
        theta_hat, Pi, K, R = self.inekf(theta_path, x)

        # Save for introspection / diagnostics
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K.detach()
        self.last_R = R.detach()

        # RoPE with corrected angles
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
