"""
MapFormer with a PARALLEL Invariant EKF on SO(2).

Following Särkkä & García-Fernández (2021) and the Marković et al. (2017)
equivalence result: on SO(2), the wrapped-innovation EKF IS the
Lie-Group EKF. Combined with steady-state gain (closed-form DARE), the
filter becomes a scalar affine recurrence — parallelizable via associative
scan or FFT convolution.

Key design:
  1. θ̂⁻ = cumsum(ω·Δ)                            [parallel cumsum, O(log T)]
  2. Measurement head receives (cos θ̂⁻, sin θ̂⁻)  [2π-invariant, fixes length gen]
  3. ẑ_t = π·tanh(measure_head(...))              [bounded SO(2) element]
  4. ν_t = atan2(sin(ẑ - θ̂⁻), cos(ẑ - θ̂⁻))      [parallel elementwise wrap]
  5. d_t = (1-K*)·d_{t-1} + K*·ν_t                [affine scan, FFT conv]
  6. θ̂_t = θ̂⁻_t + d_t                            [parallel]
  7. RoPE uses cos(θ̂_t), sin(θ̂_t)                [parallel]

Steady-state K from closed-form scalar DARE:
  P_post* = (-Q + √(Q² + 4QR)) / 2
  K*      = (P_post* + Q) / (P_post* + Q + R)

Q, R are learnable (log-space for positivity).

This preserves MapFormer's parallelism (everything is either elementwise,
cumsum, or FFT conv) AND respects SO(2) geometry (innovations wrapped),
AND length-generalizes (head is 2π-invariant).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .model import MapFormerWM, WMTransformerLayer


def parallel_affine_scan(nu: torch.Tensor, K_star: torch.Tensor) -> torch.Tensor:
    """Compute d_t = (1-K)·d_{t-1} + K·ν_t in parallel via FFT convolution.

    d_t = K · Σ_{s=0}^{t} (1-K)^(t-s) · ν_s  (causal conv with kernel K·(1-K)^k)

    Args:
        nu:     (B, T, H, NB) — wrapped innovations
        K_star: (H, NB) — steady-state Kalman gain (per head, per block)

    Returns:
        d:      (B, T, H, NB) — corrections to path integration
    """
    B, T, H, NB = nu.shape
    device = nu.device
    dtype = nu.dtype

    alpha = 1.0 - K_star                          # (H, NB)
    # kernel[h, b, k] = K[h,b] · α[h,b]^k,  k = 0..T-1
    k_idx = torch.arange(T, device=device, dtype=dtype)
    alpha_pow = alpha.unsqueeze(-1) ** k_idx       # (H, NB, T)
    kernel = K_star.unsqueeze(-1) * alpha_pow      # (H, NB, T)

    # FFT convolution — pad both to length 2T for linear (not circular) conv.
    pad_len = 2 * T
    kernel_padded = F.pad(kernel, (0, pad_len - T))                # (H, NB, 2T)
    nu_perm = nu.permute(0, 2, 3, 1).contiguous()                   # (B, H, NB, T)
    nu_padded = F.pad(nu_perm, (0, pad_len - T))                    # (B, H, NB, 2T)

    kernel_fft = torch.fft.rfft(kernel_padded, dim=-1)              # (H, NB, pad_len/2+1)
    nu_fft = torch.fft.rfft(nu_padded, dim=-1)                       # (B, H, NB, pad_len/2+1)

    conv_fft = nu_fft * kernel_fft.unsqueeze(0)                      # broadcast over batch
    conv = torch.fft.irfft(conv_fft, n=pad_len, dim=-1)              # (B, H, NB, 2T)

    d = conv[..., :T]                                                 # (B, H, NB, T)
    return d.permute(0, 3, 1, 2).contiguous()                        # (B, T, H, NB)


class InEKFParallelSO2(nn.Module):
    """Steady-state InEKF on SO(2), parallelizable via FFT-based affine scan."""

    def __init__(self, d_model: int, n_heads: int, n_blocks: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Learnable Q, R (log-space for positivity)
        self.log_Q = nn.Parameter(torch.full((n_heads, n_blocks), -3.0))
        self.log_R = nn.Parameter(torch.full((n_heads, n_blocks), 1.0))

        # Measurement head: CONTENT-ONLY.
        # Length generalization comes from wrapping the INNOVATION (atan2),
        # not from conditioning the head on position. Feeding position here
        # would create a degenerate optimum z = θ̂⁻ (filter does nothing).
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )

    def compute_steady_state_K(self) -> torch.Tensor:
        """Closed-form scalar DARE: K* = (P+Q)/(P+Q+R) where P=(-Q+√(Q²+4QR))/2."""
        Q = self.log_Q.exp()
        R = self.log_R.exp()
        P_post = 0.5 * (-Q + torch.sqrt(Q * Q + 4.0 * Q * R))
        P_pred = P_post + Q
        K_star = P_pred / (P_pred + R)
        return K_star  # (H, NB)

    def forward(self, theta_path: torch.Tensor, content_emb: torch.Tensor):
        """
        Args:
            theta_path:  (B, T, H, NB) — path-integrated angles θ̂⁻
            content_emb: (B, T, d_model)

        Returns:
            theta_hat:   (B, T, H, NB) — corrected angles for RoPE
            d:           (B, T, H, NB) — correction deltas (for analysis)
            K_star:      (H, NB)       — steady-state gain used
        """
        B, T, H, NB = theta_path.shape

        # Measurement from content only. Length generalization comes
        # from wrapping the innovation below, not from the head's input.
        z_raw = self.measure_head(content_emb).view(B, T, H, NB)       # (B, T, H, NB)
        z = math.pi * torch.tanh(z_raw)                                # bounded SO(2)

        # Parallel elementwise wrapped innovation
        diff = z - theta_path
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))             # ∈ [-π, π]

        # Steady-state gain (closed-form DARE)
        K_star = self.compute_steady_state_K()                         # (H, NB)

        # Parallel affine scan
        d = parallel_affine_scan(nu, K_star)                           # (B, T, H, NB)

        theta_hat = theta_path + d                                      # corrected angles

        return theta_hat, d, K_star


class MapFormerWM_ParallelInEKF(MapFormerWM):
    """MapFormer-WM with steady-state parallel InEKF.

    Preserves MapFormer's parallelism: everything is either elementwise,
    cumsum, or FFT-convolution — no sequential Python loop.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = InEKFParallelSO2(d_model, n_heads, self.n_blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        # Path integration (parallel cumsum)
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        # Parallel InEKF correction
        theta_hat, d_corr, K_star = self.inekf(theta_path, x)

        # Save for introspection
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_d = d_corr.detach()
        self.last_K_star = K_star.detach()

        # RoPE uses corrected angles — wrap happens naturally via cos/sin
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
