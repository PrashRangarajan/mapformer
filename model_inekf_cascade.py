"""
Two-level Invariant EKF cascade on SO(2).

Adds a slow chunk-level filter on top of Level 1.5's fast per-token filter.
The slow filter operates on chunk-pooled *residual* innovations — what the
fast filter failed to explain — catching systematic drift the fast filter
missed.

Motivation. Hierarchical predictive coding (Bastos et al. 2012) suggests
multi-timescale correction: successive levels of a cortical hierarchy
integrate evidence at increasingly coarse temporal scales. An "optimal"
Kalman filter has zero-mean white residuals; a Kalman filter with a
misspecified prior variance (Level 1.5 uses a *learned constant* Π rather
than the true posterior) has structured residuals that a second-level
filter can exploit.

Design
------

  1. Fast filter (Level 1.5 mechanism, per-token):
       - Constant learnable Π_fast, per-token R_fast_t = exp(MLP_R(x_t))
       - K_fast_t = Π_fast / (Π_fast + R_fast_t)
       - Scalar Hillis-Steele scan → d_fast_t
       - Corrected θ̂_fast_t = θ_path_t + d_fast_t
       - Residual innovation:  ν_resid_t = wrap(z_t − θ̂_fast_t)

  2. Slow filter (per chunk of length C):
       - Pool: ν̄_c = mean(ν_resid_t for t in chunk c)
       - Pool: content̄_c = mean(x_t for t in chunk c)
       - Constant learnable Π_slow, per-chunk R_slow_c = exp(MLP_slow(content̄_c))
       - K_slow_c = Π_slow / (Π_slow + R_slow_c)
       - Second scalar Hillis-Steele scan over n_chunks endpoints → D_slow_c
       - Broadcast: d_slow_t = D_slow_c for t in chunk c (piecewise constant)

  3. Combine: θ̂_t = θ_path_t + d_fast_t + d_slow_t

Both scans are O(log T_effective) at their respective scales; slow scan
runs at length L/C, so overall depth is O(log L + log(L/C)) = O(log L).
Both are pure scalar affine recurrences.

Slow filter is initialised near a no-op (K_slow ≈ 0.05 at start) via a
positive bias on log_R_head_slow — so the model behaves like Level 1.5
at init and only grows the slow correction if there is exploitable
residual structure.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, WMTransformerLayer
from .model_inekf_level15 import assoc_scan_affine_scalar


class InEKFCascade(nn.Module):
    """Two-level cascade: fast per-token Level 1.5 + slow per-chunk correction."""

    def __init__(self, d_model: int, n_heads: int, n_blocks: int,
                 chunk_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_state = n_heads * n_blocks
        self.chunk_size = chunk_size

        # --- Fast filter (Level 1.5 mechanism, inlined) --------------------
        self.log_Pi_fast = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))

        self.log_R_head_fast = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )
        with torch.no_grad():
            self.log_R_head_fast[-1].weight.mul_(0.01)
            self.log_R_head_fast[-1].bias.fill_(0.0)

        # Shared measurement head z_t = π·tanh(MLP(x_t))
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )

        # --- Slow filter ----------------------------------------------------
        self.log_Pi_slow = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))

        # Per-chunk R from pooled content embedding
        self.log_R_head_slow = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.n_state),
        )
        # Bias slow R HIGH at init → K_slow ≈ 0.05, slow filter starts as
        # near no-op. Prevents random slow corrections from destabilising
        # early training.
        with torch.no_grad():
            self.log_R_head_slow[-1].weight.mul_(0.01)
            self.log_R_head_slow[-1].bias.fill_(3.0)

    def _scalar_scan(self, K: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Compute d_t = (1-K_t) d_{t-1} + K_t nu_t via Hillis-Steele."""
        alpha = 1.0 - K
        u = K * nu
        return assoc_scan_affine_scalar(alpha, u)

    def forward(self, theta_path: torch.Tensor, content_emb: torch.Tensor):
        """
        Args:
          theta_path:  (B, L, H, NB)
          content_emb: (B, L, d_model)
        Returns:
          theta_hat: (B, L, H, NB)  — corrected angles for RoPE
          aux: dict with d_fast, d_slow, K_fast, K_slow, R_fast, R_slow
        """
        B, L, H, NB = theta_path.shape
        C = self.chunk_size

        # === FAST FILTER (Level 1.5) =====================================
        log_R_fast = self.log_R_head_fast(content_emb).view(B, L, H, NB).clamp(-5, 5)
        R_fast = log_R_fast.exp()

        z = math.pi * torch.tanh(
            self.measure_head(content_emb).view(B, L, H, NB)
        )

        diff = z - theta_path
        nu_fast = torch.atan2(torch.sin(diff), torch.cos(diff))

        Pi_fast = self.log_Pi_fast.exp().unsqueeze(0).unsqueeze(0)
        K_fast = Pi_fast / (Pi_fast + R_fast).clamp_min(1e-8)

        d_fast = self._scalar_scan(K_fast, nu_fast)
        theta_fast_corrected = theta_path + d_fast

        # === SLOW FILTER on chunk-pooled RESIDUALS ======================
        # Residual innovation: what fast filter failed to explain.
        diff_resid = z - theta_fast_corrected
        nu_resid = torch.atan2(torch.sin(diff_resid), torch.cos(diff_resid))

        # Chunk into blocks of size C. Handle remainder by using the last
        # slow correction for trailing tokens.
        n_chunks_full = L // C
        remainder = L - n_chunks_full * C

        if n_chunks_full > 0:
            L_slow = n_chunks_full * C

            nu_chunks = nu_resid[:, :L_slow].contiguous().view(
                B, n_chunks_full, C, H, NB
            )
            content_chunks = content_emb[:, :L_slow].contiguous().view(
                B, n_chunks_full, C, self.d_model
            )

            # Pool via mean over chunk.
            nu_slow = nu_chunks.mean(dim=2)                       # (B, n_chunks, H, NB)
            content_slow = content_chunks.mean(dim=2)             # (B, n_chunks, d_model)

            log_R_slow = self.log_R_head_slow(content_slow).view(
                B, n_chunks_full, H, NB
            ).clamp(-5, 5)
            R_slow = log_R_slow.exp()

            Pi_slow = self.log_Pi_slow.exp().unsqueeze(0).unsqueeze(0)
            K_slow = Pi_slow / (Pi_slow + R_slow).clamp_min(1e-8)

            D_slow = self._scalar_scan(K_slow, nu_slow)           # (B, n_chunks, H, NB)

            # Piecewise-constant broadcast to per-token.
            d_slow_main = D_slow.unsqueeze(2).expand(
                -1, -1, C, -1, -1
            ).contiguous().view(B, L_slow, H, NB)

            if remainder > 0:
                # Use the last slow correction for trailing tokens.
                tail = D_slow[:, -1:].unsqueeze(2).expand(
                    -1, -1, remainder, -1, -1
                ).contiguous().view(B, remainder, H, NB)
                d_slow = torch.cat([d_slow_main, tail], dim=1)
            else:
                d_slow = d_slow_main
        else:
            # Sequence shorter than chunk_size → slow filter is a no-op.
            d_slow = torch.zeros_like(d_fast)
            K_slow = None
            R_slow = None

        theta_hat = theta_path + d_fast + d_slow

        return theta_hat, {
            "d_fast":   d_fast,
            "d_slow":   d_slow,
            "K_fast":   K_fast,
            "K_slow":   K_slow,
            "R_fast":   R_fast,
            "R_slow":   R_slow,
            "Pi_fast":  Pi_fast,
            "Pi_slow":  self.log_Pi_slow.exp(),
        }


class MapFormerWM_Level15Cascade(MapFormerWM):
    """MapFormer-WM with two-level Kalman cascade correction.

    Same fast filter as Level 1.5, with a slow chunk-level filter added on
    top. See InEKFCascade docstring for the full design.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, chunk_size=32):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = InEKFCascade(d_model, n_heads, self.n_blocks,
                                   chunk_size=chunk_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        theta_hat, aux = self.inekf(theta_path, x)

        # Save for introspection
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_d_fast = aux["d_fast"].detach()
        self.last_d_slow = aux["d_slow"].detach()
        self.last_K_fast = aux["K_fast"].detach()
        if aux["K_slow"] is not None:
            self.last_K_slow = aux["K_slow"].detach()
            self.last_R_slow = aux["R_slow"].detach()

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
