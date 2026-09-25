"""Level15PC v3: stronger fix attempt — Fix 5 + Fix 6 + tighter R clamp.

The NoBypass v2 variant (`model_level15_pc_v2.py`) added two fixes:
  Fix 5: stop-gradient on the InEKF correction inside the PC aux loss
  Fix 6: mask the aux loss at landmark token positions

These fixed the *gradient route* by which PC's aux loss could drive R
to saturation. But empirically v2 still fails on T=512 OOD (lm200:
0.594, clean: 0.872) because R remains too low overall (mean log_R
≈ -0.5 vs Level15's +0.6) — the indirect route via shared
``action_to_lie`` still pulls R below where the wrap-stabilisation needs
it.

This v3 adds a third fix:

  Fix 7 (tighter R clamp): change the InEKF's log_R clamp from
  [-5, 5] to [-1, 5]. With Π = 1 and log_R lower-bound -1, the maximum
  Kalman gain becomes K = 1/(1 + e^{-1}) ≈ 0.73 (down from 0.99 with
  -5). Even if PC's indirect pressure pushes R lower, the clamp
  prevents saturation past K ≈ 0.73, leaving room for path integration
  to remain stable across the full trajectory.

  Lower bound -1 is chosen because Level 1.5 alone naturally lands
  log_R values in (+0.4, +0.8) range. Allowing log_R down to -1
  preserves Level 1.5's expressive range while clamping NoBypass's
  pathological -3 region.

If v3 succeeds (matches Level15 on lm200 OOD T=512 ≈ 0.82) the failure
mode is mechanistically nailed: PC + Kalman conflict requires a
combination of (a) gradient detachment, (b) landmark-token masking,
(c) bounded gain to coexist.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model_inekf_level15 import (
    InEKFLevel15,
    MapFormerWM_Level15InEKF,
    assoc_scan_affine_scalar,
)
from .model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass


class InEKFLevel15TightClamp(InEKFLevel15):
    """InEKFLevel15 with a tighter clamp on log_R.

    The base class clamps log_R to [-5, 5]. This subclass clamps to a
    user-specified range, default [-1, 5]. Used by v3 to prevent
    R-saturation under PC's indirect gradient pressure.
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int,
                 log_R_init_bias: float = 0.0,
                 log_R_min: float = -1.0,
                 log_R_max: float = 5.0):
        super().__init__(d_model, n_heads, n_blocks, log_R_init_bias)
        self.log_R_min = log_R_min
        self.log_R_max = log_R_max

    def forward(self, theta_path: torch.Tensor, content_emb: torch.Tensor):
        B, L, H, NB = theta_path.shape

        # Per-token R_t with tight clamp
        log_R_raw = self.log_R_head(content_emb).view(B, L, H, NB)
        log_R = log_R_raw.clamp(min=self.log_R_min, max=self.log_R_max)
        R = log_R.exp()

        z = math.pi * torch.tanh(
            self.measure_head(content_emb).view(B, L, H, NB)
        )
        diff = z - theta_path
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))

        Pi = self.log_Pi.exp()
        Pi_b = Pi.unsqueeze(0).unsqueeze(0)
        K = Pi_b / (Pi_b + R).clamp_min(1e-8)

        alpha = 1.0 - K
        u = K * nu
        d = assoc_scan_affine_scalar(alpha, u)

        theta_hat = theta_path + d
        return theta_hat, Pi, K, R


class MapFormerWM_Level15PC_v3(MapFormerWM_Level15PC_NoBypass):
    """Level15PC v3: NoBypass + tighter R clamp [-1, 5].

    Inherits the NoBypass forward pass (Fix 5: stop-gradient through d_t
    inside PC aux loss; Fix 6: landmark-token mask on aux loss). Adds
    Fix 7: replaces the InEKF instance with a tight-clamp version.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1,
                 grid_size: int = 64, bottleneck_r: int = 2,
                 log_R_min: float = -1.0,
                 log_R_max: float = 5.0,
                 landmark_start_id: int = None):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r,
                         landmark_start_id=landmark_start_id)
        # Replace the InEKF with the tight-clamp version
        self.inekf = InEKFLevel15TightClamp(
            d_model, n_heads, self.n_blocks,
            log_R_init_bias=0.0,
            log_R_min=log_R_min, log_R_max=log_R_max,
        )
