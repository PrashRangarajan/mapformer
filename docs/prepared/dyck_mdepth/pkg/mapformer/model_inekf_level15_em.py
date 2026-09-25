"""
Level 1.5 Parallel Invariant EKF on top of MapFormer-EM.

Same filter machinery as model_inekf_level15.py (constant learnable Pi,
per-token R_t, one scalar affine scan), but applied to the MapFormer-EM
backbone rather than MapFormer-WM.

Why this variant matters:
  The paper reports MapFormer-EM as the stronger of its two variants
  (0.999 vs 0.955 on their clean task). Every correction-mechanism
  variant in this repo (Level 1, Level 1.5, Level 2, PC, ablations) was
  built on MapFormer-WM because WM's single input-dependent rotation
  couples cleanly to the corrected theta. For a fair comparison at the
  strongest backbone, we port the Level 1.5 correction to EM here.

Mechanical difference from WM version:
  - WM: theta_hat is fed into a SINGLE rotation applied to (q, k)
  - EM: theta_hat is fed into rotations applied to q0_pos and k0_pos
    (learnable position vectors), which then produce q_pos, k_pos that
    enter the Hadamard-product attention A_X * A_P.

Everything else (InEKFLevel15 class, the affine scan, R_t head, z head,
wrapping) is reused unchanged from model_inekf_level15.py.
"""

import math
import torch
import torch.nn as nn

from .model import MapFormerEM, EMTransformerLayer, _apply_rope
from .model_inekf_level15 import InEKFLevel15


class MapFormerEM_Level15InEKF(MapFormerEM):
    """MapFormer-EM with Level 1.5 InEKF (constant Pi, per-token R_t, one scan)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 log_R_init_bias: float = 3.0):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        # Reuse the EM transformer layers from the parent (Hadamard attention).
        # The InEKF correction module is the same class used in the WM variant,
        # but with a larger log_R_init_bias so the Kalman correction starts as
        # a near-no-op. Necessary because EM's Hadamard product A_X ⊙ A_P
        # provides no fallback if the position branch is corrupted by random
        # θ̂ corrections at init — see InEKFLevel15 docstring for details.
        #
        # Default 3.0 (K≈0.05 at init) converges reliably across seeds.
        # Slightly more aggressive 5.0 (K≈0.007) can help if a seed is still
        # landing in a sticky basin (currently tested for lm200 seed 2).
        self.inekf = InEKFLevel15(d_model, n_heads, self.n_blocks,
                                  log_R_init_bias=log_R_init_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L) unified interleaved token sequence
        Returns:
            (B, L, vocab_size) next-token logits
        """
        B, L = tokens.shape
        x = self.token_emb(tokens)  # (B, L, d_model)

        # Path integration in the Lie algebra, same as WM backbone
        delta = self.action_to_lie(x)                  # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)         # (B, L, H, NB)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        # theta_path shape: (B, L, H, NB)

        # InEKF correction: d_t = (1-K_t) d_{t-1} + K_t * wrap(z_t - theta_path_t)
        theta_hat, Pi, K, R = self.inekf(theta_path, x)

        # Stash for analysis / diagnostics (consistent with WM variant)
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K.detach()
        self.last_R = R.detach()

        # Convert corrected angles to cos/sin in EM's layout (B, H, L, NB)
        theta_for_rope = theta_hat.transpose(1, 2)     # (B, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        # EM-specific position path: rotate learnable q0_pos, k0_pos by the
        # corrected angle instead of the raw path-integrated angle.
        q0 = self.q0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)  # (B, H, L, d_head)
        k0 = self.k0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        q_pos = _apply_rope(q0, cos_a, sin_a)
        k_pos = _apply_rope(k0, cos_a, sin_a)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            x = layer(x, q_pos, k_pos, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)


class MapFormerEM_Level15InEKF_b5(MapFormerEM_Level15InEKF):
    """Like MapFormerEM_Level15InEKF but with log_R_init_bias=5.0.

    Experiment variant for tightening Level15EM lm200 std (seed 2 outlier).
    Safer init (K≈0.007 at start) keeps the InEKF near-identity for more
    epochs before the R_t head comes online. Registered as "Level15EM_b5"
    in train_variant.VARIANT_MAP.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r, log_R_init_bias=5.0)
