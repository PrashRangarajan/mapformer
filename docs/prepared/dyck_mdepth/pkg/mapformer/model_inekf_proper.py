"""
Proper Invariant Extended Kalman Filter on SO(2) for MapFormer.

True to Barrau & Bonnabel (2017):
  State:      θ_t ∈ R^{n_blocks} (Lie algebra of SO(2)^{n_blocks})
  Predict:    θ_{t+1} = θ_t + ω·Δ_t
              Σ_{t+1} = Σ_t + Q        ← STATE-INDEPENDENT Q (invariant property)
  Update:     ν = z_t - θ̂              (innovation in the Lie algebra)
              K = Σ / (Σ + R)
              θ̂ ← θ̂ + K·ν             ← STATE IS ACTUALLY CORRECTED
              Σ ← (I - K)·Σ

Key differences from the previous fake InEKF:
- Q is a learned constant, NOT scaled by (ω·Δ)² — this is the invariance
- The corrected θ̂ is what feeds into RoPE, so path integration drift is
  actually undone by observations
- Measurement model h(θ) = θ (we treat θ as directly observable through a
  learned content-derived measurement)

This is a faithful per-block scalar InEKF on SO(2). For each rotation block
the algebra is 1D, so the Kalman equations are scalar and don't need matrix
inversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .model import MapFormerWM, WMTransformerLayer, _apply_rope


class InEKFSO2(nn.Module):
    """Per-block scalar InEKF on SO(2)^{n_blocks}.

    Processes a sequence sequentially, producing θ̂_t and Σ_t at each step.
    The path-integration estimate enters through the predicted increment
    (ω·Δ_t). Content embeddings produce the measurement z_t.
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Log-space for positivity. These are STATE-INDEPENDENT constants
        # (invariant EKF property) — one scalar per (head, block).
        self.log_Q = nn.Parameter(torch.full((n_heads, n_blocks), -3.0))  # process noise
        self.log_R = nn.Parameter(torch.full((n_heads, n_blocks), 1.0))   # base measurement noise

        # Measurement model: observation embedding → proposed θ correction.
        # OUTPUT IS IN [-π, π] via tanh·π because θ represents an SO(2)
        # angular class, not an unbounded real. This is the topology
        # correctness fix — innovations are later computed modulo 2π.
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )

        # Informativeness gate: 0 = ignore this observation (e.g., blank),
        # 1 = fully trust (e.g., landmark). Effectively inflates R when gate is low.
        self.gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
            nn.Sigmoid(),
        )

        # Initial uncertainty σ²_0
        self.log_sigma0 = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))

    def forward(self, theta_path: torch.Tensor, content_emb: torch.Tensor):
        """
        Args:
            theta_path:  (B, T, H, NB) — cumulative angles from raw path integration
            content_emb: (B, T, d_model) — for measurement and gating
        Returns:
            theta_hat:   (B, T, H, NB) — Kalman-corrected angles (feed into RoPE)
            sigma:       (B, T, H, NB) — tracked covariance (for analysis)
            gates:       (B, T, H, NB) — learned measurement informativeness
        """
        B, T, H, NB = theta_path.shape
        device = theta_path.device

        Q = self.log_Q.exp()        # (H, NB)
        R = self.log_R.exp()
        sigma0 = self.log_sigma0.exp()

        # Measurement in SO(2): produce an angle in [-π, π] (π·tanh)
        z_raw = self.measure_head(content_emb).view(B, T, H, NB)
        z = math.pi * torch.tanh(z_raw)                        # measurement ∈ [-π, π]
        g = self.gate(content_emb).view(B, T, H, NB)           # informativeness

        # Increments from path integration: Δθ_t = θ_path_t - θ_path_{t-1}
        # These are the "commanded" rotation steps ω·Δ_t.
        incs = torch.zeros_like(theta_path)
        incs[:, 0] = theta_path[:, 0]
        incs[:, 1:] = theta_path[:, 1:] - theta_path[:, :-1]

        theta_hat_list = []
        sigma_list = []

        theta_hat = torch.zeros(B, H, NB, device=device)              # starts at origin
        sigma = sigma0.unsqueeze(0).expand(B, -1, -1).clone()          # initial covariance

        for t in range(T):
            # --- PREDICT ---
            theta_hat = theta_hat + incs[:, t]          # θ̂ <- θ̂ + ω·Δ_t
            sigma = sigma + Q.unsqueeze(0)               # Σ <- Σ + Q (STATE-INDEPENDENT)

            # --- UPDATE ---
            # INNOVATION IN SO(2): compute the shortest angular distance,
            # not the arithmetic difference in R. This respects the circular
            # topology of SO(2) and makes the filter length-invariant.
            diff = z[:, t] - theta_hat
            innov = torch.atan2(torch.sin(diff), torch.cos(diff))  # ∈ [-π, π]

            # Effective measurement noise inflated by (1 - gate) — low gate ≈ ignore
            R_eff = R.unsqueeze(0) / (g[:, t] + 1e-4)
            K = sigma / (sigma + R_eff)                  # Kalman gain ∈ (0, 1)

            theta_hat = theta_hat + K * innov            # STATE IS CORRECTED
            sigma = (1.0 - K) * sigma                     # Σ <- (I - K) Σ

            theta_hat_list.append(theta_hat)
            sigma_list.append(sigma)

        theta_hat_out = torch.stack(theta_hat_list, dim=1)  # (B, T, H, NB)
        sigma_out = torch.stack(sigma_list, dim=1)
        return theta_hat_out, sigma_out, g


class MapFormerWM_ProperInEKF(MapFormerWM):
    """MapFormer-WM with a proper Invariant EKF that corrects the path-integrated state.

    The KEY CHANGE from the previous fake InEKF:
    - θ̂_t (Kalman-corrected angles) are used for the RoPE rotations.
      Drift is actively corrected by observations, not just flagged via σ².
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        # Use plain WMTransformerLayer (no uncertainty modulation — the
        # point is that state correction is enough).
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = InEKFSO2(d_model, n_heads, self.n_blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        # Raw path integration
        delta = self.action_to_lie(x)                # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        # theta_path: (B, L, H, NB)

        # InEKF corrects the path-integrated angles using content observations
        theta_hat, sigma, gates = self.inekf(theta_path, x)

        # Save for introspection
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_sigma = sigma.detach()
        self.last_gates = gates.detach()

        # RoPE from the CORRECTED angles
        # theta_hat: (B, L, H, NB) -> (B, H, L, NB)
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
