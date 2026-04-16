"""
MapFormer with Invariant Extended Kalman Filter (InEKF) uncertainty tracking.

Extends MapFormer-WM with:
1. Tracked covariance σ²_t per rotation block (scalar per block, for simplicity)
2. Predict step: σ²_t = σ²_{t-1} + Q · ||Δ_t||²  (variance grows with action magnitude)
3. Update step: a learned "landmark gate" detects distinctive observations and
   reduces σ² via a Kalman-like gain
4. Attention temperature modulation: high σ → softer attention

The net effect: when the model encounters noisy actions, its uncertainty grows
so attention softens (broader search over past memories). When a landmark is
detected, uncertainty collapses and attention sharpens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .model import MapFormerWM, WMTransformerLayer, _apply_rope


class InEKFTracker(nn.Module):
    """Lightweight InEKF: tracks scalar σ² per rotation block.

    State: σ²_t ∈ R^{n_heads, n_blocks}
    Predict: σ²_t = σ²_{t-1} + Q_scale · (ω·Δ)²
    Update:  At a "landmark" detection, σ²_t ← σ²_t · (1 - gate·K)
             where K = σ²_t / (σ²_t + R), gate ∈ [0, 1] learned
    """

    def __init__(self, n_heads: int, n_blocks: int, d_model: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Log-parameterize for positivity
        self.log_Q = nn.Parameter(torch.zeros(n_heads, n_blocks))     # process noise scale
        self.log_R = nn.Parameter(torch.zeros(n_heads, n_blocks))     # measurement noise
        self.log_sigma0 = nn.Parameter(torch.full((n_heads, n_blocks), -2.0))  # init σ² ≈ exp(-2)

        # Landmark detector: content embedding → probability that this is a landmark
        self.landmark_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_heads * n_blocks),  # one gate per head/block
            nn.Sigmoid(),
        )

    def forward(self, delta: torch.Tensor, content_emb: torch.Tensor, omega: torch.Tensor):
        """
        Args:
            delta:       (B, T, n_heads, n_blocks) — per-step Δ
            content_emb: (B, T, d_model) — token embeddings for landmark detection
            omega:       (n_heads, n_blocks) — angular velocities

        Returns:
            sigma:       (B, T, n_heads, n_blocks) — tracked per-step variance
            gates:       (B, T, n_heads, n_blocks) — landmark probability per step
        """
        B, T, H, NB = delta.shape

        Q = self.log_Q.exp()       # (H, NB)
        R = self.log_R.exp()
        sigma0 = self.log_sigma0.exp()

        # ω · Δ squared — the per-step rotation magnitude (angle change²)
        theta = delta * omega.unsqueeze(0).unsqueeze(0)   # (B, T, H, NB)
        theta_sq = theta ** 2

        # Landmark gate: probability this token is a landmark (low for blank/actions)
        gates = self.landmark_gate(content_emb)
        gates = gates.view(B, T, H, NB)

        # Sequential Kalman update along time
        sigma_list = []
        sigma = sigma0.unsqueeze(0).expand(B, -1, -1)  # (B, H, NB)
        for t in range(T):
            # Predict: σ²_t = σ²_{t-1} + Q · (ω·Δ)²
            sigma = sigma + Q.unsqueeze(0) * theta_sq[:, t]  # (B, H, NB)

            # Update: if landmark detected, contract σ²
            K = sigma / (sigma + R.unsqueeze(0))  # Kalman gain ∈ (0, 1)
            sigma = sigma * (1.0 - gates[:, t] * K)

            sigma_list.append(sigma)

        sigma = torch.stack(sigma_list, dim=1)  # (B, T, H, NB)
        return sigma, gates


class WMInEKFLayer(nn.Module):
    """Transformer layer with uncertainty-modulated attention."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos_a, sin_a, sigma, causal_mask):
        """
        Args:
            x:      (B, T, d_model)
            cos_a, sin_a: (B, H, T, NB)
            sigma:  (B, T, H, NB) — per-step uncertainty (averaged to scalar per (B,H,T))
            causal_mask: (T, T)
        """
        B, T, _ = x.shape

        h = self.norm1(x)

        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)

        # Mean σ per (B, H, T) — aggregate across blocks
        sigma_bht = sigma.mean(dim=-1).transpose(1, 2)  # (B, H, T)

        # Attention temperature: higher σ → higher temp → softer attention
        # Add 1 to sigma so it doesn't blow up when certain
        temp = 1.0 + sigma_bht  # (B, H, T)
        # Query-side temp (uncertain queries → softer attention out)
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale  # (B, H, T, T)
        scores = scores / temp.unsqueeze(-1)  # divide each query's row by its temperature

        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)

        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerWM_InEKF(MapFormerWM):
    """MapFormer-WM with InEKF uncertainty tracking and uncertainty-modulated attention."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        # Replace WMTransformerLayer with WMInEKFLayer
        self.layers = nn.ModuleList([
            WMInEKFLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        # InEKF uncertainty tracker
        self.inekf = InEKFTracker(n_heads, self.n_blocks, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)                  # (B, L, H, NB)
        cos_a, sin_a = self.path_integrator(delta)     # (B, H, L, NB)

        # Track uncertainty with InEKF
        sigma, gates = self.inekf(delta, x, self.path_integrator.omega)
        # sigma: (B, L, H, NB), gates: (B, L, H, NB)

        # Save for introspection
        self.last_sigma = sigma.detach()
        self.last_gates = gates.detach()

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, sigma, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
