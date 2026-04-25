"""MapFormer-Grid — architecture that can produce hexagonal grid cells.

Standard MapFormer assigns one path-integrator block per ω frequency.
With ``n_b`` blocks at ``n_b`` distinct frequencies, this is
information-theoretically complete for 2D position (two incommensurate
frequencies can disambiguate any (x, y) modulo the grid wrap), but it
is structurally *incompatible* with hexagonal grid-cell representations.
The hexagonal interference pattern requires three sinusoidal waves at
the SAME frequency superposed at 60° orientation offsets:

    h(x, y) = cos(k₁·r) + cos(k₂·r) + cos(k₃·r)
              with |k₁|=|k₂|=|k₃|, angles(kᵢ, kⱼ) = 60°

MapFormer-Grid replaces the single-block-per-ω structure with *modules*:
each module m has ``n_orientations`` blocks at the same ω_m but with
distinct orientations {0°, 60°, 120°} in 2D action space. When the
hidden state combines three same-module blocks via addition, the result
exhibits hexagonal interference at scale ω_m.

Architectural changes from MapFormer-WM:

  1. ``action_to_lie`` outputs a 2D vector ``(Δ_x, Δ_y)`` per MODULE
     rather than a scalar per block. This gives the network explicit
     access to 2D action information at each scale.

  2. For each block at orientation θ_o within module m:
         Δ_{m,o} = cos(θ_o) · Δ_x + sin(θ_o) · Δ_y
     projects the 2D action onto the block's fixed orientation.

  3. Path-integrated angle per block:
         θ_{m,o,t} = ω_m · cumsum(Δ_{m,o})

  4. The total state dimension is n_modules × n_orientations (must equal
     d_head // 2 for compatibility with the RoPE-style rotation applied to
     per-head Q/K pairs). Default d_model=128, n_heads=2 gives n_blocks=32,
     which factors as 16×2 (square-lattice orientations) but NOT as n×3.

     To test the true hexagonal-optimal {0°, 60°, 120°} configuration,
     either: (a) adjust d_model so n_blocks is divisible by 3 —
     e.g., d_model=132, n_heads=2 gives n_blocks=33 → 11 modules × 3
     orientations; or (b) use n_orientations=6 with d_model adjusted
     accordingly (12 modules × 6 orientations requires n_blocks=72,
     e.g., d_model=288 or n_heads=9 with d_model=162).

     The DEFAULT config here is n_modules=16, n_orientations=2, which
     is NOT hexagonal but is shape-compatible with stock MapFormer and
     serves as a sanity baseline. Override both at construction for
     the hexagonal experiment.

Preserved invariants:

  - Parallel cumsum-based path integration (O(log T) scan depth)
  - Compatible with Level 1.5 InEKF correction on top (same scalar
    affine scan applied to each block)
  - Revisit-only loss, unified token vocabulary, paper-faithful
    training recipe

Expected behaviour if the architecture's theoretical basis holds:

  - Hidden-state rate maps exhibit hexagonal firing fields with
    Sargolini grid scores > 0.3 for at least some dimensions
  - Multi-scale organisation (one hexagonal rate per module), with
    spacings following the geometric ω init
  - Cognitive-map task performance preserved (Level 1.5-Grid should
    match Level 1.5 on the main metrics)

Falsification would indicate either (a) the gradient landscape
disfavours hexagonal solutions even when they are architecturally
accessible, or (b) the three-waves-at-60° optimum requires different
training objectives (e.g., explicit spatial-tuning regularisation)
to be reached.

STATUS: scaffold ready; multi-seed training pending.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import MapFormerWM, WMTransformerLayer


class ActionToLie2D(nn.Module):
    """Low-rank projection from token embeddings to per-module 2D deltas.

    For each module m, produces a 2D vector (Δ_x, Δ_y) representing the
    agent's motion in abstract 2D action space at that scale. Downstream,
    each block at orientation θ_o reads out
        Δ_{m,o} = cos(θ_o) · Δ_x + sin(θ_o) · Δ_y.

    Uses the same paper-faithful low-rank bottleneck (r=2) as
    ActionToLieAlgebra in model.py; with bottleneck=2 and 2D output per
    module, the projection is naturally rank-aligned with 2D motion.
    """

    def __init__(self, d_model: int, n_heads: int, n_modules: int,
                 bottleneck_r: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.n_modules = n_modules
        self.w_in = nn.Linear(d_model, bottleneck_r, bias=False)
        # Output: n_heads * n_modules * 2 (x and y per module)
        self.w_out = nn.Linear(bottleneck_r, n_heads * n_modules * 2,
                               bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, n_heads, n_modules, 2) — per-module 2D deltas
        """
        B, T, _ = x.shape
        h = self.w_in(x)
        out = self.w_out(h).view(B, T, self.n_heads, self.n_modules, 2)
        return out


class GridPathIntegrator(nn.Module):
    """Multi-orientation path integrator with hexagonal-friendly structure.

    Each of ``n_modules`` modules contains ``n_orientations`` blocks at
    the same ω but with orientations ``θ_o ∈ {0°, 60°, 120°, ...}``.
    Total state dimension is n_modules × n_orientations.

    The ω schedule uses the same geometric initialisation as MapFormer:
        ω_m = ω_max · (1/Δ_max)^(m/(n_modules-1))
    with ω_max=2π, Δ_max=grid_size.

    Orientations are FIXED (not learnable) — the three-waves-at-60° optimum
    dictates {0°, 60°, 120°} for n_orientations=3; general k orientations
    use {i·(180°/k) : i=0..k-1}. A learnable-orientation variant is a
    natural follow-up.
    """

    def __init__(self, n_heads: int, n_modules: int, n_orientations: int = 3,
                 grid_size: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_modules = n_modules
        self.n_orientations = n_orientations

        # Geometric ω init, one per module (shared across orientations)
        omega_max = 2 * math.pi
        delta_max = grid_size
        omega_init = torch.zeros(n_heads, n_modules)
        for m in range(n_modules):
            frac = m / max(n_modules - 1, 1)
            omega_init[:, m] = omega_max * (1.0 / delta_max) ** frac
        self.omega = nn.Parameter(omega_init)                  # (H, M)

        # Fixed orientations, uniformly spaced over [0°, 180°)
        # For n_orientations=3: {0°, 60°, 120°} — the hexagonal-optimal set
        orientations = torch.tensor(
            [i * math.pi / n_orientations for i in range(n_orientations)]
        )
        # Buffers, not parameters
        self.register_buffer("cos_orient", torch.cos(orientations))  # (O,)
        self.register_buffer("sin_orient", torch.sin(orientations))  # (O,)

    def forward(self, delta_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            delta_2d: (B, T, H, M, 2) per-module 2D deltas from ActionToLie2D
        Returns:
            cos_angles, sin_angles: each (B, H, T, M, O)
                where last two dims (M, O) can be flattened to "blocks" for
                downstream RoPE-style rotation.
        """
        B, T, H, M, _ = delta_2d.shape
        # Project each module's 2D delta onto each orientation
        # delta_x, delta_y: (B, T, H, M)
        delta_x = delta_2d[..., 0]
        delta_y = delta_2d[..., 1]
        # per-block delta: (B, T, H, M, O)
        d_block = (delta_x.unsqueeze(-1) * self.cos_orient
                   + delta_y.unsqueeze(-1) * self.sin_orient)

        # Cumulative angle per (module, orientation)
        cum_delta = torch.cumsum(d_block, dim=1)              # (B, T, H, M, O)
        # Apply ω (shared across orientations within a module)
        angles = cum_delta * self.omega.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # (B, T, H, M, O)

        angles = angles.transpose(1, 2)                       # (B, H, T, M, O)
        return torch.cos(angles), torch.sin(angles)


class MapFormerWM_Grid(MapFormerWM):
    """MapFormer-WM with grid-cell-capable multi-orientation path integrator.

    Replaces the single-block-per-ω structure with n_modules × n_orientations
    blocks. The total number of rotation "blocks" in the attention path is
    n_blocks := n_modules × n_orientations; we choose this to match or
    slightly exceed the standard n_blocks so the attention head dimension
    is comparable.

    The rest of the architecture (embedding, transformer layers, output
    projection) is unchanged from MapFormerWM.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1,
                 grid_size: int = 64, bottleneck_r: int = 2,
                 n_modules: int = 11, n_orientations: int = 3):
        # MapFormerWM expects n_blocks = d_head // 2. We override the path
        # integrator to use our grid structure but need to match the
        # rotation dimension downstream.
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)

        # Override: our effective n_blocks = n_modules × n_orientations
        n_blocks_effective = n_modules * n_orientations
        assert n_blocks_effective == self.n_blocks, (
            f"n_modules × n_orientations must equal d_head/2 = {self.n_blocks}; "
            f"got {n_modules} × {n_orientations} = {n_blocks_effective}."
        )

        self.n_modules = n_modules
        self.n_orientations = n_orientations

        # Replace the parent's path integrator with the grid version
        self.path_integrator = GridPathIntegrator(
            n_heads=n_heads,
            n_modules=n_modules,
            n_orientations=n_orientations,
            grid_size=grid_size,
        )

        # Replace the parent's ActionToLieAlgebra with the 2D version
        self.action_to_lie = ActionToLie2D(d_model, n_heads, n_modules,
                                           bottleneck_r=bottleneck_r)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)                        # (B, L, d_model)

        # 2D per-module deltas
        delta_2d = self.action_to_lie(x)                  # (B, L, H, M, 2)

        # Path integration with multi-orientation hex structure
        cos_a, sin_a = self.path_integrator(delta_2d)     # (B, H, L, M, O)
        # Flatten module × orientation → blocks for _apply_rope compatibility
        B_, H_, L_, M_, O_ = cos_a.shape
        cos_a = cos_a.reshape(B_, H_, L_, M_ * O_)
        sin_a = sin_a.reshape(B_, H_, L_, M_ * O_)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
