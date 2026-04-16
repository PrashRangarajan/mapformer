"""
MapFormer models: Working Memory (WM) and Episodic Memory (EM) variants.

Faithful implementation of Rambaud et al. (2025):
- SINGLE interleaved token stream s = (a1, o1, a2, o2, ..., aT, oT)
- Unified vocabulary: model must LEARN to disentangle actions from observations
- All tokens projected to BOTH Δ (position) AND Q/K/V (content)
- Path integration via cumsum + learnable angular velocities ω
- Low-rank Δ projection
- Per-head rotations
- EM: Hadamard product of A_X ⊙ A_P
- EM: Rotating learnable p_0 vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .lie_groups import build_block_diagonal_rotations_fast


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position encoding using cos/sin angles (eq. 16).

    Args:
        x: (B, n_heads, T, d_head)
        cos: (B, n_heads, T, n_blocks)
        sin: (B, n_heads, T, n_blocks)

    Returns:
        (B, n_heads, T, d_head) rotated vectors
    """
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    out = torch.stack([out1, out2], dim=-1).reshape_as(x)
    return out


class ActionToLieAlgebra(nn.Module):
    """Low-rank projection from token embeddings to per-head Lie algebra deltas.

    Paper (A.7): W_Δ = W_Δ^out · W_Δ^in
    The model must LEARN that Δ ≈ 0 for observation tokens.
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, bottleneck_r: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.w_in = nn.Linear(d_model, bottleneck_r, bias=False)
        self.w_out = nn.Linear(bottleneck_r, n_heads * n_blocks, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — ANY token embedding (action or obs)
        Returns:
            (batch, seq_len, n_heads, n_blocks) — per-head deltas
        """
        B, T, _ = x.shape
        h = self.w_in(x)
        delta = self.w_out(h)
        return delta.view(B, T, self.n_heads, self.n_blocks)


class PathIntegrator(nn.Module):
    """Compute cumulative rotation angles via cumsum (paper: θ^PI = ω · cumsum(Δ)).

    Learnable angular velocities ω initialized geometrically (Section A.8).
    Index range: i = 1..n_b (paper eq. 17).
    """

    def __init__(self, n_heads: int, n_blocks: int, grid_size: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Geometric initialization (Section A.8, eq. 17)
        # Paper formula as literally written has a sign typo; the paper's own
        # inequality ω_min = ω_max/Δ_max ≤ ω_i ≤ ω_max requires a DECREASING
        # schedule (matching RoPE: θ_i = base^(-2i/d)).
        # ω_i goes from ω_max (highest freq, fine scale) at i=0
        # to ω_max/Δ_max (lowest freq, grid scale) at i=n_b-1.
        omega_max = 2 * math.pi
        delta_max = grid_size
        omega_init = torch.zeros(n_heads, n_blocks)
        for i in range(n_blocks):
            frac = i / max(n_blocks - 1, 1)  # 0 at i=0, 1 at i=n_b-1
            omega_init[:, i] = omega_max * (1.0 / delta_max) ** frac
        self.omega = nn.Parameter(omega_init)

    def forward(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            delta: (B, T, n_heads, n_blocks)
        Returns:
            cos_angles, sin_angles: (B, n_heads, T, n_blocks)
        """
        cum_delta = torch.cumsum(delta, dim=1)
        angles = cum_delta * self.omega.unsqueeze(0).unsqueeze(0)
        angles = angles.transpose(1, 2)  # (B, H, T, n_blocks)
        return torch.cos(angles), torch.sin(angles)


class MapFormerWM(nn.Module):
    """MapFormer Working Memory variant.

    Single interleaved token stream → unified embedding → projected to BOTH
    Δ (position path) and Q/K/V (content path). Model learns disentanglement.
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
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.vocab_size = vocab_size

        # Single unified embedding for all token types
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # All tokens → Δ (model learns obs tokens → ~0)
        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, bottleneck_r)

        # Path integrator with learnable ω
        self.path_integrator = PathIntegrator(n_heads, self.n_blocks, grid_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output: predict next token in unified vocab
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def get_position_state(self, tokens: torch.Tensor):
        """Compute cos/sin of cumulative rotation angles from token sequence."""
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)
        return cos_a, sin_a

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L) unified interleaved token sequence
        Returns:
            (B, L, vocab_size) logits for next token prediction
        """
        B, L = tokens.shape

        x = self.token_emb(tokens)  # (B, L, d_model)

        # Position path: all tokens → Δ → cumsum → ω → cos/sin
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )

        # Content path: transformer with RoPE
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)


class WMTransformerLayer(nn.Module):
    """Transformer layer with MapFormer-WM RoPE-style attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
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

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape

        h = self.norm1(x)

        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)

        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerEM(nn.Module):
    """MapFormer Episodic Memory variant (MapEM-os).

    Single interleaved token stream. Absolute positional encoding with:
    - Separate q_0^p and k_0^p rotated by cumulative rotation
    - Hadamard product: softmax(A_X ⊙ A_P) · V
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
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.vocab_size = vocab_size

        # Single unified embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # All tokens → Δ
        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, bottleneck_r)

        # Path integrator
        self.path_integrator = PathIntegrator(n_heads, self.n_blocks, grid_size)

        # Learnable initial position vectors for Q and K (Section A.7)
        self.q0_pos = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.k0_pos = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            EMTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def get_position_state(self, tokens: torch.Tensor):
        """Compute cos/sin of cumulative rotation angles."""
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)
        return cos_a, sin_a

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape

        x = self.token_emb(tokens)

        # Position path
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)

        # Compute position Q/K by rotating learnable p_0
        q0 = self.q0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
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


class EMTransformerLayer(nn.Module):
    """Transformer layer with Hadamard product of content + structure attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_content = nn.Linear(d_model, d_model)
        self.k_content = nn.Linear(d_model, d_model)
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

    def forward(self, x, q_pos, k_pos, causal_mask):
        B, T, _ = x.shape

        h = self.norm1(x)

        Q_c = self.q_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K_c = self.k_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)

        A_X = torch.matmul(Q_c, K_c.transpose(-1, -2)) / scale
        A_P = torch.matmul(q_pos, k_pos.transpose(-1, -2)) / scale

        # Hadamard product: A_P acts as attention mask on A_X (paper eq. 2)
        scores = A_X * A_P

        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)

        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x
