"""
MapFormer models: Working Memory (WM) and Episodic Memory (EM) variants.

MapFormer-WM: Relative positional encoding (generalises RoPE).
    Rotation comes from action stream, applied to Q/K in attention.

MapFormer-EM: Absolute positional encoding with parallel attention streams.
    Separate content (AX) and structure (AP) attention, combined additively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .lie_groups import build_block_diagonal_rotations_fast, exp_map_2d
from .prefix_scan import parallel_prefix_product


class ActionToLieAlgebra(nn.Module):
    """Maps action token embeddings to Lie algebra elements.

    Simple linear projection -- no activation function.
    The unconstrained output is then exponentiated to get valid rotation matrices.
    """

    def __init__(self, action_dim: int, n_rot_dims: int):
        super().__init__()
        self.proj = nn.Linear(action_dim, n_rot_dims)
        self.n_rot = n_rot_dims

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_tokens: (batch, seq_len, action_dim)
        Returns:
            (batch, seq_len, n_rot_dims) -- delta in Lie algebra
        """
        return self.proj(action_tokens)


class RotaryPositionEncoding(nn.Module):
    """Apply learned rotation matrices to queries and keys (generalised RoPE).

    Standard RoPE: rotation angle = t * theta_base (fixed, index-based)
    MapFormer-WM: rotation = P_t (learned, action-based, from prefix scan)
    """

    def __init__(self, d_head: int, n_rot_dims: int):
        """
        Args:
            d_head: dimension of each attention head
            n_rot_dims: number of 2x2 rotation blocks (must equal d_head // 2)
        """
        super().__init__()
        self.d_head = d_head
        self.n_rot = n_rot_dims
        assert d_head == 2 * n_rot_dims, \
            f"d_head ({d_head}) must equal 2 * n_rot_dims ({2*n_rot_dims})"

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply position-dependent rotation to input vectors.

        Args:
            x: (batch, seq_len, d_head) queries or keys
            positions: (batch, seq_len, d_head, d_head) cumulative rotation matrices

        Returns:
            (batch, seq_len, d_head) rotated vectors
        """
        # x: (B, T, d) -> (B, T, d, 1)
        x_col = x.unsqueeze(-1)
        # positions @ x: (B, T, d, d) @ (B, T, d, 1) -> (B, T, d, 1)
        rotated = torch.matmul(positions, x_col).squeeze(-1)
        return rotated


class MapFormerWM(nn.Module):
    """MapFormer Working Memory variant.

    Uses relative positional encoding: Q and K are rotated by the
    cumulative position matrix P_t, so attention naturally computes
    similarity based on relative displacement P_t @ P_s^{-1}.

    This is a strict generalisation of RoPE where the rotation comes
    from the learned action stream rather than the token index.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_rot_dims: int,
        action_vocab: int,
        obs_vocab: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_rot_dims = n_rot_dims

        # Embeddings
        self.action_emb = nn.Embedding(action_vocab, d_model)
        self.obs_emb = nn.Embedding(obs_vocab, d_model)

        # Action -> Lie algebra
        self.action_to_lie = ActionToLieAlgebra(d_model, n_rot_dims)

        # Rotary encoding
        self.rope = RotaryPositionEncoding(self.d_head, n_rot_dims)

        # Transformer layers (custom attention with RoPE)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, n_rot_dims, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, obs_vocab)

    def get_position_state(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute cumulative position matrices from actions.

        Args:
            actions: (batch, seq_len) action indices

        Returns:
            (batch, seq_len, d_head, d_head) cumulative rotation matrices
        """
        a_emb = self.action_emb(actions)
        delta = self.action_to_lie(a_emb)  # (B, T, n_rot)
        M = build_block_diagonal_rotations_fast(delta)  # (B, T, d_head, d_head)
        P = parallel_prefix_product(M)  # (B, T, d_head, d_head)
        return P

    def forward(
        self, actions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            actions: (batch, seq_len) action indices
            observations: (batch, seq_len) observation indices

        Returns:
            (batch, seq_len, obs_vocab) logits for next observation prediction
        """
        B, T = actions.shape

        # Compute position matrices
        P = self.get_position_state(actions)  # (B, T, d_head, d_head)

        # Content embeddings
        x = self.obs_emb(observations)  # (B, T, d_model)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=actions.device, dtype=torch.bool), diagonal=1
        )

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, P, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)


class WMTransformerLayer(nn.Module):
    """Single transformer layer with MapFormer-WM attention."""

    def __init__(self, d_model: int, n_heads: int, n_rot_dims: int, dropout: float):
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

    def forward(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Pre-norm attention
        h = self.norm1(x)

        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head)

        # Apply position-dependent rotation to Q and K (per head)
        # P: (B, T, d_head, d_head)
        # Q[..., h, :]: (B, T, d_head) -> rotate by P
        Q_rot = torch.zeros_like(Q)
        K_rot = torch.zeros_like(K)
        for head in range(self.n_heads):
            q_h = Q[:, :, head, :]  # (B, T, d_head)
            k_h = K[:, :, head, :]
            Q_rot[:, :, head, :] = torch.matmul(
                P, q_h.unsqueeze(-1)
            ).squeeze(-1)
            K_rot[:, :, head, :] = torch.matmul(
                P, k_h.unsqueeze(-1)
            ).squeeze(-1)

        # Scaled dot-product attention
        # (B, n_heads, T, d_head)
        Q_rot = Q_rot.transpose(1, 2)
        K_rot = K_rot.transpose(1, 2)
        V = V.transpose(1, 2)

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q_rot, K_rot.transpose(-1, -2)) / scale

        # Apply causal mask
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)

        x = x + self.dropout(out)

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x


class MapFormerEM(nn.Module):
    """MapFormer Episodic Memory variant.

    Uses absolute positional encoding with two parallel attention streams:
    - Content stream (AX): standard attention over observation embeddings
    - Structure stream (AP): attention based on absolute position matrices

    The two streams are combined additively in the attention scores.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_rot_dims: int,
        action_vocab: int,
        obs_vocab: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_rot_dims = n_rot_dims

        self.action_emb = nn.Embedding(action_vocab, d_model)
        self.obs_emb = nn.Embedding(obs_vocab, d_model)
        self.action_to_lie = ActionToLieAlgebra(d_model, n_rot_dims)

        # Position embedding: flatten P matrix to vector, project to d_model
        self.pos_proj = nn.Linear(self.d_head * self.d_head, d_model)

        self.layers = nn.ModuleList([
            EMTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, obs_vocab)

    def get_position_state(self, actions: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_emb(actions)
        delta = self.action_to_lie(a_emb)
        M = build_block_diagonal_rotations_fast(delta)
        P = parallel_prefix_product(M)
        return P

    def forward(
        self, actions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        B, T = actions.shape

        P = self.get_position_state(actions)  # (B, T, d_head, d_head)

        # Flatten position matrices and project
        P_flat = P.reshape(B, T, -1)  # (B, T, d_head^2)
        pos_emb = self.pos_proj(P_flat)  # (B, T, d_model)

        # Content embeddings
        content = self.obs_emb(observations)

        causal_mask = torch.triu(
            torch.ones(T, T, device=actions.device, dtype=torch.bool), diagonal=1
        )

        x = content
        for layer in self.layers:
            x = layer(x, pos_emb, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)


class EMTransformerLayer(nn.Module):
    """Transformer layer with parallel content + structure attention (EM variant)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Content attention projections
        self.q_content = nn.Linear(d_model, d_model)
        self.k_content = nn.Linear(d_model, d_model)

        # Structure attention projections
        self.q_struct = nn.Linear(d_model, d_model)
        self.k_struct = nn.Linear(d_model, d_model)

        # Shared value and output
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

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        h = self.norm1(x)

        # Content stream: Q_c, K_c from observation embeddings
        Q_c = self.q_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K_c = self.k_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Structure stream: Q_s, K_s from position embeddings
        Q_s = self.q_struct(pos_emb).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K_s = self.k_struct(pos_emb).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)

        # Combined attention: content similarity + structural affinity
        scores_content = torch.matmul(Q_c, K_c.transpose(-1, -2)) / scale
        scores_struct = torch.matmul(Q_s, K_s.transpose(-1, -2)) / scale
        scores = scores_content + scores_struct

        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)

        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))

        return x
