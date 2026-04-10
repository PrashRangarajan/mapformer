"""
Baseline models for comparison with MapFormer.

1. TransformerRoPE: Standard Transformer with fixed RoPE (angle = t * theta_base).
   Expected to fail OOD because angles grow unbounded with sequence length.

2. LSTMBaseline: LSTM over interleaved (action, observation) tokens.
   Expected to fail on long sequences due to information bottleneck.

These baselines reproduce the comparison in Rambaud et al. (2025) Table 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FixedRoPE(nn.Module):
    """Standard Rotary Position Embedding (fixed, index-based)."""

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, d_head/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: (batch, n_heads, seq_len, d_head)

        Returns:
            (batch, n_heads, seq_len, d_head) with RoPE applied
        """
        T = x.shape[2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)

        cos = self.cos_cached[:T]  # (T, d_head/2)
        sin = self.sin_cached[:T]

        # Split into pairs and rotate
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.stack([out1, out2], dim=-1).flatten(-2)


class TransformerRoPE(nn.Module):
    """Standard Transformer with fixed RoPE positional encoding.

    Uses index-based rotation angles (t * theta_base) rather than
    action-dependent rotations. Expected to fail on OOD sequence lengths
    because angles go out of range.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        action_vocab: int,
        obs_vocab: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.action_emb = nn.Embedding(action_vocab, d_model)
        self.obs_emb = nn.Embedding(obs_vocab, d_model)
        self.token_type = nn.Embedding(2, d_model)  # 0=action, 1=obs

        self.rope = FixedRoPE(self.d_head)

        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, self.d_head, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, obs_vocab)

    def forward(
        self, actions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        B, T = actions.shape

        # Interleave action and observation embeddings
        a_emb = self.action_emb(actions) + self.token_type(
            torch.zeros(B, T, dtype=torch.long, device=actions.device)
        )
        o_emb = self.obs_emb(observations) + self.token_type(
            torch.ones(B, T, dtype=torch.long, device=actions.device)
        )

        # Use observation embeddings as main sequence (predict next obs)
        x = o_emb + a_emb  # fuse action context into observations

        causal_mask = torch.triu(
            torch.ones(T, T, device=actions.device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            x = layer(x, self.rope, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

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
        self, x: torch.Tensor, rope: FixedRoPE, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = x.shape

        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = rope(Q)
        K = rope(K)

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


class LSTMBaseline(nn.Module):
    """LSTM baseline over interleaved (action, observation) sequences.

    Expected to fail on long sequences due to the information bottleneck
    of the fixed-size hidden state.
    """

    def __init__(
        self,
        d_model: int,
        action_vocab: int,
        obs_vocab: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.action_emb = nn.Embedding(action_vocab, d_model)
        self.obs_emb = nn.Embedding(obs_vocab, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.out_proj = nn.Linear(d_model, obs_vocab)

    def forward(
        self, actions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        # Fuse action and observation embeddings
        x = self.action_emb(actions) + self.obs_emb(observations)
        out, _ = self.lstm(x)
        return self.out_proj(out)
