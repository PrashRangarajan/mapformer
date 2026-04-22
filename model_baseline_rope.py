"""
RoPE baseline — standard transformer with fixed index-based RoPE.

This is the baseline the MapFormer paper compares against (their "RoPE"
rows in Tables 1-2). Unlike MapFormer, rotations are determined by
token index t rather than by the action stream.

Architecture matches MapFormer-WM otherwise (same embedding dim, heads,
vocab, FFN) so the only architectural difference is:

  MapFormer path integration:  θ_t = ω · cumsum(Δ(x_t))
  Standard RoPE:                θ_t = t · base_freqs  (fixed, ignores content)

This is the right baseline for "does input-dependent path integration help?"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, _apply_rope


class MapFormerWM_RoPE(nn.Module):
    """Transformer with fixed RoPE, same architecture as MapFormer-WM otherwise."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, base=10000.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Precompute RoPE base frequencies (standard RoPE: θ_i = base^(-2i/d_head))
        k_idx = torch.arange(self.n_blocks, dtype=torch.float32)
        inv_freq = base ** (-k_idx / max(self.n_blocks - 1, 1))
        self.register_buffer("inv_freq", inv_freq)  # (n_blocks,)

        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _rope_cos_sin(self, L, device, dtype):
        """Return cos, sin of shape (1, n_heads, L, n_blocks) for standard RoPE."""
        t = torch.arange(L, device=device, dtype=dtype)  # (L,)
        # angles[t, i] = t * inv_freq[i]
        angles = torch.outer(t, self.inv_freq.to(device=device, dtype=dtype))
        # Broadcast to (1, H, L, n_blocks) — same rotation for every head
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0).expand(1, self.n_heads, L, -1).contiguous()
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0).expand(1, self.n_heads, L, -1).contiguous()
        return cos, sin

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self._rope_cos_sin(L, tokens.device, x.dtype)
        # Broadcast cos_a, sin_a across batch dim inside _apply_rope.
        # _apply_rope expects (B, H, T, nb) — expand cos/sin accordingly.
        cos_a = cos_a.expand(B, -1, -1, -1)
        sin_a = sin_a.expand(B, -1, -1, -1)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        x = self.out_norm(x)
        return self.out_proj(x)
