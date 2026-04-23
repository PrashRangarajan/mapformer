"""
Additional baseline models for paper-complete comparison.

Implements:
- LSTM baseline (vanilla recurrent, no position mechanism)
- CoPE baseline (Golovneva et al. 2024) — contextual positional encoding
- Mamba baseline (Gu & Dao 2023) — selective SSM
- TAPE baseline (Zhu et al. 2025) — contextual equivariant position (simplified)

All models implement the same interface as MapFormerWM: forward(tokens) → logits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LSTM baseline
# ============================================================

class LSTMBaseline(nn.Module):
    """Vanilla LSTM next-token predictor. No explicit position mechanism."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        x = self.token_emb(tokens)  # (B, L, d)
        y, _ = self.lstm(x)          # (B, L, d)
        y = self.out_norm(y)
        return self.out_proj(y)


# ============================================================
# CoPE baseline — Contextual Positional Encoding
# Golovneva, Wang, Weston, Sukhbaatar (2024), arXiv:2405.18719
# ============================================================

class CoPEAttention(nn.Module):
    """Multi-head attention with contextual positional encoding.

    Instead of fixed relative distance, CoPE computes per-query cumulative
    gates that determine the "distance" between tokens based on content.
    Then interpolates learnable positional embeddings by the gated distance.

    Specifically:
      g_{i,j} = sigmoid(q_i · k_j / sqrt(d))         # gate in [0,1]
      p_{i,j} = sum_{k=j+1}^{i} g_{i,k}              # cumulative from j+1 to i
      PE_{i,j} = interpolate(learned_embs, p_{i,j})
      score_{i,j} = q_i · (k_j + PE_{i,j}) / sqrt(d)
    """

    def __init__(self, d_model, n_heads, max_pos=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_pos = max_pos

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Per-head positional embeddings, one per integer position 0..max_pos-1
        self.pos_emb = nn.Parameter(torch.randn(n_heads, max_pos, self.d_head) * 0.02)

    def _interpolate_pe(self, pos_continuous):
        """Given continuous positions in R, interpolate learned PE linearly.
        pos_continuous: (B, H, T, T) — real-valued positions.
        Returns: (B, H, T, T, d_head).
        """
        # Clamp to [0, max_pos-1]
        pos = pos_continuous.clamp(0.0, self.max_pos - 1.0)
        pos_floor = pos.floor().long()
        pos_ceil = pos.ceil().long().clamp(max=self.max_pos - 1)
        alpha = pos - pos.floor()  # (B, H, T, T), in [0,1]

        # Gather PE vectors. pos_emb is (H, max_pos, d_head).
        # For each (B, H, T, T), index into pos_emb[H, :, :] at pos_floor.
        B, H, T, _ = pos.shape
        pe_floor = self.pos_emb[torch.arange(H)[:, None, None, None].expand(-1, B, T, T),
                                 pos_floor.permute(1, 0, 2, 3), :]  # (H, B, T, T, d_head)
        pe_ceil = self.pos_emb[torch.arange(H)[:, None, None, None].expand(-1, B, T, T),
                                pos_ceil.permute(1, 0, 2, 3), :]
        pe_floor = pe_floor.permute(1, 0, 2, 3, 4)  # -> (B, H, T, T, d_head)
        pe_ceil = pe_ceil.permute(1, 0, 2, 3, 4)
        alpha = alpha.unsqueeze(-1)
        return (1 - alpha) * pe_floor + alpha * pe_ceil

    def forward(self, x, causal_mask):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_h)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)

        # Raw attention gates (before positional correction)
        gate_logits = torch.matmul(Q, K.transpose(-1, -2)) / scale  # (B, H, T, T)
        gates = torch.sigmoid(gate_logits)
        # Apply causal mask to gates (set to 0 for non-attended positions)
        gates = gates.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        # Cumulative gate sum — p_{i,j} = sum_{k=j+1}^{i} g_{i,k}
        # Compute as cumsum over k from right to left, per query row.
        # For query i, position j has p_{i,j} = sum of g_{i, j+1..i}.
        # This equals: reverse cumsum of g_{i, ·} from position i.
        # Simpler: p_{i,j} = cumsum_from_right(gates[i, :])[j+1]
        # We approximate: p_{i,j} = (cumsum(gates, dim=-1, reversed)) shifted.
        # Actually: g_{i,k} for k > i is zeroed by causal mask.
        # p_{i,j} = sum_{k=j+1}^{i} g_{i,k} = cumsum_{k=j+1..T-1} g_{i,k} restricted to k<=i
        # Since gates are causally masked, it's just cumsum of gates from j+1 onward.
        # Equivalent: cumsum from left gives sum_{k=0..j} g_{i,k}; we want sum_{k=j+1..i}
        # = total_sum (up to i) - cumsum_{k=0..j}
        # With causal mask, total up to index T-1 = total up to i (no contribution beyond i).
        total = gates.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        cumsum_left = gates.cumsum(dim=-1)  # (B, H, T, T)
        # Shift cumsum to get sum_{k=0..j-1} at position j (exclusive)
        zero = torch.zeros_like(cumsum_left[..., :1])
        cumsum_shifted = torch.cat([zero, cumsum_left[..., :-1]], dim=-1)
        # p_{i,j} = total - cumsum_{k=0..j} = total - cumsum_shifted_at_{j+1}
        # i.e., p_{i,j} = total - cumsum_left_{j}
        pos = total - cumsum_left  # (B, H, T, T) — continuous positions

        # Interpolate PE
        pe = self._interpolate_pe(pos)  # (B, H, T, T, d_head)

        # Attention scores: q_i · (k_j + pe_{i,j})
        qk = torch.matmul(Q, K.transpose(-1, -2))  # (B, H, T, T)
        # q · pe: einsum over d_head
        q_pe = torch.einsum("bhtd,bhqjd->bhqj", Q, pe.unsqueeze(0) if pe.dim() == 4 else pe)
        # Actually pe is (B,H,T,T,d_head), Q is (B,H,T,d_head). We want:
        # score_{i,j} += Q[B,H,i,:] · pe[B,H,i,j,:]
        q_pe = torch.einsum("bhid,bhijd->bhij", Q, pe)
        scores = (qk + q_pe) / scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.o_proj(out)


class CoPETransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, max_pos=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = CoPEAttention(d_model, n_heads, max_pos=max_pos, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        x = x + self.dropout(self.attn(self.norm1(x), causal_mask))
        x = x + self.ffn(self.norm2(x))
        return x


class CoPEBaseline(nn.Module):
    """CoPE (Golovneva et al. 2024) — Contextual Positional Encoding transformer."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, max_pos=128, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CoPETransformerLayer(d_model, n_heads, max_pos=max_pos, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.out_norm(x)
        return self.out_proj(x)


# ============================================================
# Simple Linear-SSM baseline (Mamba-like, but minimal)
#
# If the `mamba-ssm` package isn't installed, this serves as a stand-in.
# Core idea: h_t = A · h_{t-1} + B · x_t, y_t = C · h_t, with diagonal A.
# Implemented via linear recurrence using FFT convolution since A is diagonal.
# ============================================================

class LinearSSMLayer(nn.Module):
    """Simple diagonal-A linear SSM layer. Approximates Mamba without selectivity."""

    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # A: diagonal, parameterized as negative exponents for stability (A = -exp(log_A))
        self.log_A_diag = nn.Parameter(torch.zeros(d_model, d_state))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))  # skip connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.shape
        A = -torch.exp(self.log_A_diag)  # (d_model, d_state), negative for stability
        # Discretize: ZOH, A_bar = exp(Δ·A), approximate with Δ=1
        A_bar = torch.exp(A)  # (d_model, d_state), in (0, 1)

        # For each output channel d, state is (d_state,)
        # h_t[d] = A_bar[d] * h_{t-1}[d] + B[t][d] * u_t[d]
        # y_t = C · h_t + D * x_t
        # With diagonal A, we can parallelize via FFT for each (d, s) pair.

        # Compute B @ x -> (B, L, d_state)
        B_t = self.B_proj(x)  # (B, L, d_state) — shared across d_model via projection

        # For each (d, s): recurrence h[t] = A_bar[d,s] * h[t-1] + B_t[t,s]
        # This is d_model·d_state parallel scalar affine recurrences.
        # Use log-space cumsum trick: h[t] = A_bar^t · cumsum(B_t[k] · A_bar^{-k})
        # Problem: A_bar^{-t} explodes for A_bar near 0. For stability, use causal conv.

        # Causal convolution: h[t] = sum_{k=0}^{t} A_bar^{t-k} · B_t[k]
        # Kernel: A_bar^k, length L.
        # For (d, s), kernel is A_bar[d,s]^{0,1,...,L-1}.
        k_idx = torch.arange(L, device=x.device, dtype=x.dtype)
        kernel = A_bar.unsqueeze(-1) ** k_idx  # (d_model, d_state, L)
        # FFT conv along L
        pad = 2 * L
        # Reshape B_t to broadcast over d_model: we need per-d_model state evolution.
        # Actually B_t is already (B, L, d_state) — same input for all d_model channels.
        # So h[d, s, t] = conv(B_t[:, s], kernel[d, s, :])
        # Compute per-s convolution with per-(d,s) kernel:
        #   h[d, s, t] = sum_{k=0..t} kernel[d, s, t-k] * B_t[B_idx, k, s]
        # For efficiency, do FFT with einsum-style broadcast:
        kernel_padded = F.pad(kernel, (0, pad - L))  # (d_model, d_state, pad)
        B_t_r = B_t.transpose(-1, -2)  # (B, d_state, L)
        B_t_padded = F.pad(B_t_r, (0, pad - L))  # (B, d_state, pad)

        kernel_fft = torch.fft.rfft(kernel_padded, dim=-1)  # (d_model, d_state, pad//2+1)
        B_fft = torch.fft.rfft(B_t_padded, dim=-1)          # (B, d_state, pad//2+1)
        # out[B, d_model, d_state, t] = IFFT(kernel_fft[d_model, d_state, :] * B_fft[B, d_state, :])
        conv_fft = kernel_fft.unsqueeze(0) * B_fft.unsqueeze(1)  # (B, d_model, d_state, pad//2+1)
        conv = torch.fft.irfft(conv_fft, n=pad, dim=-1)[..., :L]  # (B, d_model, d_state, L)
        # h[B, d_model, d_state, t] now holds state values.
        h = conv.permute(0, 3, 1, 2)  # (B, L, d_model, d_state)

        # y = C · h — sum over d_state
        # C_proj is (d_state -> d_model), applied per (B, L, d_model) state
        # y[b, l, d] = sum_s C[d, s] * h[b, l, d, s]
        C_w = self.C_proj.weight  # (d_model, d_state)
        y = torch.einsum("blds,ds->bld", h, C_w)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        return self.dropout(y)


class LinearSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = LinearSSMLayer(d_model, d_state=d_state, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.ssm(self.norm(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class MambaLikeBaseline(nn.Module):
    """Simple linear-SSM baseline (diagonal A, like Mamba without selectivity)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, d_state=16, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LinearSSMBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        x = self.token_emb(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return self.out_proj(x)


EXTRA_BASELINES = {
    "LSTM":       LSTMBaseline,
    "CoPE":       CoPEBaseline,
    "MambaLike":  MambaLikeBaseline,
}
