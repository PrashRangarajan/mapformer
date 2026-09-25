"""
Hierarchical (two-scale) attention for MapFormer-WM.

Tests whether a HIERARCHY ON THE ATTENTION ITSELF helps, as opposed to the
Kalman cascade (which layered a second correction on theta and did NOT help).

Flat MapFormer attention is a single softmax over all past tokens. At long T
this dilutes: with ~128 aliased cells per obs type, the softmax spreads mass
over many equally-plausible candidates. A two-scale attention decomposes
retrieval into:

  - LOCAL (fine): each query attends to a causal window of the last W tokens.
    Sharp, recent, full resolution.
  - COARSE (long-range): the past is pooled into chunk summaries (mean of the
    rotated K / V over each chunk of size C); the query attends over all
    chunk summaries that lie ENTIRELY in its past. Cheap reach to far-back
    regions; combats dilution by routing at chunk granularity.

Output = o_proj(local_out + coarse_out). Everything else (RoPE from the
InEKF-corrected theta, norms, FFN, residuals) is identical to the flat layer,
so the ONLY difference vs flat Level 1.5 is the attention decomposition.

Causality: local uses a causal + windowed mask; coarse only sees chunks whose
last token < t (chunk c valid iff (c+1)*C <= t). No future leakage.

Chunk pooling is over ROTATED K/V, so the coarse summaries live in the same
rotated space as the queries. Depth is unchanged; compute at long T is
O(T*W + T*T/C) rather than O(T^2), but this prototype keeps the full score
matrices (accuracy-focused) — the question here is retrieval quality, not
speed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, _apply_rope
from .model_inekf_level15 import MapFormerWM_Level15InEKF


class WMHierAttnLayer(WMTransformerLayer):
    """WM transformer layer with two-scale (local + coarse-chunk) attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 chunk_size: int = 64, window: int = 128,
                 use_local: bool = True, use_coarse: bool = True):
        super().__init__(d_model, n_heads, dropout)
        self.chunk_size = chunk_size
        self.window = window
        self.use_local = use_local
        self.use_coarse = use_coarse

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        C, W = self.chunk_size, self.window

        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, H, dh).transpose(1, 2)
        K = self.k_proj(h).view(B, T, H, dh).transpose(1, 2)
        V = self.v_proj(h).view(B, T, H, dh).transpose(1, 2)
        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)
        scale = math.sqrt(dh)

        idx = torch.arange(T, device=x.device)

        # --- LOCAL (fine): causal window of size W ---
        if self.use_local:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / scale       # (B,H,T,T)
            local_mask = causal_mask | (idx[:, None] - idx[None, :] >= W)  # True = masked
            scores_local = scores.masked_fill(local_mask[None, None], float('-inf'))
            attn_local = F.softmax(scores_local, dim=-1)
            attn_local = self.dropout(attn_local)
            local_out = torch.matmul(attn_local, V)                     # (B,H,T,dh)
        else:
            local_out = torch.zeros(B, H, T, dh, device=x.device, dtype=Q.dtype)

        # --- COARSE (long-range): attend over chunk-pooled summaries ---
        nC = T // C
        if nC > 0 and self.use_coarse:
            Tc = nC * C
            Kc = K[:, :, :Tc].view(B, H, nC, C, dh).mean(dim=3)     # (B,H,nC,dh)
            Vc = V[:, :, :Tc].view(B, H, nC, C, dh).mean(dim=3)
            cscores = torch.matmul(Q, Kc.transpose(-1, -2)) / scale  # (B,H,T,nC)
            chunk_end = (torch.arange(nC, device=x.device) + 1) * C  # last idx+1 of chunk
            cmask = chunk_end[None, :] > idx[:, None]                # True = chunk not fully past
            cscores = cscores.masked_fill(cmask[None, None], float('-inf'))
            cattn = F.softmax(cscores, dim=-1)
            cattn = torch.nan_to_num(cattn)                          # early rows: no valid chunk -> 0
            cattn = self.dropout(cattn)
            coarse_out = torch.matmul(cattn, Vc)                    # (B,H,T,dh)
        else:
            coarse_out = torch.zeros_like(local_out)

        out = (local_out + coarse_out).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)

        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerWM_HierAttn(MapFormerWM_Level15InEKF):
    """Level 1.5 InEKF backbone with two-scale hierarchical attention.

    Identical to Level15 except the attention layer is WMHierAttnLayer, so a
    head-to-head vs Level15 isolates the effect of the attention hierarchy.
    """

    USE_LOCAL = True
    USE_COARSE = True

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 chunk_size=64, window=128):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMHierAttnLayer(d_model, n_heads, dropout, chunk_size, window,
                            use_local=self.USE_LOCAL, use_coarse=self.USE_COARSE)
            for _ in range(n_layers)
        ])


class MapFormerWM_HierAttn_CoarseOnly(MapFormerWM_HierAttn):
    """Ablation: coarse chunk-pooled attention only (no local window).

    Isolates whether POOLING is what wins the aggregation task.
    """
    USE_LOCAL = False
    USE_COARSE = True


class MapFormerWM_HierAttn_LocalOnly(MapFormerWM_HierAttn):
    """Ablation: local causal window only (no coarse pooling).

    Control for CoarseOnly: a windowed-but-unpooled attention.
    """
    USE_LOCAL = True
    USE_COARSE = False
