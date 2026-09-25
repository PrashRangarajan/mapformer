"""
Recursive two-level hierarchy: the coarse level consumes the fine level's
PROCESSED output, and its result is added back by residual.

Third family in the taxonomy, after:
  - pool-and-read (Kalman cascade, HierAttn, BoundedHier) -- reads from a mean
    of raw K/V, which on exact recall destroys the answer. Failed.
  - select-and-read (RouteAttn) -- pooling routes, full-resolution read.

Here, following Hourglass / Megabyte / HAN, the chunk representation is NOT a
mean of raw keys and values. It is a LEARNED attention-weighted readout over
the fine level's already-processed token representations, and the coarse
level's output is broadcast back down and added by RESIDUAL, so the fine
stream is never replaced -- only corrected.

  fine level   : local-window attention over tokens          -> h_fine
  chunk readout: learned softmax-weighted sum over each chunk -> c_rep
  coarse level : causal self-attention over chunk reps        -> c_out
  broadcast    : token in chunk c receives c_out[c-1]         (strictly past)
  combine      : h = h_fine + proj(c_out_broadcast)           (residual)

The scientific question this isolates: a mean provably collapses per-token
identity, but a LEARNED readout need not -- 64 tokens of ~log2(21) bits is
~280 bits, which a 128-d vector can carry in principle. So this asks whether a
learned, abstractive summary can encode retrievable content where an average
cannot.

Honest caveat: the fine level is deliberately LOCAL (otherwise the coarse level
has no job), so distant tokens are reachable ONLY through the coarse path. The
residual preserves each token's own representation, not a lossless route to
distant tokens -- that is what separates this from RouteAttn, and it is the
point of the comparison.

Depth confound: this is ~two attention blocks, so the fair control is a
2-layer flat model trained in the same batch, not the 1-layer default.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer
from .model_hier_attn import WMHierAttnLayer
from .model_inekf_level15 import MapFormerWM_Level15InEKF


class ChunkReadout(nn.Module):
    """Learned attention-weighted aggregation over a chunk (HAN-style, not a mean)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, h: torch.Tensor, C: int) -> torch.Tensor:
        B, T, d = h.shape
        nC = T // C
        hc = h[:, :nC * C].reshape(B, nC, C, d)
        s = torch.matmul(torch.tanh(self.proj(hc)), self.query) / math.sqrt(d)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        return (hc * w).sum(dim=2)


class CoarseBlock(nn.Module):
    """Causal self-attention + FFN over chunk representations."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        B, N, d = c.shape
        h = self.norm1(c)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        cm = torch.triu(torch.ones(N, N, device=c.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(cm[None, None], float('-inf'))
        a = self.drop(F.softmax(s, dim=-1))
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        c = c + self.drop(self.o(o))
        return c + self.ffn(self.norm2(c))


class MapFormerWM_Recursive(MapFormerWM_Level15InEKF):
    """Local fine level + learned chunk readout + coarse level + residual back."""

    CHUNK = 64
    WINDOW = 128

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMHierAttnLayer(d_model, n_heads, dropout, self.CHUNK, self.WINDOW,
                            use_local=True, use_coarse=False)
        ])
        self.readout = ChunkReadout(d_model)
        self.coarse = CoarseBlock(d_model, n_heads, dropout)
        self.coarse_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cum = torch.cumsum(delta, dim=1)
        theta_path = cum * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        theta_hat, Pi, K, R = self.inekf(theta_path, x)
        tf = theta_hat.transpose(1, 2)
        cos_a, sin_a = torch.cos(tf), torch.sin(tf)
        causal = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool),
                            diagonal=1)

        h = self.layers[0](x, cos_a, sin_a, causal)

        C = self.CHUNK
        nC = L // C
        if nC > 0:
            creps = self.readout(h, C)
            cout = self.coarse(creps)
            shifted = torch.cat([torch.zeros_like(cout[:, :1]), cout[:, :-1]], dim=1)
            bcast = shifted.unsqueeze(2).expand(-1, -1, C, -1).reshape(B, nC * C, -1)
            rem = L - nC * C
            if rem > 0:
                tail = cout[:, -1:].unsqueeze(2).expand(-1, -1, rem, -1).reshape(B, rem, -1)
                bcast = torch.cat([bcast, tail], dim=1)
            h = h + self.coarse_proj(bcast)

        h = self.out_norm(h)
        return self.out_proj(h)
