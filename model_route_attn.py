"""
Identity-preserving hierarchical routing: coarse SELECTS, fine READS at full
resolution. The one hierarchy shape the sweep never tested.

Every hierarchy we built (Kalman cascade, HierAttn, BoundedHier) ends in the
same move: mean-pool a chunk and then READ from the pooled summary. On an
exact-recall task that is fatal -- averaging 64 tokens destroys the very
per-token identity the query is asking for.

The hierarchical-transformer literature splits on exactly this:
  - selection-based (Longformer, BigBird): restrict WHICH tokens are attended,
    but every attended token stays at FULL RESOLUTION -> survives retrieval.
  - pooling-based (Hourglass, Funnel, HAN, Swin): average tokens into coarser
    summaries -> only viable when the task wants gist.

We built the pooling kind and only ever tested it on exact recall. This module
builds the selection kind:

  1. ROUTE (coarse, lossy is fine here): score each past chunk with pooled keys
     Kc = mean(K over chunk); take the top-k chunks per query.
  2. READ (fine, lossless): attend over the union of {local window} and {all
     tokens inside the selected chunks}, at FULL RESOLUTION, reading the
     ORIGINAL V. Nothing is ever read from a pooled summary.

The chunk score is also added as a BIAS to the token-level scores inside its
chunk. Without it the router is non-differentiable and would never learn --
the inert-module trap we hit with the Kalman cascade, where a module that
receives no gradient sits at its initialisation and does nothing.

This prototype keeps the full score matrix (accuracy-focused); the question
here is whether the inductive bias helps, not whether it is fast.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, _apply_rope
from .model_inekf_level15 import MapFormerWM_Level15InEKF


class WMRouteAttnLayer(WMTransformerLayer):
    """Coarse top-k chunk routing + full-resolution read."""

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 chunk_size: int = 64, window: int = 128, top_k: int = 2,
                 route_bias: bool = True):
        super().__init__(d_model, n_heads, dropout)
        self.chunk_size = chunk_size
        self.window = window
        self.top_k = top_k
        self.route_bias = route_bias

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

        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        local_ok = (~causal_mask) & ((idx[:, None] - idx[None, :]) < W)
        allowed = local_ok[None, None].expand(B, H, T, T).clone()

        nC = T // C
        if nC > 0:
            Tc = nC * C
            Kc = K[:, :, :Tc].view(B, H, nC, C, dh).mean(dim=3)
            cs = torch.matmul(Q, Kc.transpose(-1, -2)) / scale        # (B,H,T,nC)
            chunk_end = (torch.arange(nC, device=x.device) + 1) * C
            cmask = chunk_end[None, :] > idx[:, None]                  # not fully past
            cs_masked = cs.masked_fill(cmask[None, None], float('-inf'))

            k = min(self.top_k, nC)
            topi = cs_masked.topk(k, dim=-1).indices                   # (B,H,T,k)
            sel = torch.zeros_like(cs, dtype=torch.bool)
            sel.scatter_(-1, topi, True)
            sel = sel & (~cmask)[None, None]                           # drop -inf picks

            chunk_of = (idx[:Tc] // C)                                 # (Tc,)
            tok_sel = sel[..., chunk_of]                               # (B,H,T,Tc)
            allowed[..., :Tc] |= tok_sel

            if self.route_bias:
                cs_causal = cs.masked_fill(cmask[None, None], 0.0)
                bias = torch.zeros_like(scores)
                bias[..., :Tc] = cs_causal[..., chunk_of]
                scores = scores + bias

        scores = scores.masked_fill(~allowed, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)                                    # ORIGINAL V

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class _RouteBase(MapFormerWM_Level15InEKF):
    TOP_K = 2
    ROUTE_BIAS = True

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 chunk_size=64, window=128):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMRouteAttnLayer(d_model, n_heads, dropout, chunk_size, window,
                             top_k=self.TOP_K, route_bias=self.ROUTE_BIAS)
            for _ in range(n_layers)
        ])


class MapFormerWM_RouteAttn(_RouteBase):
    """Top-2 chunk routing, full-resolution read."""
    TOP_K = 2


class MapFormerWM_RouteAttn_K4(_RouteBase):
    """Wider routing budget (top-4 chunks)."""
    TOP_K = 4


class MapFormerWM_RouteAttn_NoBias(_RouteBase):
    """Ablation: router gets NO gradient (selection only).

    Tests whether the router actually learns to route, or whether any benefit
    comes merely from having a bounded, non-local attention span.
    """
    TOP_K = 2
    ROUTE_BIAS = False
