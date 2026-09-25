"""
Bounded-memory attention: the regime the brain actually operates in.

Every prior hierarchy test gave the model UNBOUNDED memory (full attention over
all T past tokens). That is precisely the regime where hierarchy has no job:
compression is hierarchy's reason to exist, and nothing forced compression.
The brain is hierarchical *because* it cannot store everything.

So here we impose a hard read budget M: at every query, the model may attend to
at most M items. The only question is HOW TO SPEND THAT BUDGET.

  BoundedFlat  : spend all M on RECENCY -- attend to the M most recent tokens.
                 Beyond that the past is simply gone.
  BoundedHier  : spend M/2 on recent tokens (fine detail) and M/2 on chunk
                 summaries that span the ENTIRE history (coarse gist). Chunk
                 size adapts, C = ceil(T / (M/2)), so coverage is total and the
                 budget stays fixed as T grows.

At T=4096 with M=128, BoundedFlat sees the last 3% of history and is blind to
the rest; BoundedHier still has a (coarse) view of all of it. On revisit
prediction -- where the needed evidence may lie far in the past -- hierarchy
should now win, because the flat model literally cannot see the answer.

This is the brain-aligned test: recent episodic detail + increasingly
compressed older memory, under a fixed capacity.

Caveat (stated honestly): the coarse summaries are computed here by mean-pooling
the full K/V, so the *implementation* still touches all tokens. It is the READ
budget that is bounded (both variants attend to <= M items), which is what makes
the "how to spend the budget" comparison fair. A streaming system would maintain
the same summaries incrementally as running means.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, _apply_rope
from .model_inekf_level15 import MapFormerWM_Level15InEKF


class WMBoundedAttnLayer(WMTransformerLayer):
    """Attention with a hard read budget M, spent flat (recency) or hierarchically."""

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 mem_budget: int = 128, hierarchical: bool = True):
        super().__init__(d_model, n_heads, dropout)
        self.mem_budget = mem_budget
        self.hierarchical = hierarchical

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        M = self.mem_budget

        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, H, dh).transpose(1, 2)
        K = self.k_proj(h).view(B, T, H, dh).transpose(1, 2)
        V = self.v_proj(h).view(B, T, H, dh).transpose(1, 2)
        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)
        scale = math.sqrt(dh)
        idx = torch.arange(T, device=x.device)

        if self.hierarchical:
            W = max(1, M // 2)          # fine budget
            n_coarse = max(1, M // 2)   # coarse budget
            C = max(1, math.ceil(T / n_coarse))
        else:
            W = M                        # spend everything on recency
            C = None

        # --- FINE: the W most recent tokens ---
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        fine_mask = causal_mask | (idx[:, None] - idx[None, :] >= W)
        scores_f = scores.masked_fill(fine_mask[None, None], float('-inf'))
        attn_f = self.dropout(F.softmax(scores_f, dim=-1))
        out = torch.matmul(attn_f, V)

        # --- COARSE: chunk summaries spanning the whole past (hier only) ---
        if self.hierarchical:
            nC = T // C
            if nC > 0:
                Tc = nC * C
                Kc = K[:, :, :Tc].view(B, H, nC, C, dh).mean(dim=3)
                Vc = V[:, :, :Tc].view(B, H, nC, C, dh).mean(dim=3)
                cs = torch.matmul(Q, Kc.transpose(-1, -2)) / scale
                chunk_end = (torch.arange(nC, device=x.device) + 1) * C
                cmask = chunk_end[None, :] > idx[:, None]      # chunk not fully past
                cs = cs.masked_fill(cmask[None, None], float('-inf'))
                ca = torch.nan_to_num(F.softmax(cs, dim=-1))
                ca = self.dropout(ca)
                out = out + torch.matmul(ca, Vc)

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class _BoundedBase(MapFormerWM_Level15InEKF):
    MEM_BUDGET = 128
    HIERARCHICAL = True

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMBoundedAttnLayer(d_model, n_heads, dropout,
                               mem_budget=self.MEM_BUDGET,
                               hierarchical=self.HIERARCHICAL)
            for _ in range(n_layers)
        ])


class MapFormerWM_BoundedFlat(_BoundedBase):
    """Budget M spent entirely on recency (blind beyond the last M tokens)."""
    HIERARCHICAL = False


class MapFormerWM_BoundedHier(_BoundedBase):
    """Budget M split: M/2 recent detail + M/2 summaries spanning all history."""
    HIERARCHICAL = True
