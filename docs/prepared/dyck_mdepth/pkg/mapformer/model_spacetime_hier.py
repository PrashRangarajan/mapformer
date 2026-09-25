"""
Space-and-time hierarchy that PRESERVES the cognitive-map property at both
levels, by rotating along theta at the coarse level too.

Why the earlier designs broke the map. Measured on a trained model, chunk-
pooling of ROTATED keys destroys 56-69% of the position code's magnitude
(survival 0.31 fine / 0.44 coarse, against a 0.125 random-phase floor): you are
summing vectors with different phases, so they cancel. Worse, the coarse level
then did attention with NO rotation structure at all, so its scores were not a
function of relative displacement. The map property -- score depends on
theta_i - theta_j -- held at the fine level and was simply absent above it.

The fix, on both axes:

  SPACE. MapFormer's omega spectrum is already geometric from fine to coarse,
  so the spatial hierarchy is a FREQUENCY SPLIT, not a new mechanism. The
  coarse level uses the LOW-omega blocks -- the ones that resolve region rather
  than cell, and the ones that survive aggregation best. This is the Stensola
  grid-module reading of the spectrum, made explicit.

  TIME. A coarse node carries a POSITION: theta at its last token, restricted
  to the coarse blocks. Coarse queries and keys are rotated by that coarse
  theta, so coarse scores depend on theta_coarse_c - theta_coarse_c', i.e. on
  relative displacement BETWEEN REGIONS. The coarse level is therefore a
  cognitive map over regions, not a bag of summaries.

  CONTENT. Chunk content is a LEARNED attention-weighted readout of the fine
  level's processed output (as in Hourglass/Megabyte/HAN), never a mean of raw
  rotated keys -- so no destructive interference on the content path either.

  COMBINE. Coarse output is broadcast back one chunk later and added by
  RESIDUAL, so the fine stream is corrected, never replaced.

Known limitation, stated rather than hidden: segmentation is still by token
count. The map-aligned version segments by REGION TRANSITION (a new coarse node
when the agent crosses a coarse-cell boundary), which makes the time hierarchy
derive from the space hierarchy instead of being imposed. That needs dynamic
segmentation; this is the fixed-stride approximation of it.

Higher-order policy (subgoal at the coarse level, action at the fine level) is
the natural next layer for the goal-directed tasks and is NOT implemented here.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_hier_attn import WMHierAttnLayer
from .model_recursive import ChunkReadout
from .model_inekf_level15 import MapFormerWM_Level15InEKF


def _rope_partial(x, cos, sin, nb):
    """Rotate the first 2*nb dims of each head by the given angles; leave the rest."""
    r, keep = x[..., :2 * nb], x[..., 2 * nb:]
    x1, x2 = r[..., 0::2], r[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    rot = torch.stack([o1, o2], dim=-1).reshape_as(r)
    return torch.cat([rot, keep], dim=-1)


class CoarseMapBlock(nn.Module):
    """Causal attention over coarse nodes, with RoPE by COARSE theta."""

    def __init__(self, d_model, n_heads, n_coarse_blocks, dropout):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.nb = min(n_coarse_blocks, self.dh // 2)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, c, th_c):
        B, N, d = c.shape
        h = self.norm1(c)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        ang = th_c[..., :self.nb].permute(0, 2, 1, 3)
        cos, sin = torch.cos(ang), torch.sin(ang)
        q = _rope_partial(q, cos, sin, self.nb)
        k = _rope_partial(k, cos, sin, self.nb)
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        cm = torch.triu(torch.ones(N, N, device=c.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(cm[None, None], float('-inf'))
        a = self.drop(F.softmax(s, dim=-1))
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        c = c + self.drop(self.o(o))
        return c + self.ffn(self.norm2(c))


class MapFormerWM_SpaceTimeHier(MapFormerWM_Level15InEKF):
    """Fine map over cells + coarse map over regions, both rotation-structured."""

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
        n_coarse = max(1, self.n_blocks // 2)
        self.n_coarse = n_coarse
        self.coarse = CoarseMapBlock(d_model, n_heads, n_coarse, dropout)
        self.coarse_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cum = torch.cumsum(delta, dim=1)
        theta_path = cum * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        theta_hat, Pi, K, R = self.inekf(theta_path, x)
        tf = theta_hat.transpose(1, 2)
        causal = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool),
                            diagonal=1)

        h = self.layers[0](x, torch.cos(tf), torch.sin(tf), causal)

        C = self.CHUNK
        nC = L // C
        if nC > 0:
            creps = self.readout(h, C)
            last = torch.arange(nC, device=tokens.device) * C + (C - 1)
            th_c = theta_hat[:, last][..., -self.n_coarse:]
            cout = self.coarse(creps, th_c)
            shifted = torch.cat([torch.zeros_like(cout[:, :1]), cout[:, :-1]], dim=1)
            bcast = shifted.unsqueeze(2).expand(-1, -1, C, -1).reshape(B, nC * C, -1)
            rem = L - nC * C
            if rem > 0:
                tail = cout[:, -1:].unsqueeze(2).expand(-1, -1, rem, -1).reshape(B, rem, -1)
                bcast = torch.cat([bcast, tail], dim=1)
            h = h + self.coarse_proj(bcast)

        h = self.out_norm(h)
        return self.out_proj(h)
