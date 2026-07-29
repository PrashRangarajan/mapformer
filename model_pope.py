"""PoPE (Polar Coordinate Position Embeddings, Gopalakrishnan et al. 2025)
plugged into the MapFormer family, plus the MapFormer x PoPE combo and its
hierarchical version.

RoPE (and MapFormer's rotary) entangle content and position: the score is
  Sum_j |q_j||k_j| cos((theta_q,j - theta_k,j) + content_phase)
where the pre-rotation content phase bleeds into the positional term. PoPE
DECOUPLES them -- magnitude = content (softplus), phase = position ONLY:
  Q_cos = softplus(Q) * cos(theta),  Q_sin = softplus(Q) * sin(theta)
  score = Q_cos K_cos^T + Q_sin K_sin^T = Sum_j softplus(q_j)softplus(k_j) cos(theta_q,j - theta_k,j)
so content (magnitude) and position (phase) cannot confound. This is the
principled version of our CoarseIdx/CoarsePI lesson (spatial position interfering
with content matching). Here 'theta' is whatever position the host supplies:
INDEX (plain PoPE) or the MapFormer PATH-INTEGRATION angle (the combo).

All PoPE variants are PARAM-IDENTICAL to their RoPE counterparts (same q/k/v/o
projections, norms, FFN); only the attention score computation differs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, MapFormerWM
from .model_baseline_rope import MapFormerWM_RoPE
from .model_hourglass import MapFormerWM_Hourglass_k2, MapFormerWM_Hourglass_CoarseIdx


def _pope(x, cos_a, sin_a):
    """x (B,H,T,dh); cos_a/sin_a (B,H,T,nb). Returns (x_cos, x_sin): magnitude
    = softplus(content), phase = the supplied position angle (per element)."""
    mag = F.softplus(x)
    cos = cos_a.repeat_interleave(2, dim=-1)     # (B,H,T,dh): each block's angle -> 2 elems
    sin = sin_a.repeat_interleave(2, dim=-1)
    return mag * cos, mag * sin


class WMTransformerLayer_PoPE(WMTransformerLayer):
    """PoPE-decoupled attention WITH the learnable per-frequency phase bias
    delta_c (faithful PoPE, Eq. 6):
        score = sum_c softplus(q_c) softplus(k_c) cos( (theta_q - theta_k) + delta_c )
    delta_c is a content-INDEPENDENT learnable offset per (head, frequency),
    applied to the keys (shifting the key angle by -delta gives +delta in the
    score). It inits to 0 (so at init this equals the un-biased form) and is
    learned -- adding per-band flexibility without re-entangling content.
    Frequencies kept at d/2 (MapFormer's block structure) so the comparison
    across position sources stays controlled; a full d-frequency PoPE would need
    restructuring MapFormer's per-block path integration."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__(d_model, n_heads, dropout)
        n_blocks = (d_model // n_heads) // 2
        self.pope_delta = nn.Parameter(torch.zeros(n_heads, n_blocks))   # learnable delta_c

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        cd = torch.cos(self.pope_delta).view(1, self.n_heads, 1, -1)
        sd = torch.sin(self.pope_delta).view(1, self.n_heads, 1, -1)
        cosK = cos_a * cd + sin_a * sd          # key angle shifted by -delta_c
        sinK = sin_a * cd - cos_a * sd
        Qc, Qs = _pope(Q, cos_a, sin_a)
        Kc, Ks = _pope(K, cosK, sinK)
        scale = math.sqrt(self.d_head)
        scores = (torch.matmul(Qc, Kc.transpose(-1, -2))
                  + torch.matmul(Qs, Ks.transpose(-1, -2))) / scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


def _swap_pope(layers, d_model, n_heads):
    drop = layers[0].dropout.p if len(layers) else 0.1
    return nn.ModuleList([WMTransformerLayer_PoPE(d_model, n_heads, drop) for _ in layers])


class MapFormerWM_PoPE(MapFormerWM):
    """The COMBO: MapFormer path-integration position + PoPE decoupling. Flat."""
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        self.layers = _swap_pope(self.layers, d_model, n_heads)


class MapFormerWM_RoPEIndex_PoPE(MapFormerWM_RoPE):
    """PoPE ALONE: standard index position + PoPE decoupling (the paper's method,
    no path integration). Flat."""
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, base=10000.0, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, base)
        self.layers = _swap_pope(self.layers, d_model, n_heads)


class MapFormerWM_Hourglass_PoPE(MapFormerWM_Hourglass_k2):
    """MapFormer path-integration + PoPE + single-level hourglass hierarchy.
    Coarse position = pooled path angle (inherited from Hourglass_k2)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d, h = self.d_model, self.n_heads
        self.pre_layers = _swap_pope(self.pre_layers, d, h)
        self.coarse_layers = _swap_pope(self.coarse_layers, d, h)
        self.post_layers = _swap_pope(self.post_layers, d, h)


class MapFormerWM_Hourglass_PoPE_CoarseIdx(MapFormerWM_Hourglass_CoarseIdx):
    """BEST-OF-BOTH: PoPE decoupling everywhere (wins OOD length) + ORDINAL index
    coarse position (wins content), on the path-integration fine backbone +
    hierarchy. Combines the length-axis winner (PoPE) with the content-axis
    winner (CoarseIdx's index coarse position). Param-identical to Hourglass_k2."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d, h = self.d_model, self.n_heads
        self.pre_layers = _swap_pope(self.pre_layers, d, h)
        self.coarse_layers = _swap_pope(self.coarse_layers, d, h)
        self.post_layers = _swap_pope(self.post_layers, d, h)
