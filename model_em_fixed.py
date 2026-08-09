"""MapFormer-EM with the Hadamard product applied to PROBABILITIES, per the
paper's eq. 13 --- fixing a sign pathology in our original reimplementation.

    paper : (Att(Q,K) o Att(Q_P,K_P)) V  =  (A_X o A_P) V
            where Att(.) INCLUDES the softmax, so A_X, A_P are attention
            matrices: non-negative, rows summing to 1. A_P is a mask in [0,1].

    ours  : softmax(A_X_logits o A_P_logits)
            i.e. the elementwise product of RAW SIGNED logits.

Because logits are signed, the original computes an XNOR rather than an AND:
measured on a trained VanillaEM (layer 0, causal pairs), 35.5% of pairs had
A_X<0 AND A_P<0, and 69.9% of all POSITIVE scores came from such
double-mismatches. Symmetrically, content-match-with-position-mismatch --- which
is exactly cross-instance retrieval --- was driven negative. That is the likely
cause of MapEM-Flat's 0.097 on the compositional task.

Note on renormalisation: the elementwise product of two probability rows does
not sum to 1 (it is ~1/L smaller), so we renormalise. The paper's equation as
written does not show this; without it the attention output is scaled down by
orders of magnitude. Set `renormalise=False` to follow the equation literally.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerEM, EMTransformerLayer


class EMTransformerLayer_Fixed(EMTransformerLayer):
    """Hadamard product on softmaxed attention matrices (paper eq. 13)."""

    renormalise = True

    def forward(self, x, q_pos, k_pos, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        Q_c = self.q_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K_c = self.k_content(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        m = causal_mask.unsqueeze(0).unsqueeze(0)

        # softmax EACH branch first -> both are proper attention matrices
        A_X = F.softmax(
            (torch.matmul(Q_c, K_c.transpose(-1, -2)) / scale).masked_fill(m, float('-inf')), dim=-1)
        A_P = F.softmax(
            (torch.matmul(q_pos, k_pos.transpose(-1, -2)) / scale).masked_fill(m, float('-inf')), dim=-1)

        attn = A_X * A_P                                   # non-negative AND-gate
        if self.renormalise:
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class MapFormerEM_Fixed(MapFormerEM):
    """MapFormer-EM, Hadamard product on probabilities (paper-faithful)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList(
            [EMTransformerLayer_Fixed(d_model, n_heads, dropout) for _ in self.layers])
