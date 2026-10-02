"""Objects as fixed random codes, for the new-object transfer test (NEWOBJ_PREREG.md).

`use_object_codes(model, n_special, pool_size)` replaces any arm's `token_emb` and `out_proj` (every arm used
here calls exactly those two) so that:
  - the 5 special tokens (4 actions, blank) keep a learned embedding and a learned readout;
  - object o has a FIXED random code c_o in R^64 (one codebook for every run, generator seed 12345), embedded
    as A c_o with A a learned linear "sensory encoder", and read out with logit (B h) . c_o, B learned.
Train-pool and test-pool codes come from the same distribution; the model never sees a test code in training,
so it can name a test object only by copying its code from context. A learned linear encoder (rather than raw
frozen codes) lets the model map ALL codes, seen or not, into a subspace its step map ignores: MapFormer's
step is computed from every token's embedding, and raw random codes would span the whole space.
`readout.active` ("train" / "test") masks the other pool's logits to -inf, so no unseen object is ever a
training distractor.

`MapFormerEM_PosOnly`: MapEM with the content term removed -- scores = A_P only (q, k from rotated q0/k0),
content enters only through V. The strict TEM-style separation reference.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import MapFormerEM, EMTransformerLayer
from mapformer.model_rank import _rank_em

D_CODE = 64


def codebook(n, seed=12345):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D_CODE, generator=g)


class CodeEmbedding(nn.Module):
    def __init__(self, d_model, n_special, pool_size):
        super().__init__()
        self.n_special = n_special
        self.special = nn.Embedding(n_special, d_model)
        self.encoder = nn.Linear(D_CODE, d_model, bias=False)
        self.register_buffer("codes", codebook(2 * pool_size))

    def forward(self, tokens):
        is_obj = tokens >= self.n_special
        sp = self.special(tokens.clamp(max=self.n_special - 1))
        ob = self.encoder(self.codes[(tokens - self.n_special).clamp(min=0)])
        return torch.where(is_obj[..., None], ob, sp)


class CodeReadout(nn.Module):
    def __init__(self, d_model, n_special, pool_size):
        super().__init__()
        self.n_special, self.P = n_special, pool_size
        self.special = nn.Linear(d_model, n_special)
        self.to_code = nn.Linear(d_model, D_CODE)
        self.active = "train"

    def forward(self, h, codes):
        obj = self.to_code(h) @ codes.T / math.sqrt(D_CODE)            # (..., 2P)
        mask = torch.full((2 * self.P,), float("-inf"), device=h.device)
        lo = 0 if self.active == "train" else self.P
        mask[lo:lo + self.P] = 0.0
        return torch.cat([self.special(h), obj + mask], dim=-1)


def use_object_codes(model, n_special, pool_size):
    d = model.token_emb.embedding_dim
    model.token_emb = CodeEmbedding(d, n_special, pool_size)
    ro = CodeReadout(d, n_special, pool_size)
    emb = model.token_emb

    class _Out(nn.Module):
        def __init__(self):
            super().__init__(); self.ro = ro

        def forward(self, h):
            return self.ro(h, emb.codes)
    model.out_proj = _Out()
    return model


def set_pool(model, pool):
    model.out_proj.ro.active = pool


class EMPosOnlyLayer(EMTransformerLayer):
    def forward(self, x, q_pos, k_pos, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q_pos, k_pos.transpose(-1, -2)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.dropout(self.o_proj(out))
        return x + self.ffn(self.norm2(x))


class MapFormerEM_PosOnly(_rank_em(4)):
    """r=4 MapEM whose attention scores are A_P alone. q0/k0 start at 1.0 scale (MapEM's 0.02 would leave
    the position-only softmax uniform at init)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size)
        self.layers = nn.ModuleList([EMPosOnlyLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        with torch.no_grad():
            self.q0_pos.normal_(0, 1.0); self.k0_pos.normal_(0, 1.0)
