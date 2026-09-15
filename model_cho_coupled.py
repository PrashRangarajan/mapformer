"""Position coupling in Cho et al.'s own architecture (arXiv:2405.20671, Appendix C, Table 1), for reproducing
their 1-layer result (95.65% exact match at 200-digit addition, trained on 1-30 digits).

Decoder-only Transformer; learned absolute position embeddings indexed by coupled position IDs (max_pos 202);
no rotary position; GEGLU feed-forward (hidden 2048); RMSNorm. Their table says "PreNorm and PostNorm"; this is
read as a sandwich block, x = x + PostNorm(sublayer(PreNorm(x))), which is our interpretation, stated. No dropout.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, d_ff, bias=False); self.v = nn.Linear(d, d_ff, bias=False); self.w2 = nn.Linear(d_ff, d, bias=False)
        self.pre1, self.post1, self.pre2, self.post2 = RMSNorm(d), RMSNorm(d), RMSNorm(d), RMSNorm(d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.pre1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, L, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.post1(self.o(a.transpose(1, 2).reshape(B, L, D)))
        y = self.pre2(x)
        return x + self.post2(self.w2(F.gelu(self.w1(y)) * self.v(y)))


class ChoCoupledAPE(nn.Module):
    wants_pos_ids = True

    def __init__(self, vocab_size, d_model=512, n_heads=4, n_layers=1, grid_size=64, max_pos=202, d_ff=2048, **kw):
        super().__init__()
        self.max_pos = max_pos
        self.tok = nn.Embedding(vocab_size, d_model); self.pos = nn.Embedding(max_pos + 2, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model); self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, pos_ids=None):
        B, L = tokens.shape
        if pos_ids is None:
            pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.tok(tokens) + self.pos(pos_ids[:, :L].clamp(max=self.max_pos + 1))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))
