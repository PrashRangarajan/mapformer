"""Position-coupling oracle in Cho et al.'s own form (arXiv:2405.20671, sec. 3): LEARNED ABSOLUTE position
embeddings indexed by coupled position IDs, added to the token embeddings, with no rotary position.
Training draws a random starting ID so every embedding row up to MAX_POS is trained; evaluation starts at 1.

Architecture otherwise identical to the repo's index baselines (MapFormer's WM layer with an identity
rotation), so it differs from Cho et al.'s GEGLU / RMSNorm / d=512 model; that difference is stated, not hidden.
"""
import torch
import torch.nn as nn

from mapformer.environment_addition import MAX_POS
from mapformer.model_baseline_nope import MapFormerWM_NoPE


class MapFormerWM_CoupledAPE(MapFormerWM_NoPE):
    wants_pos_ids = True

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, **kw)
        self.pos_emb = nn.Embedding(MAX_POS + 2, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, tokens, pos_ids=None):
        B, L = tokens.shape
        if pos_ids is None:
            pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos_ids[:, :L].clamp(max=MAX_POS + 1))
        cos_a, sin_a = self._rope_cos_sin(L, tokens.device, x.dtype)
        cos_a, sin_a = cos_a.expand(B, -1, -1, -1), sin_a.expand(B, -1, -1, -1)
        mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, mask)
        return self.out_proj(self.out_norm(x))
