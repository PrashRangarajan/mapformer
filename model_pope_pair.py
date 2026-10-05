"""MapPoPE with PAIRWISE frequencies (MAPPOPE_PAIR_PREREG.md): PoPE's score with MapWM's frequency count.

Our MapPoPE (`model_pope.MapFormerWM_PoPE`) differs from MapWM in TWO ways: (1) the score rule -- PoPE's
a_ts = sum_c softplus(q)_c softplus(k)_c cos(theta_t,c - theta_s,c + delta_c) instead of rotating content Q/K; and
(2) the frequency count -- one angle per ELEMENT (n_blocks = d_head = 64 per head, `_widen_to_d`) instead of one per
PAIR (n_blocks = d_head / 2 = 32). This class keeps (1) and undoes (2): MapWM's path machinery (n_blocks = 32, the same
omega range 2pi .. 2pi/grid, same rank), each angle shared by two adjacent elements (repeat_interleave), and PoPE's
layer unchanged (per-element softplus magnitudes and per-element delta_c, clamp [-2pi, 0], zero init). PoPE has no
pairing in the score, so which two elements share an angle is a permutation of learned Q/K rows: immaterial.
"""
import torch

from mapformer.model import MapFormerWM
from mapformer.model_pope import _swap


class MapFormerWM_PoPEPair(MapFormerWM):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        assert self.n_blocks * 2 == self.d_head, (self.n_blocks, self.d_head)
        self.layers = _swap(self.layers, d_model, n_heads)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.action_to_lie(x))           # (B, H, T, d_head / 2)
        cos_a, sin_a = cos_a.repeat_interleave(2, dim=-1), sin_a.repeat_interleave(2, dim=-1)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_PoPEPair_r4(MapFormerWM_PoPEPair):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r=4)
        assert self.action_to_lie.w_in.out_features == 4
