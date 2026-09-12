"""The capacity control for PAIRORIGIN: the same pathway, driven by a CONSTANT.

`PAIRORIGIN_RESULTS.md`: giving MapEM per-pair origins is worth +0.280 (7/8) on varying-k
recency. But it also adds 2,048 parameters (+0.92%) in the position pathway, and this line has
been caught before by exactly that confound -- `AlignLock` vs `AlignFree` differed in the
optimiser treatment of a parameter as well as in the freedom it granted, and `MagOnly` was built
to separate them (audit finding 8, rule 31).

`EMPairConst` is that control. Identical architecture and parameter count to `EMPair_r4`, with
one change: the origin projections read a LEARNED CONSTANT instead of the token embedding.

    EMPair       q^p_t = p0 + W^q_out W^q_in x_t      -> origins vary per token  (per-pair kernel)
    EMPairConst  q^p_t = p0 + W^q_out W^q_in c        -> origins are the same for every token
                                                        (ONE kernel, as in single-p0 EM)

So the pathway is alive and trains, the shapes and the optimiser treatment match, and the only
thing removed is the content dependence. `W_out` is zero-initialised, so at step 0 this is
`VanillaEM_P0_r4` exactly, as `EMPair_r4` is.

If `EMPair - EMPairConst` is detectable, per-pair freedom is doing the work. If instead
`EMPairConst` recovers the +0.280, the parameters were, and the kernel-sharing reading of
PAIRORIGIN is withdrawn.
"""
import torch
import torch.nn as nn

from mapformer.model_em_pairorigin import _PairOrigin


class MapFormerEM_PairConst_r4(_PairOrigin):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size,
                         bottleneck_r, origin_r=4)
        # one learned input, shared by every token: same parameter count as an embedding row,
        # initialised the same way, so the pathway trains on the same scale as EMPair's.
        self.origin_const = nn.Parameter(torch.randn(d_model) * 0.02)

    def _origins(self, x):
        B, L, _ = x.shape
        H, dh = self.n_heads, self.d_head
        c = self.origin_const[None, None].expand(B, L, -1)      # content dependence removed
        p0 = self.p0_pos.unsqueeze(0).unsqueeze(2)
        dq = self.q_origin_out(self.q_origin_in(c)).view(B, L, H, dh).transpose(1, 2)
        dk = self.k_origin_out(self.k_origin_in(c)).view(B, L, H, dh).transpose(1, 2)
        return p0 + dq, p0 + dk
