"""MapPoPE-Pair at rank 4 with MATCHED initialisation (SCORE_RANK_PREREG.md).

The four arms of SCORE_RANK form a 2 x 2 (score rule x per-head rank) in which every shared component starts from the
same draws at a given seed:

    Vanilla            base r2 (MapFormerWM)                         -- MapWM score, rank 2
    MapPoPE-Pair       base r2, then PoPE layers drawn                -- PoPE score,  rank 2  (model_pope_pair, unchanged)
    Vanilla_r4mi       base r2, then an r4 bottleneck drawn           -- MapWM score, rank 4  (model_rank_perhead, unchanged)
    MapPoPE-Pair_r4mi  base r2, r4 bottleneck drawn as Vanilla_r4mi,  -- PoPE score,  rank 4  (this file)
                       RNG rewound to the post-base state, PoPE layers drawn as MapPoPE-Pair

So, at the same seed: the embeddings, omega, readout and (MapWM arms) attention layers equal Vanilla's; the rank-4
bottleneck equals Vanilla_r4mi's; the PoPE layers equal MapPoPE-Pair's. Within each rank only the score rule differs;
within each score rule only the bottleneck differs. Checked in docs/audits/2026-10-05/score_rank_init_check.py.

The existing `MapPoPE-Pair_r4` (model_pope_pair) builds its r4 bottleneck inside the base constructor, which shifts
every later draw, so it shares no initial weights with the other three; it is not used here.
"""
import torch

from mapformer.model import MapFormerWM, ActionToLieAlgebra
from mapformer.model_pope import _swap
from mapformer.model_pope_pair import MapFormerWM_PoPEPair


class MapFormerWM_PoPEPair_r4mi(MapFormerWM_PoPEPair):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        # MapFormerWM's constructor only (not MapFormerWM_PoPEPair's, which would swap the layers before the r4 draw)
        MapFormerWM.__init__(self, vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        assert self.n_blocks * 2 == self.d_head, (self.n_blocks, self.d_head)
        state = torch.get_rng_state()
        lie4 = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, 4)      # = Vanilla_r4mi's draws
        torch.set_rng_state(state)
        self.layers = _swap(self.layers, d_model, n_heads)                 # = MapPoPE-Pair's draws
        self.action_to_lie = lie4
        assert self.action_to_lie.w_in.out_features == 4
