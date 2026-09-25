"""MapFormer with the PAPER'S per-head bottleneck (paper-faithful r), 2026-09-24.

Our `ActionToLieAlgebra` shares ONE r-dimensional latent across heads:
    Delta = W_out W_in x,  W_in: d -> r,  W_out: r -> nh*nb.
The paper's is per head (main text ~l.276-294: W_in in R^{d x r}, W_out in R^{r x nb} per
head; App. A.7 l.1517: W_in in R^{d x nh x r}):
    Delta_h = W_out^h W_in^h x   for each head h.
At nh=2, r=2 that is 4 latent dimensions with a BLOCK-DIAGONAL W_out, 640 params, between
our r=2 (2 dims, 384) and our r=4 (4 dims, 768). See RANK_PERHEAD_PREREG.md.

Initialisation: the base model is built exactly as our r=2 (`Vanilla`) at the same seed,
and the per-head module is created AFTER it, so every other weight (embeddings, attention,
FFN, readout, omega) is identical to Vanilla's at that seed; only the bottleneck differs.
Per-head W_out uses nn.Linear's default (bound 1/sqrt(r)), the same scale as our r=2's
W_out; W_in the same scale as ours (fan-in d).
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM


class ActionToLieAlgebraPerHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_blocks: int, bottleneck_r: int = 2):
        super().__init__()
        self.n_heads, self.n_blocks, self.r = n_heads, n_blocks, bottleneck_r
        self.w_in = nn.Linear(d_model, n_heads * bottleneck_r, bias=False)   # all heads' W_in^h
        self.w_out = nn.ModuleList(nn.Linear(bottleneck_r, n_blocks, bias=False)
                                   for _ in range(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.w_in(x).view(B, T, self.n_heads, self.r)
        return torch.stack([self.w_out[i](h[:, :, i]) for i in range(self.n_heads)], dim=2)


class MapFormerWM_PerHead(MapFormerWM):
    PER_HEAD_R = 2

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        # replaced AFTER the base is built, so the base's RNG draws match Vanilla's exactly
        self.action_to_lie = ActionToLieAlgebraPerHead(d_model, n_heads, self.n_blocks,
                                                       self.PER_HEAD_R)


class MapFormerWM_r4MatchedInit(MapFormerWM):
    """Our SHARED r=4, built from our r=2's base at the same seed (RANK_MI_PREREG.md).

    The stored r=4 (`Vanilla_r4`) draws its wider W_in inside the base constructor, which
    shifts the random draws of EVERY later weight, so at seed s it shares no initial weights
    with our r=2. Here the base is built exactly as `Vanilla` (r=2) and the r=4 bottleneck is
    created afterwards: every non-bottleneck weight equals our r=2's (and the per-head r=2's)
    at that seed; only the bottleneck differs. W_out keeps nn.Linear's default (bound
    1/sqrt(4)), as in the stored r=4.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        from mapformer.model import ActionToLieAlgebra
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, 4)


# ------------------------------------------------------------------ rank-separation arms
# RANK_SEP_PREREG.md. Both are built, like Vanilla_r2ph and Vanilla_r4mi, from our r=2's base
# at the same seed, so every non-bottleneck weight is shared across all five rank arms.

from mapformer.model import ActionToLieAlgebra as _Lie


class ActionToLieAlgebraBlockDiag(_Lie):
    """A shared-latent bottleneck whose W_out is masked BLOCK-DIAGONAL: head h reads only latent
    dims [h*r, (h+1)*r). Built with exactly the draws of the unmasked module, then masked, so its
    W_in and its diagonal blocks equal C's (Vanilla_r4mi) at the same seed. The mask is applied in
    forward; masked entries get zero gradient and stay exactly zero under AdamW."""

    def __init__(self, d_model, n_heads, n_blocks, per_head_r):
        super().__init__(d_model, n_heads, n_blocks, n_heads * per_head_r)
        m = torch.zeros(n_heads * n_blocks, n_heads * per_head_r)
        for h in range(n_heads):
            m[h * n_blocks:(h + 1) * n_blocks, h * per_head_r:(h + 1) * per_head_r] = 1.0
        self.register_buffer("mask", m)
        with torch.no_grad():
            self.w_out.weight.mul_(m)

    def forward(self, x):
        B, T, _ = x.shape
        delta = torch.nn.functional.linear(self.w_in(x), self.w_out.weight * self.mask)
        return delta.view(B, T, self.n_heads, self.n_blocks)


class MapFormerWM_r4MIBlockDiag(MapFormerWM):
    """C_bd: C (shared r=4 from r=2's base) with cross-head W_out blocks zeroed and held at zero.
    Equivalently a per-head r=2 whose W_in is C's and whose W_out starts at C's scale (bound 0.5)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        self.action_to_lie = ActionToLieAlgebraBlockDiag(d_model, n_heads, self.n_blocks, 2)


class MapFormerWM_PerHead4(MapFormerWM_PerHead):
    """D: per-head r=4 (8 latent dims, 4 per head, block-diagonal), from r=2's base."""
    PER_HEAD_R = 4
