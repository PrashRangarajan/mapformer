"""RoPE with the CANONICAL frequency schedule, as a control.

This repository's RoPE baseline computes

    inv_freq_c = base^(-c / (n_b - 1)),        n_b = d_head / 2

which reaches exactly base^-1 at the last block. Canonical RoPE is

    inv_freq_c = base^(-2c / d_head) = base^(-c / n_b),

which stops slightly short. The two agree to within 1% over the high-frequency
blocks that actually resolve position at the sequence lengths used here, and differ
by up to 25% at the lowest frequencies -- whose wavelengths (47k vs 63k tokens) are
effectively DC over any sequence in this project.

The repo's convention is plausibly deliberate: `PathIntegrator` uses the same
`max(n_blocks - 1, 1)` endpoint handling, so the index and path-integrated arms
share a frequency ladder and differ only in what drives the angle. But the comment
above the line claims the canonical form, so the file contradicts itself, and no
experiment has ever checked whether the choice matters.

This class exists to check it. If the difference is inside the noise floor, the
baseline should simply be switched to canonical and the discussion deleted --
a documented quirk is worse than either fixing it or leaving it alone.

NOTE ON REPRODUCIBILITY: `inv_freq` is a registered buffer, so it lives in the
state dict. Existing RoPE checkpoints therefore keep their own schedule when
loaded, whatever the code later says; switching the default affects future runs
only.
"""
import torch

from mapformer.model_baseline_rope import MapFormerWM_RoPE


class MapFormerWM_RoPE_Canonical(MapFormerWM_RoPE):
    """Identical to the RoPE baseline except for the frequency denominator."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, base=10000.0, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, base)
        c = torch.arange(self.n_blocks, dtype=torch.float32)
        self.inv_freq.copy_(base ** (-c / self.n_blocks))   # canonical
