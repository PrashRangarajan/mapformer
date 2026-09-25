"""NoPE baseline: NO positional encoding at all.

The null hypothesis for every positional-encoding claim. A causal decoder can
recover position WITHOUT any explicit encoding, because the causal mask already
breaks permutation symmetry: token t attends over t predecessors, token t+1 over
t+1, so "how many tokens are visible" is itself a positional signal the model can
learn to count (Kazemnejad et al., NeurIPS 2023, arXiv:2305.19466 -- who found
NoPE often LENGTH-GENERALISES BETTER than RoPE/ALiBi on reasoning tasks).

Why this arm matters here. Our index arms (MapFormerWM_RoPE, PoPE-Flat) do not
merely lack a useful position -- they actively rotate q/k by an ORDINAL angle,
which is the wrong signal for a spatial map. MiniWorld showed that a
confidently-wrong position code can be WORSE than a degenerate one (the 24-bin
allocentric recode scored below raw for exactly this reason). So index-RoPE may be
an unnecessarily weak control, and part of a measured "position effect" could be
the handicap rather than the benefit. NoPE settles that: it is the same network
with the rotation removed.

Implementation: identical to MapFormerWM_RoPE except the rotation is the IDENTITY
(cos = 1, sin = 0). Same layers, same code path, and the SAME PARAMETER COUNT
(RoPE's inv_freq is a buffer, not a parameter), so this is a true ablation of the
positional signal and nothing else.
"""
import torch
import torch.nn as nn

from mapformer.model_baseline_rope import MapFormerWM_RoPE


class MapFormerWM_NoPE(MapFormerWM_RoPE):
    """Same architecture as the RoPE index baseline, with NO position rotation."""

    def _rope_cos_sin(self, L, device, dtype):
        """Identity rotation: angle 0 everywhere -> cos = 1, sin = 0.

        Shape matches the parent's (1, n_heads, L, n_blocks) so the layer code is
        untouched; applying a zero-angle rotation leaves q and k unchanged.
        """
        cos = torch.ones(1, self.n_heads, L, self.n_blocks, device=device, dtype=dtype)
        sin = torch.zeros(1, self.n_heads, L, self.n_blocks, device=device, dtype=dtype)
        return cos, sin
