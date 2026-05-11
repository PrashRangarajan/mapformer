"""Level15GSF + NoDrop: K=8 Gaussian Sum Filter with post-attn dropout removed.

Tests whether the two independent fixes — multi-modal Bayes and the dropout
removal — compose. If they do, this should land near TEMFaithful's 0.969 on
lm200 OOD T=512.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_inekf_gsf import MapFormerWM_Level15GSF
from .model_inekf_level15_nodrop import WMTransformerLayer_NoDrop


class MapFormerWM_Level15GSF_NoDrop(MapFormerWM_Level15GSF):
    """Identical to MapFormerWM_Level15GSF except the transformer layers are
    `WMTransformerLayer_NoDrop` (post-attention residual dropout removed)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, n_modes: int = 8):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r, n_modes=n_modes)
        # Replace layers with NoDrop variant; everything else stays.
        self.layers = nn.ModuleList([
            WMTransformerLayer_NoDrop(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
