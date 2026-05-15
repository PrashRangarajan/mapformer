"""Control ablation: vanilla MapFormer-WM with post-attn residual dropout removed.

This is the control for `Level15NoDrop`. If `Level15NoDrop` reaches ~0.95 on
lm200 OOD T=512 (vs Level15 0.82, +13pp), the obvious question is whether
the dropout removal alone — without the InEKF correction — is sufficient.

Three possible outcomes:
- VanillaNoDrop ~= Level15NoDrop: dropout was the whole story; InEKF
  redundant once dropout is removed.
- VanillaNoDrop ~= 0.85 (intermediate): dropout helps both; InEKF still
  adds ~10pp on top.
- VanillaNoDrop ~= Vanilla: dropout removal only helps once InEKF is
  wrapping it.

The architecture is identical to MapFormerWM except `WMTransformerLayer` is
replaced with `WMTransformerLayer_NoDrop` (defined in
``model_inekf_level15_nodrop.py``).
"""

from __future__ import annotations

import torch.nn as nn

from .model import MapFormerWM
from .model_inekf_level15_nodrop import WMTransformerLayer_NoDrop


class MapFormerWM_VanillaNoDrop(MapFormerWM):
    """Vanilla MapFormer-WM with post-attention residual dropout removed."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer_NoDrop(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
