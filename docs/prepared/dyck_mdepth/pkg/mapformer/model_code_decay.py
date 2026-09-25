"""ALiBi-style decay envelope on the RoPE-ENCODING arms.

`model_pope_decay.py` already provides the envelope for the two PoPE-encoding
arms. The arms that actually blow up past the training context on code are the
RoPE-encoding ones (CODE_RESULTS_OOD.md: RoPE 4.46, MapWM 4.58 bpc at 2-4x), so
a comparison that only repairs PoPE is a comparison against an unrepaired
baseline. These two close that gap.

Same envelope as the PoPE version so `lambda` means the same thing everywhere:
`scores -= softplus(lambda_h) * distance`, one learnable scalar per head on
ALiBi's geometric spread 2^-1..2^-n_heads. Two distances, matching the two
position mechanisms:

  MapFormerWM_RoPE_Decay   index RoPE, distance |t - s|           (classical ALiBi)
  MapFormerWM_Decay        path RoPE, distance |S_t - S_s| / step (the model's own)

The layer must materialise the score matrix, so it takes the manual attention
path and never SDPA -- an envelope cannot be added to a fused kernel's scores.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WMTransformerLayer, MapFormerWM, _apply_rope
from .model_baseline_rope import MapFormerWM_RoPE


class WMTransformerLayer_Decay(WMTransformerLayer):
    """RoPE-style attention plus an additive, distance-proportional log-decay."""

    def __init__(self, d_model, n_heads, dropout, lam_init=None):
        super().__init__(d_model, n_heads, dropout)
        lam = (torch.full((n_heads,), float(lam_init)) if lam_init is not None
               else 2.0 ** -torch.arange(1, n_heads + 1, dtype=torch.float32))
        self.lam_raw = nn.Parameter(torch.log(torch.expm1(lam)))
        self._dist = None                      # (B, H, T, T) or (1, 1, T, T)

    def forward(self, x, cos_a, sin_a, causal_mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        sh = lambda z: z.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = sh(self.q_proj(h)), sh(self.k_proj(h)), sh(self.v_proj(h))
        Q, K = _apply_rope(Q, cos_a, sin_a), _apply_rope(K, cos_a, sin_a)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head)
        if self._dist is not None:
            lam = F.softplus(self.lam_raw).view(1, -1, 1, 1)
            scores = scores - lam * self._dist
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), V)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, self.d_model))
        x = x + self.dropout(out)
        return x + self.ffn(self.norm2(x))


def _swap(layers, d_model, n_heads, lam_init):
    drop = layers[0].dropout.p if len(layers) else 0.1
    return nn.ModuleList([WMTransformerLayer_Decay(d_model, n_heads, drop, lam_init)
                          for _ in layers])


class MapFormerWM_RoPE_Decay(MapFormerWM_RoPE):
    """Index RoPE + the classical |t - s| ALiBi envelope."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, lam_init=None, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, **kw)
        self.layers = _swap(self.layers, d_model, n_heads, lam_init)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self._rope_cos_sin(L, tokens.device, x.dtype)
        cos_a, sin_a = cos_a.expand(B, -1, -1, -1), sin_a.expand(B, -1, -1, -1)
        t = torch.arange(L, device=tokens.device, dtype=x.dtype)
        dist = (t.view(-1, 1) - t.view(1, -1)).abs()[None, None]
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            layer._dist = dist
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_Decay(MapFormerWM):
    """Path-integrated RoPE + an envelope in the model's OWN accumulated distance.

    The accumulator is averaged over frequency blocks and divided by its own
    per-sequence mean absolute step, so lambda means "decay per token-equivalent"
    whatever rate the model learned, and the initialisation is comparable with
    the index arm's.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, bottleneck_r=2, lam_init=None):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = _swap(self.layers, d_model, n_heads, lam_init)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)
        S = delta.mean(-1).cumsum(1).transpose(1, 2)
        step = delta.mean(-1).abs().mean(dim=(1,), keepdim=True).transpose(1, 2).clamp_min(1e-6)
        dist = (S.unsqueeze(-1) - S.unsqueeze(-2)).abs() / step.unsqueeze(-1)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            layer._dist = dist
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))
