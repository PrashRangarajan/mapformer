"""Selective RoPE's angle generator, dropped into MapFormer's scaffold.

WHY. Selective RoPE (Movahedi et al., arXiv:2511.17388, ICLR 2026; posted 21 Nov
2025, three days before MapFormer, neither citing the other) occupies the SAME slot
as MapFormer's path integration: it drives the rotation PHASE from a
content-dependent cumulative sum. Its pseudocode (their Fig. 4) is

    omega = conv1d(W_omega @ q)
    omega = temp * cumsum(omega)
    return rope(q, k, sincos(omega))

against MapFormer's  theta = omega * cumsum(W_out W_in x).  Four differences, and
three of them are knobs this project's own ablations say should matter:

    input        query (post-projection, per head)  vs  token embedding
    bottleneck   none                               vs  rank r = 2
    smoothing    causal conv1d over positions       vs  none
    gating       sigmoid phase gate + temperature   vs  none

WHAT THIS FILE TESTS, AND WHAT IT DOES NOT. It swaps the GENERATOR while keeping
MapFormer's placement -- the angle is still computed once from the token embeddings
before the blocks. At 1 layer that is a close approximation to their design, since
q = W_Q LayerNorm(x) is itself a learned linear map of the token, so
"W_omega @ q" and "W_omega @ x" differ only in which linear map is learned. It is
NOT a faithful reimplementation of Selective RoPE: the per-head, per-layer,
query-sourced placement is not reproduced, and at depth > 1 that difference is real.

The three single-knob arms below exist so that any difference can be ATTRIBUTED
rather than just observed.

PARAMETER NOTE, stated because it cannot be avoided: removing the rank bottleneck
is most of the cost. W_omega is d_model x (H*nb) = 128 x 64 = 8,192 against
MapFormer's W_in + W_out = 256 + 128 = 384. These arms are deliberately NOT
parameter-matched to Vanilla; that is what "no bottleneck" means, and the
comparison has to be read with it.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM


class _CausalDepthwiseConv(nn.Module):
    """Depthwise conv over POSITIONS, left-padded so position t sees only <= t."""

    def __init__(self, channels: int, kernel: int = 4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)

    def forward(self, u):                      # u: (B, T, C)
        T = u.shape[1]
        z = torch.nn.functional.pad(u.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(z)[..., :T].transpose(1, 2)


class SelectiveAngle(nn.Module):
    """theta_t = temp * cumsum( gate . conv1d( W_omega x ) )_t  -- their Fig. 4."""

    def __init__(self, d_model, n_heads, n_blocks, kernel=4,
                 use_conv=True, use_gate=True, bottleneck_r=None):
        super().__init__()
        self.n_heads, self.n_blocks = n_heads, n_blocks
        C = n_heads * n_blocks
        if bottleneck_r is None:                       # full rank, as in SRoPE
            self.proj = nn.utils.parametrizations.weight_norm(
                nn.Linear(d_model, C, bias=True))
        else:                                          # MapFormer's low-rank map
            self.proj = nn.Sequential(nn.Linear(d_model, bottleneck_r, bias=False),
                                      nn.Linear(bottleneck_r, C, bias=False))
        self.conv = _CausalDepthwiseConv(C, kernel) if use_conv else None
        self.gate = nn.Linear(d_model, C) if use_gate else None
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x):                              # x: (B, T, d_model)
        B, T, _ = x.shape
        w = self.proj(x)
        if self.conv is not None:
            w = self.conv(w)
        if self.gate is not None:
            w = w * torch.sigmoid(self.gate(x))
        theta = self.log_temp.exp() * torch.cumsum(w, dim=1)
        return theta.view(B, T, self.n_heads, self.n_blocks)


class _SelectiveBase(MapFormerWM):
    """MapFormer, with the angle generator replaced. Everything else identical."""
    USE_CONV, USE_GATE, RANK = True, True, None

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        del self.action_to_lie, self.path_integrator
        self.angle = SelectiveAngle(d_model, n_heads, self.n_blocks,
                                    use_conv=self.USE_CONV, use_gate=self.USE_GATE,
                                    bottleneck_r=self.RANK)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        theta = self.angle(x).transpose(1, 2)          # (B, H, T, nb)
        cos_a, sin_a = torch.cos(theta), torch.sin(theta)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_SRoPEGen(_SelectiveBase):
    """The full Selective-RoPE generator: no bottleneck + conv + gate + temp."""
    USE_CONV, USE_GATE, RANK = True, True, None


class MapFormerWM_NoBottleneck(_SelectiveBase):
    """Single knob: MapFormer's generator with the rank bottleneck REMOVED."""
    USE_CONV, USE_GATE, RANK = False, False, None


class MapFormerWM_ConvAngle(_SelectiveBase):
    """Single knob: MapFormer's low-rank generator PLUS the causal conv."""
    USE_CONV, USE_GATE, RANK = True, False, 2


class MapFormerWM_GateAngle(_SelectiveBase):
    """Single knob: MapFormer's low-rank generator PLUS the sigmoid gate."""
    USE_CONV, USE_GATE, RANK = False, True, 2
