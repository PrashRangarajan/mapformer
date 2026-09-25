"""A per-token sigmoid GATE on a SIGNED increment -- CoPE's selection, MapFormer's direction.

The design question, from the review's sec:borrow. MapFormer's what/where separator
is a linear bottleneck: Delta must reach zero on observation tokens by cancellation
inside W_out W_in, and nothing about that map makes zero a natural output. CoPE's
separator is a sigmoid, for which zero IS the resting state, and deciding which
tokens count is its entire purpose.

That MapFormer wants this is not speculation. On contextual counting an
UNCONSTRAINED increment learns a content gate in 8/8 seeds, and a magnitude-matched
intervention shows the gate is most of the mechanism rather than a component of it
(RECENCY_GATE_ABLATION.md). The model is re-deriving CoPE's gate through a linear
bottleneck. This gives it the gate directly.

But CoPE's gate cannot simply be adopted: it is sigma(.) in (0,1), hence
non-negative, hence a CLOCK and not a map (see the clock/map section). A gate and a
sign answer different questions -- a gate says WHETHER a token moves you, a sign
says WHICH WAY -- and CoPE has only the first. So:

    Delta_gated = sigmoid(W_g x + b) * (W_out W_in x)
                  \_____ whether _____/  \___ which way ___/

One deliberate departure from CoPE. Its gate is computed per query-key PAIR, which
is why CoPE has no theta_t at all and leaves the frame -- and with it the parallel
prefix scan. The gate here is PER TOKEN, which is strictly weaker and keeps the
scan. That is the trade being made, stated rather than hidden.

Initialisation. `gate_bias_init` defaults to 4.0, so sigmoid ~= 0.982 at init and
the model STARTS as its ungated twin. This is not cosmetic: the recency ablation
showed this architecture is acutely sensitive to theta's absolute scale (uniformly
rescaling Delta while counting the same tokens collapsed accuracy from 1.000 to
0.110). A gate initialised at 0.5 would halve the accumulator on step one and the
run would measure that, not the gate. With the bias open, any measured difference is
what the gate LEARNS to close.

Granularity is per-token x per-head, matching the shape Delta already varies over,
at a cost of d_model*n_heads + n_heads parameters (258 at the default config,
+0.13%).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mapformer.model import MapFormerWM, ActionToLieAlgebra


class GatedActionToLie(ActionToLieAlgebra):
    """Signed low-rank increment, multiplied by a learned per-token, per-head gate."""

    def __init__(self, d_model, n_heads, n_blocks, bottleneck_r=2,
                 gate_bias_init: float = 4.0):
        super().__init__(d_model, n_heads, n_blocks, bottleneck_r)
        self.w_gate = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, gate_bias_init)
        self.last_gate = None          # diagnostic; see probe_gate_separation.py

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = super().forward(x)                        # (B,T,H,n_blocks) signed
        g = torch.sigmoid(self.w_gate(x))                 # (B,T,H)
        self.last_gate = g.detach()
        return delta * g.unsqueeze(-1)


def _gated_variant(r, gate_bias_init=4.0, frozen=False):
    class _G(MapFormerWM):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, r)
            self.action_to_lie = GatedActionToLie(
                d_model, n_heads, self.n_blocks, r,
                gate_bias_init=gate_bias_init)
            if frozen:
                # CONTROL: the gate exists (so parameter count and the RNG draw
                # match) but cannot learn. Isolates 'a gate that adapts' from
                # 'an extra module and a constant rescale'.
                for p in self.action_to_lie.w_gate.parameters():
                    p.requires_grad_(False)

        def delta_of(self, tokens):
            return self.action_to_lie(self.token_emb(tokens))

        def gate_of(self, tokens):
            """Per-token, per-head gate values. The separation diagnostic."""
            self.action_to_lie(self.token_emb(tokens))
            return self.action_to_lie.last_gate
    _G.__name__ = f"MapFormerWM_Gated_r{r}" + ("_frozen" if frozen else "")
    _G.__doc__ = ("MapFormer-WM, rank r=%d, signed increment with a learned "
                  "per-token per-head sigmoid gate." % r)
    return _G


MapFormerWM_Gated_r4        = _gated_variant(4)
MapFormerWM_Gated_r2        = _gated_variant(2)
MapFormerWM_Gated_r4_frozen = _gated_variant(4, frozen=True)
