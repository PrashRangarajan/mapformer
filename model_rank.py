"""MapFormer with a wider low-rank bottleneck -- the cheap "between".

WHY. Selective RoPE's two parameter-adding knobs (removing the rank bottleneck,
adding a sigmoid gate) each help on the torus by +0.02 to +0.09 and are
STATISTICALLY INDISTINGUISHABLE FROM EACH OTHER (GateAngle - NoBottleneck is +0.018
and +0.028, both inside their MDEs). Two unrelated ways of spending ~8k parameters
buy the same thing, so the likely explanation is capacity, not either mechanism.

The gate-as-token-suppressor hypothesis was tested directly and FALSIFIED: the gate
is 1.35x larger on action than observation tokens on the torus (0.560 vs 0.416, far
from the ~0 suppression required), and 1.54x on parity -- a LARGER split on the task
where it HURTS. See GATE_PROBE.md.

So if it is capacity, MapFormer already has the knob, and it is far cheaper:

    action_to_lie params = W_in (r x d) + W_out (H*nb x r) = 192r  at d=128, H=2, nb=32

    r = 2   384 params   (the paper's "for instance r = 2")
    r = 4   768          +384
    r = 8   1,536        +1,152
    r = 16  3,072        +2,688

If r=8 at +1,152 captures the +0.086 the gate bought for +8,193, the answer to
"something between MapFormer and Selective RoPE" is that you do not need their
machinery -- you need to widen the bottleneck MapFormer already has, at a seventh
of the cost, with no new mechanism and no conv to pay for.

This also tests the paper's own choice. App. A.7 says only r << d, "for instance
r = 2". The one claim that r is load-bearing (r=1 -> 0.66) has no experiment behind
it anywhere in this repository.
"""
from mapformer.model import MapFormerWM


def _rank(r):
    class _R(MapFormerWM):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, r)
    _R.__name__ = f"MapFormerWM_r{r}"
    _R.__doc__ = f"MapFormer-WM with bottleneck rank r={r}."
    return _R


MapFormerWM_r4 = _rank(4)
MapFormerWM_r8 = _rank(8)
MapFormerWM_r16 = _rank(16)
MapFormerWM_r32 = _rank(32)
