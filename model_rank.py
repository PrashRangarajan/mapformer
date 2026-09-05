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

WHAT THE PAPER ACTUALLY ARGUES, checked rather than assumed. App. A.7: "the
internal projection W_in in R^{d x r} (r << d) maps the high-dimensional input X to
a low-dimensional Delta_in (for instance, in a 2D environment, Delta_in could be
the 2D movement vector Delta_tk, where r = 2)". App. A.1: "a in R^r (e.g. in 2D,
r = 2 and 'move right')".

So r is meant to be the DIMENSIONALITY OF THE ACTION SPACE, and the bottleneck is
an INDUCTIVE BIAS rather than a capacity budget -- it forces the rotation to depend
on exactly a 2-vector, which is what a 2D displacement is. That is not arbitrary,
and it checks out: the torus's four actions are +/-x and +/-y, so two dimensions
span them exactly, and observations need Delta = 0, which lies in any subspace.
**r = 2 is sufficient by construction on this task.**

Consequently the paper predicts these arms are FLAT, and flat is the informative
outcome: it would confirm the inductive-bias account AND rule capacity out as the
explanation for Selective RoPE's ~8k-parameter torus win, leaving optimisation
(a full-rank map is better conditioned than a rank-2 one) as the live candidate.

The separate claim that r=1 collapses to 0.66 in 2D has no experiment behind it
anywhere in this repository -- see mapformer_math.tex. Note that r=1 would be a
genuine test of the bias, since one dimension cannot span two independent axes;
these arms only probe the over-provisioned side.
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


MapFormerWM_r3 = _rank(3)   # = D at D=3, for the D x r threshold test
MapFormerWM_r4 = _rank(4)
MapFormerWM_r5 = _rank(5)   # = D at D=5, and D+2 at D=3
MapFormerWM_r7 = _rank(7)   # = D+2 at D=5
MapFormerWM_r8 = _rank(8)
MapFormerWM_r16 = _rank(16)
MapFormerWM_r32 = _rank(32)


# --- EM counterparts, for the Fig. 4 C4 check --------------------------------
# The paper's Fig. 4 reports value-embedding norms much larger for observations
# than actions. On MapWM we measure 0.57 -- inverted. Sec 5.4's framing is
# explicitly about EM's factorisation into "two separate pools of neurons ...
# specialized for either position or observation", which MapWM's additive
# attention does not have, so the claim may simply be scoped to EM.
from mapformer.model import MapFormerEM


def _rank_em(r):
    class _E(MapFormerEM):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, r)
    _E.__name__ = f"MapFormerEM_r{r}"
    _E.__doc__ = f"MapFormer-EM with bottleneck rank r={r}."
    return _E


MapFormerEM_r4 = _rank_em(4)
MapFormerEM_r8 = _rank_em(8)


# --- the loop at a wider bottleneck -------------------------------------------
# LOOP_HEADROOM showed the loop's contribution on Match-Query is mostly to the
# FLOOR: 8/8 seeds >= 0.77 against the 1-layer arm's 0.11-0.80. RANK_SWEEP's
# mechanism is that r=2 learns a SKEWED basis, which is exactly the shape of a
# bimodal seed distribution. So r=4 and the loop may be removing the same failure
# mode by different routes -- untestable without an arm that has both.
from mapformer.model_looped import MapFormerWM_Looped


class MapFormerWM_Looped_r4(MapFormerWM_Looped):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, n_loops=None, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, 4, n_loops)
