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
