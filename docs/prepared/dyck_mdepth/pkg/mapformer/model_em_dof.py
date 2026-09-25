"""MapFormer-EM variants that isolate PHASE DEGREES OF FREEDOM in the position kernel.

The observation to explain (RECENCY_EM_RESULTS.md): single `p_0` beats separate
`q_0`/`k_0` on three map tasks (+0.089 / +0.167 / +0.358) and LOSES on the one
clock task (-0.237). N5 killed the coherence explanation -- coherence helps on
BOTH tasks with the same sign (N5_RESULTS.md), so it cannot produce an inversion.

Third candidate, and the one this module tests: the two forms differ in how many
PHASES they can move.

    kappa(dS) = sum_i a_i cos(omega_i dS + phi_i),   phi_i = angle(q_0i, k_0i)

    single p_0        phi_i == 0 forever          -> 0 phase DOF, rho == 1
    separate q_0/k_0  phi_i free                  -> n_b phase DOF, rho drifts

A map task wants a matched filter, which `phi = 0` already is, so phase freedom is
worth nothing. A clock task must RESHAPE the kernel, so it is worth a lot. That
predicts the inversion without appealing to the value of rho.

The confound N5 fell into
-------------------------
N5 varied coherence and freedom TOGETHER (its frozen arms had neither). Comparing
single-`p_0` against a randomly-initialised separate form does the same thing: the
init coherences differ (1 vs ~0) AND the phase freedom differs. So neither
comparison can attribute anything.

The discriminator here holds init coherence FIXED at rho = 1 and varies ONLY phase
freedom:

    AlignFree   k_0 initialised EQUAL to q_0, both free   rho(0) = 1, n_b phase DOF
    AlignLock   k_0 = s_i * q_0i, s_i free                rho    = 1 always, 0 phase DOF

`AlignFree - AlignLock` is the phase-freedom effect at matched initial kernel.
If the DOF account is right it is large on the clock task and ~0 on the map task.
If instead the initial coherence was doing the work, the two tie everywhere.

`AlignLock` keeps per-block MAGNITUDE freedom (`s_i`), so it is not a freeze -- the
kernel's spectral weights `a_i` can still be learned, only the phases cannot. That
matters because N5's freeze removed both and was catastrophic on recency.

Parameter note: AlignLock carries `n_heads * n_blocks` scale parameters in place of
a second full vector, so it has 64 FEWER parameters than AlignFree at d=128/h=2 --
0.03% of 222k. Stated rather than hidden; it is far below anything measurable here,
and the alternative (a free 2-vector whose angle is discarded) would create dead
parameters, which is worse.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerEM


class _AlignFree(MapFormerEM):
    """Separate q_0/k_0, initialised EQUAL (rho = 1 at step 0), both free."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, r=4, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, r)
        with torch.no_grad():
            self.k0_pos.data.copy_(self.q0_pos.data)
        assert torch.equal(self.q0_pos.data, self.k0_pos.data), "not aligned at init"
        assert self.q0_pos.requires_grad and self.k0_pos.requires_grad, "must stay free"
        assert self.action_to_lie.w_in.out_features == r, "rank lost in construction"


class _AlignLock(MapFormerEM):
    """k_0i = s_i * q_0i: phases pinned at 0 forever, per-block magnitudes free.

    `k0_pos` becomes a PROPERTY, so the base class's forward (which reads
    `self.k0_pos`) picks up the computed tensor. The base Parameter is removed
    from `_parameters` first -- a class-level property is a data descriptor and
    wins over `nn.Module.__getattr__`, but leaving the Parameter registered would
    keep it in `state_dict()` and in the optimizer.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, r=4, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, r)
        del self._parameters["k0_pos"]
        self.k0_scale = nn.Parameter(torch.ones(n_heads, self.n_blocks))
        assert "k0_pos" not in dict(self.named_parameters()), "k0_pos still a parameter"
        assert self.action_to_lie.w_in.out_features == r, "rank lost in construction"

    @property
    def k0_pos(self):
        H, nb = self.k0_scale.shape
        q = self.q0_pos.view(H, nb, 2)
        return (q * self.k0_scale.unsqueeze(-1)).reshape(H, nb * 2)


def _mk(base, name, r=4):
    class _P(base):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, bottleneck_r, r=r)
    _P.__name__ = name
    _P.__doc__ = base.__doc__
    return _P


MapFormerEM_AlignFree_r4 = _mk(_AlignFree, "MapFormerEM_AlignFree_r4")
MapFormerEM_AlignLock_r4 = _mk(_AlignLock, "MapFormerEM_AlignLock_r4")
