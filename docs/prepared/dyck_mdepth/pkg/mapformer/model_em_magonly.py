"""MagOnly: the matched-optimiser control for the phase-freedom result.

AUDIT_2026-09-10.md finding 8: `AlignFree` and `AlignLock` differ in MORE than phase
freedom. AlignLock stores magnitude as a scale `s` initialised at 1.0, which Adam
moves ~0.1% per step in relative terms, against ~5% for AlignFree's raw `k0`
entries (~0.02); weight decay alone explains most of `s`'s drift. So D1
(+0.165 at n=24) could be phase freedom OR parameterisation.

MagOnly removes that confound. `k0` is stored exactly as in AlignFree -- a free
vector `u` of shape [n_heads, d_head], initialised EQUAL to q0 -- but the forward
pass uses only each block's NORM, with the phase pinned to q0's:

    k0_i = |u_i| * q0_i / |q0_i|        -> rho = 1 always, magnitudes free

Consequences, each verified at construction:
  * same parameter count as AlignFree (u replaces k0_pos one-for-one);
  * same init scale and the same Adam / weight-decay treatment;
  * IDENTICAL FUNCTION AT INIT: u = q0 gives k0 = q0, exactly AlignFree's start,
    and the base-class RNG draws are consumed identically, so every other
    parameter matches too.

So `AlignFree - MagOnly` is phase freedom at matched optimiser treatment and
matched initial function. The dead-parameter objection in DOF_PREREG.md does not
apply: the gradient of |u_i| is radial, so both coordinates of each block receive
gradient in general.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerEM

_EPS = 1e-12


class MapFormerEM_MagOnly_r4(MapFormerEM):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, 4)
        u = self.q0_pos.data.clone()
        del self._parameters["k0_pos"]
        self.k0_u = nn.Parameter(u)
        assert "k0_pos" not in dict(self.named_parameters()), "k0_pos still a parameter"
        assert self.action_to_lie.w_in.out_features == 4, "rank lost in construction"

    @property
    def k0_pos(self):
        H, D = self.q0_pos.shape
        q = self.q0_pos.view(H, D // 2, 2)
        qhat = q / (q.norm(dim=-1, keepdim=True) + _EPS)
        mag = self.k0_u.view(H, D // 2, 2).norm(dim=-1, keepdim=True)
        return (qhat * mag).reshape(H, D)
