"""MONOTONE_PREREG.md: the sign ablation, carried outside MapFormer-WM.

SIGN_ABLATION.md removed the sign of the phase increment INSIDE one architecture
(MapFormer-WM) and found a map task needs it (-0.280 loss-matched at T=1024) while a
recency task does not (unmeasured). Two gaps, both closed here with the same knob
(|.| on the final per-channel increment, nothing else):

  1. MapFormer-EM has never been run monotone. EM's recency solution is a SUBTRACTION --
     the query token rewinds theta by -(k-1) symbol steps (SEARCH_RESULTS.md) -- so
     "recency needs only a clock" was established only on the architecture that does
     not have to rewind. theta enters through cos/sin, so a forward shift of
     2 pi n / omega_i - (k-1) c_i is an equivalent per block; whether training finds it
     is the question.

  2. A second, natively signed generator: Selective RoPE's (model_selective.py,
     SRoPEGen): w = gate * conv1d(W_omega x), theta = temp * cumsum(w). Signed because
     W_omega x is; the sigmoid gate only scales it.

CONSTRUCTION, stated because it is the manipulation check. Both constrained models are
the SAME FUNCTION as their signed parent at init except for |.|:
  - EM: the constrained ActionToLie is built inside fork_rng (no CPU RNG consumed) and
    loads the parent's weights, so at seed s every parameter equals VanillaEM_P0_r4's.
  - SRoPE: no module is replaced; only SelectiveAngle.forward changes.
"""
import torch

from mapformer.model_em_fixed import MapFormerEM_SingleP0_r4
from mapformer.model_selective import MapFormerWM_SRoPEGen, SelectiveAngle
from mapformer.model_sign import SignConstrainedActionToLie


def _em_p0_sign(mode):
    class _E(MapFormerEM_SingleP0_r4):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size)
            old = self.action_to_lie
            with torch.random.fork_rng(devices=[]):
                new = SignConstrainedActionToLie(d_model, n_heads, self.n_blocks,
                                                 old.w_in.out_features, mode=mode)
            new.load_state_dict(old.state_dict())
            self.action_to_lie = new
    _E.__name__ = f"MapFormerEM_SingleP0_{mode}_r4"
    _E.__doc__ = f"MapFormer-EM, shared p0, r=4, phase increment constrained: {mode}."
    return _E


MapFormerEM_P0_Abs_r4 = _em_p0_sign("abs")
MapFormerEM_P0_Signed_r4 = _em_p0_sign("signed")   # construction check only


class _MonotoneSelectiveAngle(SelectiveAngle):
    """theta_t = temp * cumsum( |gate . conv1d(W_omega x)| )_t."""

    def forward(self, x):
        B, T, _ = x.shape
        w = self.proj(x)
        if self.conv is not None:
            w = self.conv(w)
        if self.gate is not None:
            w = w * torch.sigmoid(self.gate(x))
        theta = self.log_temp.exp() * torch.cumsum(w.abs(), dim=1)
        return theta.view(B, T, self.n_heads, self.n_blocks)


class MapFormerWM_SRoPEGen_Abs(MapFormerWM_SRoPEGen):
    """Selective RoPE's full generator with the increment forced non-negative."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.angle.__class__ = _MonotoneSelectiveAngle   # same parameters, new forward
