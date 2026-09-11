"""Leakage test: close the content -> Delta channel in the unfreeze-at-0 warm start.

UNFREEZE_RESULTS.md U4: the rewind's effective slope can leave -1 only through (i) the
latent embedding code changing, or (ii) content leaking into Delta through w_in's
content columns -- w_out and w_in's latent columns preserve the cancellation by
construction. At the 8x install scale the latent code survived (-0.989) and leakage
(final 0.73) was the residual, holding accuracy at 0.941 instead of the frozen 1.000.

These variants are the unfreeze-at-step-0 arms with w_in's content columns (every
column beyond the two latent coordinates) held at zero by a gradient mask. They start
at zero in the construction, their gradient is masked, and AdamW's decoupled decay of
zero is zero, so they stay exactly zero. ActionToLieAlgebra has no bias, so Delta is
then a function of the latent code alone and the leak channel is closed completely.
Everything else trains, including the latent code, w_in's latent columns, w_out, omega
and p0.

Built as subclasses in a NEW module so model_em_unfreeze.py -- which defines the
EMUnf_0 / EMUnf_0_e8 arms these are compared against -- is untouched.
"""
import torch

from mapformer.model_em_unfreeze import _unfreeze, N_LAT
from mapformer.model_em_warm import VOCAB, N_SYM, Q0, K_MAX


def _noleak(eps):
    Base = _unfreeze(0, eps)

    class _N(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            w = self.action_to_lie.w_in.weight
            assert torch.count_nonzero(w[:, N_LAT:]) == 0, "content columns must start at zero"
            m = torch.zeros_like(w)
            m[:, :N_LAT] = 1.0
            self.register_buffer("win_grad_mask", m)
            w.register_hook(lambda g: g * self.win_grad_mask)
            # COMPLETE latent-pathway slope. The inherited `traj_lat` reads coordinate 0 of
            # the embedding code only; the code has two coordinates and every token row
            # trains, so coordinate-1 contamination and symbols drifting apart are further
            # routes by which the rewind breaks. `traj_latpath` is the same statistic as
            # `traj_slope` computed from the latent-driven Delta alone. With the leak
            # closed the two must be IDENTICAL -- that is the manipulation check.
            self.register_buffer("traj_latpath", torch.full_like(self.traj_slope, float("nan")))

        @torch.no_grad()
        def _record(self, ep):
            super()._record(ep)
            x = self.token_emb(torch.arange(VOCAB, device=self.p0_pos.device))
            wi, wo = self.action_to_lie.w_in.weight, self.action_to_lie.w_out.weight
            dl = (x[:, :N_LAT] @ wi[:, :N_LAT].T) @ wo.T
            k = torch.arange(K_MAX, device=dl.device, dtype=dl.dtype)
            sym = dl[:N_SYM].mean(0)
            y = dl[Q0:Q0 + K_MAX] @ sym / (sym @ sym)
            self.traj_latpath[ep] = ((k - k.mean()) * (y - y.mean())).sum() / ((k - k.mean()) ** 2).sum()

    _N.__name__ = f"MapFormerEM_NoLeak_eps{round(1/eps)}_r4"
    return _N


MapFormerEM_NoLeak_e8_r4 = _noleak(1 / 8)
MapFormerEM_NoLeak_e64_r4 = _noleak(1 / 64)
