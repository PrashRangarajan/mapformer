"""Freeze-then-unfreeze warm-start, with the rewind's fate RECORDED in the checkpoint.

WARM_RESULTS.md: single-p0 EM with the recency rewind installed and FROZEN scores 1.000
(8/8); the same install left TRAINABLE falls to from-scratch level. Two explanations:

  early window  d score / d A_P = A_X, so while the content branch is random the position
                pathway gets noise gradients that dismantle the rewind. Predicts survival
                if the pathway is released only after the content branch has learned.
  install scale the rewind lives in embedding coords at EPS = 1/64 (one symbol step =
                0.0156), and Adam moves every coordinate ~lr per step regardless of scale.
                Predicts survival if the SAME rewind is installed at a larger EPS.

Factory `_unfreeze(unfreeze_epoch, eps)`. The position pathway (latent code, w_in, w_out,
omega, p0) is frozen until training step `unfreeze_epoch * STEPS_PER_EPOCH`, then released
in place. Every pathway tensor is a Parameter from the start, so it is already in the
optimiser (AdamW skips parameters whose grad is None, which also means no weight decay
while frozen -- same treatment as EMWarm_freeze).

Recorded every epoch into buffers, i.e. saved with the checkpoint:
  traj_slope   effective rewind slope from the full Delta (-1 = exact, 0 = none)
  traj_lat     rewind slope of the latent embedding code alone (is the code itself kept?)
  traj_leak    ||Delta from content coords|| / ||Delta from latent coords||, over symbol
               tokens (is content leaking into Delta through w_in?)
The recording runs under no_grad and draws no random numbers, so it cannot change training.

With unfreeze_epoch=0 and eps=1/64 this is EMWarm_train plus the recording; construction
order is copied from model_em_warm so the RNG draws match. That equivalence is CHECKED
(bitwise weights) rather than assumed.
"""
import math

import torch
import torch.nn as nn

from mapformer.model_em_fixed import MapFormerEM_SingleP0_r4
from mapformer.model_em_warm import (_LatentEmbedding, rewind_latent, N_LAT, W_SCALE,
                                     P0_NORM, VOCAB, N_SYM, Q0, K_MAX)

STEPS_PER_EPOCH = 48          # run_unfreeze.sh passes --n-batches 48
N_EPOCHS = 300


def _unfreeze(unfreeze_epoch: int, eps: float):
    class _U(MapFormerEM_SingleP0_r4):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, bottleneck_r)
            assert vocab_size == VOCAB
            H, nb = self.n_heads, self.n_blocks
            self.token_emb = _LatentEmbedding(self.token_emb, eps * rewind_latent(),
                                              trainable=True)
            with torch.no_grad():
                wi = self.action_to_lie.w_in.weight
                wi.zero_(); wi[0, 0] = 1.0 / eps; wi[1, 1] = 1.0 / eps
                wo = self.action_to_lie.w_out.weight
                wo.zero_(); wo[:, :N_LAT] = torch.randn(H * nb, N_LAT) * W_SCALE / (2 * math.pi)
                p = self.p0_pos.data
                self.p0_pos.data.copy_(p / p.norm(dim=-1, keepdim=True) * P0_NORM)
            self._pathway = [self.token_emb.latent, self.action_to_lie.w_in.weight,
                             self.action_to_lie.w_out.weight, self.path_integrator.omega,
                             self.p0_pos]
            self.unfreeze_step = unfreeze_epoch * STEPS_PER_EPOCH
            if self.unfreeze_step > 0:
                for prm in self._pathway:
                    prm.requires_grad_(False)
            self._step = 0
            for name in ("traj_slope", "traj_lat", "traj_leak"):
                self.register_buffer(name, torch.full((N_EPOCHS,), float("nan")))

        @torch.no_grad()
        def _record(self, ep):
            x = self.token_emb(torch.arange(VOCAB, device=self.p0_pos.device))
            wi, wo = self.action_to_lie.w_in.weight, self.action_to_lie.w_out.weight
            d_lat = (x[:, :N_LAT] @ wi[:, :N_LAT].T) @ wo.T
            d_con = (x[:, N_LAT:] @ wi[:, N_LAT:].T) @ wo.T
            d = d_lat + d_con
            k = torch.arange(K_MAX, device=d.device, dtype=d.dtype)
            def slope(y):
                return ((k - k.mean()) * (y - y.mean())).sum() / ((k - k.mean()) ** 2).sum()
            sym = d[:N_SYM].mean(0)
            self.traj_slope[ep] = slope(d[Q0:Q0 + K_MAX] @ sym / (sym @ sym))
            lat = self.token_emb.latent[:, 0]
            self.traj_lat[ep] = slope(lat[Q0:Q0 + K_MAX] / lat[:N_SYM].mean())
            self.traj_leak[ep] = (d_con[:N_SYM].norm(dim=-1) /
                                  d_lat[:N_SYM].norm(dim=-1).clamp_min(1e-12)).mean()

        def forward(self, tokens):
            if self.training:
                if self._step == self.unfreeze_step and self.unfreeze_step > 0:
                    for prm in self._pathway:
                        prm.requires_grad_(True)
                if self._step % STEPS_PER_EPOCH == 0 and self._step // STEPS_PER_EPOCH < N_EPOCHS:
                    self._record(self._step // STEPS_PER_EPOCH)
                self._step += 1
            return super().forward(tokens)

    _U.__name__ = f"MapFormerEM_Unfreeze{unfreeze_epoch}_eps{round(1/eps)}_r4"
    return _U


MapFormerEM_Unf0_r4 = _unfreeze(0, 1 / 64)
MapFormerEM_Unf5_r4 = _unfreeze(5, 1 / 64)
MapFormerEM_Unf30_r4 = _unfreeze(30, 1 / 64)
MapFormerEM_Unf100_r4 = _unfreeze(100, 1 / 64)
MapFormerEM_Unf0_e8_r4 = _unfreeze(0, 1 / 8)
