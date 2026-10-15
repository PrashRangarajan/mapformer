"""TORUS_START (TORUS_START_PREREG.md): the MECHANISM start manipulation on the 2D torus. Observation-step-free 4-move models
(SignedAO4_r4 / AbsAO4_r4 from `model_wrap_exist.py`, imported, not edited) whose four move angles are set at init, in the
balanced gauge (move latents c e_i, i = 0..3 = r; omega' = max(|omega|, 1)), to iid U(lo, hi) rad per channel and move:
  *_High  U(1.7, 4.5): each opposite pair (N/S, W/E) sums past pi (MECHANISM's AO_high / SO_high)
  *_Mid   U(0.9, 1.5): large but every pair sums below pi (MECHANISM's AO_mid)
The angles come from torch.Generator(seed = torch.initial_seed() + 7919), so they consume no global RNG: every other weight
equals the default arm's at a seed, and *_High arms of both signs get the same angles."""
import torch

from mapformer.model_wrap_exist import MapFormerWM_SignedAO4_r4, MapFormerWM_AbsAO4_r4, action_angles


@torch.no_grad()
def set_move_angles_n(m, ang):
    """Set the move angles (A, C) in the balanced gauge; returns the max wrap-aware angle error (rad)."""
    e = m.token_emb.weight.double(); W_in = m.action_to_lie.w_in.weight.double(); r = W_in.shape[0]; A = ang.shape[0]
    assert A <= r, (A, r)
    lat = e[m.act] @ W_in.T; c = float(lat.norm(dim=1).mean())
    L = torch.zeros(A, r, dtype=torch.float64); L[torch.arange(A), torch.arange(A)] = c
    m.action_to_lie.w_in.weight.copy_((torch.linalg.pinv(e[m.act]) @ L).T.float())
    om = m.path_integrator.omega.double().reshape(-1).abs().clamp_min(1.0)
    m.path_integrator.omega.copy_(om.reshape(m.path_integrator.omega.shape).float())
    W = torch.zeros_like(m.action_to_lie.w_out.weight, dtype=torch.float64); W[:, :A] = (ang.double() / om).T / c
    m.action_to_lie.w_out.weight.copy_(W.float())
    return float(torch.angle(torch.exp(1j * (action_angles(m).double() - ang.double()))).abs().max())


def _start(base, lo, hi):
    class _S(base):
        def __init__(self, vocab_size, *a, **kw):
            super().__init__(vocab_size, *a, **kw)
            g = torch.Generator().manual_seed(torch.initial_seed() + 7919)
            C = self.path_integrator.omega.numel()
            ang = lo + (hi - lo) * torch.rand(len(self.act), C, generator=g)
            self.init_angle_err = set_move_angles_n(self, ang)
    return _S


MapFormerWM_AbsAO4_High = _start(MapFormerWM_AbsAO4_r4, 1.7, 4.5)
MapFormerWM_SignedAO4_High = _start(MapFormerWM_SignedAO4_r4, 1.7, 4.5)
MapFormerWM_AbsAO4_Mid = _start(MapFormerWM_AbsAO4_r4, 0.9, 1.5)
ARMS = {"SO4": MapFormerWM_SignedAO4_r4, "AO4": MapFormerWM_AbsAO4_r4, "AO4_high": MapFormerWM_AbsAO4_High,
        "SO4_high": MapFormerWM_SignedAO4_High, "AO4_mid": MapFormerWM_AbsAO4_Mid}
