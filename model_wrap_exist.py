"""Rule 14 existence for the ring learnability claim (WRAP_EXISTENCE_PREREG.md).

RING_DIAL: a non-negative rotary step (Abs_r4, |W_out W_in x|) learns the ring map on 2/8 seeds vs 8/8 for the signed step,
although a non-negative rotary step can REPRESENT any signed action code (angles wrap mod 2 pi: a step of
2 pi / omega - Delta reproduces -Delta). The first existence attempt (`docs/audits/2026-10-13/wrap_existence_build.py`)
found that trained Signed_r4 ring solutions rely on their observation steps, which the rank-4 bottleneck cannot also wrap.
So this test uses models whose observation and blank steps are 0 BY CONSTRUCTION and whose two action steps are free
(RANK_TESTS' R1AO, at the shared rank 4 of Signed_r4 / Abs_r4):
  SignedAO_r4   step(+1) = W_out W_in e(+1), step(-1) = W_out W_in e(-1), every other token 0
  AbsAO_r4      the same with |.| per channel (the Abs_r4 constraint)
Both are built from MapWM's base at the seed, so at one seed their initial weights are identical (Abs differs only by |.|).
`wrap_construct(signed)` returns an AbsAO_r4 whose action angles equal the signed model's mod 2 pi: W_in and every other
weight copied, W_out[:, c] the minimum-norm solution of  latent(a) . W_out[c] = (omega_c Delta_c(a) mod 2 pi) / omega_c.
"""
import math

import torch

from mapformer.model_sign import _sign_variant


def _ao_variant(mode):
    base = _sign_variant(mode, 4)

    class _AO(base):
        def __init__(self, vocab_size, *a, action_ids=(0, 1), **kw):
            super().__init__(vocab_size, *a, **kw)
            self.register_buffer("act", torch.tensor(action_ids, dtype=torch.long), persistent=False)

        def step_table(self):
            """(V, H, nb) per-token step before omega: the two actions' steps (after the sign constraint), 0 otherwise."""
            e = self.token_emb.weight
            v = self.action_to_lie(e[self.act].view(1, 2, -1))[0]          # (2, H, nb)
            out = torch.zeros(e.shape[0], *v.shape[1:], device=e.device, dtype=v.dtype)
            out[self.act] = v
            return out

        def delta_of(self, tokens):
            return self.step_table()[tokens]

        def get_position_state(self, tokens: torch.Tensor):
            return self.path_integrator(self.step_table()[tokens])

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            B, L = tokens.shape
            x = self.token_emb(tokens)
            cos_a, sin_a = self.path_integrator(self.step_table()[tokens])
            causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
            for layer in self.layers:
                x = layer(x, cos_a, sin_a, causal_mask)
            return self.out_proj(self.out_norm(x))

    _AO.__name__ = f"MapFormerWM_{mode}AO_r4"
    return _AO


MapFormerWM_SignedAO_r4 = _ao_variant("signed")
MapFormerWM_AbsAO_r4 = _ao_variant("abs")


def action_angles(m):
    """(2, C) per-channel angle step omega * step of the +1 and -1 actions."""
    with torch.no_grad():
        st = m.step_table()[m.act].reshape(2, -1)
        return st * m.path_integrator.omega.reshape(-1)


def wrap_share(m, big=0.05, tol=0.1):
    """Over channels with |omega step(+1)| > big: median |wrap(angle(+1) + angle(-1))| and the share below tol rad
    (0 = the two actions cancel mod 2 pi; RING_DIAL's wrap-aware opposition, on the AO step table)."""
    A = action_angles(m).double()
    sel = A[0].abs() > big
    w = torch.angle(torch.exp(1j * (A[0] + A[1])))[sel].abs()
    if w.numel() == 0:
        return float("nan"), float("nan")
    return float(w.median()), float((w < tol).double().mean())


def wrap_construct(signed, vocab_size, action_ids, **kw):
    """AbsAO_r4 with the signed model's action angles mod 2 pi (see module docstring). Returns (model, report)."""
    ab = MapFormerWM_AbsAO_r4(vocab_size, action_ids=action_ids, **kw)
    sd = {k: v.detach().clone() for k, v in signed.state_dict().items()}
    with torch.no_grad():
        om = signed.path_integrator.omega.detach().reshape(-1).double()                  # (C,)
        e = signed.token_emb.weight.detach().double()
        W_in = signed.action_to_lie.w_in.weight.detach().double()                        # (4, d)
        lat = e[signed.act] @ W_in.T                                                     # (2, 4)
        D = (lat @ signed.action_to_lie.w_out.weight.detach().double().T)                # (2, C) signed steps
        tgt = torch.remainder(om * D, 2 * math.pi) / om                                  # >= 0, same angle mod 2 pi
        W_out = (torch.linalg.pinv(lat) @ tgt).T                                         # (C, 4) min-norm
    sd["action_to_lie.w_out.weight"] = W_out.to(signed.action_to_lie.w_out.weight.dtype)
    ab.load_state_dict(sd)
    with torch.no_grad():
        Da = ab.step_table()[ab.act].reshape(2, -1).double()
        err = float(torch.angle(torch.exp(1j * om * (Da - D))).abs().max())
    rep = {"angle_err_max": err, "abs_step_min": float(Da.min()),
           "w_out_absmax_signed": float(signed.action_to_lie.w_out.weight.detach().abs().max()),
           "w_out_absmax_wrapped": float(W_out.abs().max()),
           "step_absmax_signed": float(D.abs().max()), "step_absmax_wrapped": float(Da.abs().max())}
    return ab, rep
