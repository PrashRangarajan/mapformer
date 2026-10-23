"""SIGNSEP (SIGNSEP_PREREG.md): separate the two remaining accounts of the |.| learnability cost on the ring.

SIGNROUTE's hybrid (non-negative readout matrix, signed increment) solved 16/16 and removed BOTH accounts at once
(its Amendment 1, finding 2):

  (A) MISSING COLLECTIVE ROUTE (SEARCH_THEORY S3). Under |.| the latent move that cancels every signed channel
      ends at a(y) = a(x) in every channel (evenness), and at a generic readout no latent choice wraps more than
      2r of the C channels (the wrap targets 2 pi n / omega_c differ across the omega ladder). Cancellation must
      be found channel by channel.
  (B) WRAP BARRIER (S1 + appendix P3, with the kink of P4). A sign-coherent channel's only non-blind cancelling
      points are k = a(x) + a(y) = 2 pi n; any path to one from |k| < pi crosses |k| = pi (P3(d)), and the kink at
      zero (P4) is the sticky wall on the near side.

This file builds two NEW classes that each remove exactly one of them, on the same start as the two anchors:

                        no barrier (signed increment)          barrier (non-negative increment)
  collective route      signed            (anchor, S)          abs_wrap        (NEW, W)
  no collective route   signed_frozen     (NEW, F)             abs             (anchor, A)
  + control C1          abs_wrap_past: W's class, started past the barrier along W's collective coordinate

  abs_wrap   a_c(x) = omega_c * | w_c . z_x + tau_x * pi / |omega_c| |        (tau_x a learnable scalar per move)
             Every move angle has the sign of omega_c (a |.| class: S1, P3, P4 apply verbatim). On the region where
             the pre-|.| steps are >= 0 the clock is LINEAR in (u, T), u = z_x + z_y, T = tau_x + tau_y:
                 k_c = omega_c w_c . u + pi sign(omega_c) T,
             so {u = 0, T = 2} cancels EVERY channel at once by wrapping (k_c = 2 pi sign(omega_c)), for any readout:
             the fixed readout column pi / |omega_c| is exactly the target vector 2 pi n / omega_c (n = 1) that S3(c)
             says a generic readout lacks. From the common start (tau = 0) the straight route to it raises every
             clock together, monotonically, from k_c(0) to 2 pi, crossing pi in every channel that starts below pi.
             It ends, mod 2 pi, at the SAME function as the signed collective route from the same start (a_x0, -a_x0).
  signed_frozen  SignedAO_r4 with the two move latents frozen at the start (ACCOUNTS T2's SO_absmag_fz): W_in gets no
             grad, the step path reads a buffer copy of the move embeddings. Each channel's two angles are then free
             per-channel coordinates (a_c(x) = omega_c c W[c,0], a_c(y) = omega_c c W[c,1]): every step-path
             parameter moves exactly one channel's clock, so there is no collective direction; through-zero
             cancellation is available per channel with |k| non-increasing (P3(d), signed half): no barrier, no wall.
  abs_wrap_past  abs_wrap with tau_x = tau_y = TAU_PAST / 2 at the start: every channel's clock is moved past pi along
             the collective coordinate, everything else as abs_wrap (MECHANISM's M2 logic applied to W's route).

THE COMMON START (every arm, `build`). The seed's default init, then MECHANISM's balanced-gauge |.| start
(`train_wrap_mech.set_move_angles(m, |default angles|)`): move latents c e1, c e2, omega' = max(|omega|, 1),
W_out[:, :2] = |angle| / (omega' c), W_out[:, 2:] = 0. All pre-|.| steps are >= 0 there, so |.| is the identity and
signed, abs, abs_wrap (tau = 0) and signed_frozen are FUNCTION-IDENTICAL at init (test_signsep.py checks the logits);
abs_wrap_past differs only by tau. This is MECHANISM's SO_absmag / AO_absmag start (signed 7/8, |.| 0/8 there), not the
LRSWEEP / SIGNROUTE default start: the 2x2 needs one start for all four cells, and abs_wrap's route is clean only
where the pre-|.| steps are non-negative.

Parameters (measured, vocab 19): 204,243 for signed / abs / signed_frozen (signed_frozen has 512 of them, W_in, with no
grad); 204,245 for the abs_wrap arms (+ tau). The balanced start clamps omega' = max(|omega|, 1), so 30 of the 64 channels
sit at omega = 1 and the ladder ratio is 6.3, not 32 -- the same for every arm.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from mapformer.model_signroute import _signroute_variant
from mapformer.train_wrap_mech import set_move_angles

TAU_PAST = 1.5          # abs_wrap_past: tau_x = tau_y = 0.75 at the start (k_c(0) + 1.5 pi; see SIGNSEP_PREREG.md Sec. 2)
OMEGA_FLOOR = 1e-6      # |omega| clamp inside pi / |omega| (the balanced start has |omega| >= 1)


def _wrap_variant(mode):
    """`mode` ('abs' or 'signed') with the collective wrap coordinate tau (one scalar per move)."""
    Base = _signroute_variant(mode)

    class _W(Base):
        def __init__(self, vocab_size, *a, **kw):
            super().__init__(vocab_size, *a, **kw)
            # zeros: no RNG is consumed, so every other weight equals the base arm's at a seed
            self.wrap_tau = nn.Parameter(torch.zeros(len(self.act)))

        def pre_steps(self):
            """(A, C) the pre-|.| steps w_c . z_x + tau_x pi / |omega_c|."""
            e = self.token_emb.weight
            ea = self.frozen_act_emb if self.freeze_latent else e[self.act]
            a2l = self.action_to_lie
            pre = a2l.w_out(a2l.w_in(ea))
            om = self.path_integrator.omega.reshape(-1)
            return pre + self.wrap_tau[:, None] * (math.pi / om.abs().clamp_min(OMEGA_FLOOR))[None, :]

        def step_table(self):
            pre = self.pre_steps()
            v = pre.abs() if mode == "abs" else pre
            v = v.view(len(self.act), self.n_heads, self.n_blocks)
            e = self.token_emb.weight
            out = torch.zeros(e.shape[0], *v.shape[1:], device=e.device, dtype=v.dtype)
            out[self.act] = v
            return out

    _W.__name__ = f"MapFormerWM_{mode}WrapAO_r4"
    _W.__doc__ = f"Observation-step-free MapFormer-WM, rank 4, step {mode!r} plus the collective wrap coordinate tau."
    return _W


MapFormerWM_SignedAO_r4 = _signroute_variant("signed")      # S and (frozen) F
MapFormerWM_AbsAO_r4 = _signroute_variant("abs")            # A
MapFormerWM_AbsWrapAO_r4 = _wrap_variant("abs")             # W and W_past

ARM_CLASS = {"signed": MapFormerWM_SignedAO_r4, "abs": MapFormerWM_AbsAO_r4,
             "abs_wrap": MapFormerWM_AbsWrapAO_r4, "signed_frozen": MapFormerWM_SignedAO_r4,
             "abs_wrap_past": MapFormerWM_AbsWrapAO_r4}
ARMS = tuple(ARM_CLASS)
CELL = {"signed": "route, no barrier (anchor S)", "abs": "no route, barrier (anchor A)",
        "abs_wrap": "route, barrier (NEW W)", "signed_frozen": "no route, no barrier (NEW F)",
        "abs_wrap_past": "route, start past the barrier (control C1)"}


@torch.no_grad()
def default_angles(m):
    """(A, C) the seed's default SIGNED move angles omega * W_out W_in e (identical for every arm at a seed)."""
    a2l = m.action_to_lie
    st = a2l.w_out(a2l.w_in(m.token_emb.weight[m.act]))
    return st * m.path_integrator.omega.reshape(-1)


def build(arm, vocab_size, action_ids, seed=None, **kw):
    """The arm at the COMMON START. Seeds torch if `seed` is given (the trainer seeds before calling)."""
    if seed is not None:
        torch.manual_seed(seed)
    m = ARM_CLASS[arm](vocab_size=vocab_size, d_model=kw.pop("d_model", 128), n_heads=kw.pop("n_heads", 2),
                       n_layers=kw.pop("n_layers", 1), grid_size=kw.pop("grid_size", 32),
                       action_ids=action_ids, **kw)
    ang0 = default_angles(m)
    err = set_move_angles(m, ang0.abs())
    if arm == "abs_wrap_past":
        with torch.no_grad():
            m.wrap_tau.fill_(TAU_PAST / 2)
    if arm == "signed_frozen":
        m.freeze_move_latents()
    return m, {"start_angle_err": err, "default_angles": ang0}


@torch.no_grad()
def angles(m):
    """(A, C) the move angles omega * step (raw, not wrapped)."""
    return m.step_table()[m.act].reshape(len(m.act), -1) * m.path_integrator.omega.reshape(-1)


def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.no_grad()
def diagnostics(m, big=0.05, tol=0.1):
    """Descriptive end-state readouts (CPU, after training): the collective coordinates and how channels cancel.
    A channel CANCELS if |wrap(a(x))| > big and |wrap(a(x) + a(y))| < tol; it cancels THROUGH ZERO if its raw sum is
    below pi in magnitude, BY WRAPPING otherwise (MECHANISM's post hoc split)."""
    m = m.cpu()
    z = m.latents().double()
    u = (z[0] + z[1]).norm(); den = z[0].norm() + z[1].norm()
    A = angles(m).double()
    k = A[0] + A[1]
    sel = _wrap(A[0]).abs() > big
    canc = sel & (_wrap(k).abs() < tol)
    out = {"u_ratio": float(u / den) if float(den) > 0 else float("nan"),
           "latent_cos": float(torch.nn.functional.cosine_similarity(z[0], z[1], dim=0)),
           "clock_wrapped_rms": float(_wrap(k).pow(2).mean().sqrt()),
           "cancel_share": float(canc.double().sum() / max(1, int(sel.sum()))),
           "cancel_zero_share": float((canc & (k.abs() < math.pi)).double().sum() / max(1, int(sel.sum()))),
           "cancel_wrap_share": float((canc & (k.abs() >= math.pi)).double().sum() / max(1, int(sel.sum()))),
           "sign_coherent_share": float(((A[0] * A[1]) >= 0).double().mean()),
           "step_frac_negative": float((m.step_table()[m.act] < 0).double().mean())}
    if hasattr(m, "wrap_tau"):
        t = m.wrap_tau.detach().double()
        out.update({"tau_x": float(t[0]), "tau_y": float(t[1]), "tau_sum": float(t.sum()),
                    "pre_frac_negative": float((m.pre_steps() < 0).double().mean())})
    return out


@torch.no_grad()
def start_stats(m):
    """Clock at the start: median raw |k_c| and the share of channels already past pi."""
    A = angles(m).double(); k = (A[0] + A[1]).abs()
    return {"k0_median": float(k.median()), "k0_share_past_pi": float((k > math.pi).double().mean()),
            "k0_max": float(k.max())}
