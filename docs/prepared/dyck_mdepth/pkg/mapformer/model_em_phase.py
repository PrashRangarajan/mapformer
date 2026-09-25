"""MapFormer-EM with the position kernel's COHERENCE set by construction.

N5 in THEORY_KERNEL.md. The position branch of EM's score is

    A_P[t,s] = q_0^T R(theta_s - theta_t) k_0
             = sum_i a_i cos( omega_i * dS + phi_i ),
      a_i   = |q_0i| |k_0i|          (per-block magnitude)
      phi_i = angle(q_0i, k_0i)      (per-block phase offset)

and its COHERENCE is

    rho = kappa(0) / sum_i a_i = ( sum_i a_i cos phi_i ) / sum_i a_i.

Theorem 2 says rho = 1 exactly when q_0 == k_0 (a matched filter peaked at zero
displacement -- what a MAP task wants) and E[rho] = 0, sd ~ 1/sqrt(2 n_b) when the
two are drawn independently. The five measured EM cells say the sign of the
single-p_0 advantage tracks the task kind, which is the corollary this module
exists to test by INTERVENTION rather than correlation.

Construction, and why it is magnitude-matched
---------------------------------------------
Draw q_0 as usual. Build k_0 by ROTATING each 2D block of q_0 by a chosen phi:

    k_0i = R(phi_i) q_0i          =>   a_i = |q_0i|^2  for EVERY condition.

So the per-block magnitudes a_i, the total kernel power sum_i a_i, and hence
||q_0||, ||k_0|| are IDENTICAL across all conditions; only the phases differ.
This matters: the recency gate ablation found that a control changing only
theta's SCALE collapsed as hard as the condition of interest (0.110 vs 0.086),
and only a magnitude-matched pair established anything there.

Conditions
----------
    phase="plus"   phi_i = 0      -> rho = +1 exactly. kappa peaked at dS = 0.
    phase="zero"   phi_i = pi/2   -> rho =  0 exactly, COHERENTLY (every block a
                                     quarter turn), so it isolates the phase VALUE
                                     from the phase RANDOMNESS.
    phase="minus"  phi_i = pi     -> rho = -1 exactly. kappa MINIMISED at dS = 0.
    phase="rand"   phi_i ~ U(0,2pi) -> rho ~ 0 incoherently: the paper-faithful
                                     draw, magnitude-matched to the other three.

q_0 and k_0 are FROZEN (requires_grad=False). rho is the controlled variable, and
`probe_ap_coherence.py` measured that a trainable pair does NOT converge toward
rho = 1 -- it drifts 0.25-0.27 MORE negative over training on both tasks -- so
leaving them trainable would let the variable wander off its set point.

Freezing removes 2 * n_heads * d_head trainable parameters from EVERY arm
equally, so the four conditions stay parameter-matched to each other. They are
NOT matched to a trainable-origin EM arm; do not cross-compare.
"""
import math

import torch
import torch.nn as nn

from mapformer.model import MapFormerEM


def _rotate_blocks(v, phi):
    """Rotate each 2D block of v (shape [n_heads, d_head]) by phi (broadcastable
    to [n_heads, n_blocks]). Returns the same shape."""
    H, D = v.shape
    b = v.view(H, D // 2, 2)
    c, s = torch.cos(phi), torch.sin(phi)
    out = torch.stack([b[..., 0] * c - b[..., 1] * s,
                       b[..., 0] * s + b[..., 1] * c], dim=-1)
    return out.reshape(H, D)


def coherence_of(q0, k0):
    """rho per head, for verification. Matches probe_ap_coherence.coherence."""
    H, D = q0.shape
    qa = q0.view(H, D // 2, 2).double()
    ka = k0.view(H, D // 2, 2).double()
    a = qa.norm(dim=-1) * ka.norm(dim=-1)
    cosphi = (qa * ka).sum(-1) / a.clamp_min(1e-30)
    return ((a * cosphi).sum(-1) / a.sum(-1))


_PHI = {"plus": 0.0, "zero": math.pi / 2, "minus": math.pi}


def _em_phase(phase, r=4, tol=1e-6):
    class _P(MapFormerEM):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, r)
            nb = self.n_blocks
            with torch.no_grad():
                q0 = self.q0_pos.data
                if phase == "rand":
                    phi = torch.rand(n_heads, nb) * (2 * math.pi)
                else:
                    phi = torch.full((n_heads, nb), _PHI[phase])
                self.k0_pos.data.copy_(_rotate_blocks(q0, phi))
            self.q0_pos.requires_grad_(False)
            self.k0_pos.requires_grad_(False)

            # Verify the controlled variable actually took the intended value,
            # and that the magnitude match is exact -- a probe that only checks
            # "it ran" is how five analysis bugs got through in this project.
            rho = coherence_of(self.q0_pos.data, self.k0_pos.data)
            if phase in ("plus", "zero", "minus"):
                want = math.cos(_PHI[phase])
                assert torch.allclose(rho, torch.full_like(rho, want), atol=tol), \
                    f"rho={rho.tolist()} but phase={phase} wants {want}"
            qn = self.q0_pos.data.view(n_heads, nb, 2).norm(dim=-1)
            kn = self.k0_pos.data.view(n_heads, nb, 2).norm(dim=-1)
            assert torch.allclose(qn, kn, atol=1e-6), "per-block magnitudes not matched"
            assert self.action_to_lie.w_in.out_features == r, "rank lost in construction"
            assert not self.q0_pos.requires_grad and not self.k0_pos.requires_grad, \
                "origin vectors are not frozen"
            self.phase, self.rho_set = phase, rho.tolist()

    _P.__name__ = f"MapFormerEM_Phase_{phase}_r{r}"
    _P.__doc__ = (f"MapFormer-EM, frozen origin vectors at phase={phase} "
                  f"(rho = {'random' if phase=='rand' else math.cos(_PHI[phase]):}), "
                  f"bottleneck rank r={r}. Magnitude-matched across conditions.")
    return _P


MapFormerEM_Phase_plus_r4 = _em_phase("plus")
MapFormerEM_Phase_zero_r4 = _em_phase("zero")
MapFormerEM_Phase_minus_r4 = _em_phase("minus")
MapFormerEM_Phase_rand_r4 = _em_phase("rand")
