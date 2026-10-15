"""A0 (propagator identity). (1) Integrate Howard's homogeneous equation dF/dt = -s alpha(t) F, s = i omega, with alpha
piecewise constant (alpha = Delta_u on (u-1, u]) by RK4 at 2000 substeps per token, and compare F(t)/F(t') with
exp(-i omega (Phi_t - Phi_t')), Phi = cumsum(Delta). (2) The rotary score of model._apply_rope, q^T R(theta_t)^T R(theta_s) k,
equals Re(conj(q~) k~ G(t, s)) with q~ = q_even + i q_odd and G the propagator above. (3) Real s gives the decay
exp(-s (Phi_t - Phi_s)) -- the additive-bias slot (GRAPE-AP / FoX: log of it is -s (Phi_t - Phi_s))."""
import math
import numpy as np
import torch
from common import REPO
from mapformer.model import _apply_rope

rng = np.random.RandomState(0)
out = []
P = lambda s: (print(s), out.append(s))
T, om = 20, 0.7
for name, Delta in (("signed", rng.normal(0, 1, T)), ("non-negative", np.abs(rng.normal(0, 1, T))), ("constant", np.ones(T))):
    F = [1.0 + 0j]; f = 1.0 + 0j; n = 2000; h = 1.0 / n
    for u in range(T):
        a = Delta[u]
        rhs = lambda F_: -1j * om * a * F_
        for _ in range(n):
            k1 = rhs(f); k2 = rhs(f + h / 2 * k1); k3 = rhs(f + h / 2 * k2); k4 = rhs(f + h * k3)
            f = f + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F.append(f)
    F = np.array(F); Phi = np.concatenate([[0], np.cumsum(Delta)])
    err = max(abs(F[t] / F[s] - np.exp(-1j * om * (Phi[t] - Phi[s]))) for t in range(T + 1) for s in range(t + 1))
    P(f"(1) {name:12s} rate: max |ODE propagator - exp(-i omega (Phi_t - Phi_s))| over all pairs = {err:.2e}")

# (2) rotary score vs propagator
torch.manual_seed(0)
nb = 8; q = torch.randn(1, 1, T, 2 * nb, dtype=torch.float64); k = torch.randn(1, 1, T, 2 * nb, dtype=torch.float64)
omega = torch.rand(nb, dtype=torch.float64) * 2; Delta = torch.randn(T, nb, dtype=torch.float64)
theta = (torch.cumsum(Delta, 0) * omega)[None, None]
S = _apply_rope(q, theta.cos(), theta.sin()) @ _apply_rope(k, theta.cos(), theta.sin()).transpose(-1, -2)
qc = torch.complex(q[..., 0::2], q[..., 1::2])[0, 0]; kc = torch.complex(k[..., 0::2], k[..., 1::2])[0, 0]
G = torch.exp(-1j * (theta[0, 0][:, None, :] - theta[0, 0][None, :, :]))           # (t, s, c)
S2 = (qc.conj()[:, None, :] * kc[None, :, :] * G).real.sum(-1)
P(f"(2) max |rotary score (model._apply_rope) - sum_c Re(conj(q~_c) k~_c G_c(t,s))| = {(S[0, 0] - S2).abs().max():.2e}")
# (3) real s
s_real = 0.3; Phi = np.concatenate([[0], np.cumsum(np.abs(rng.normal(0, 1, T)))])
P(f"(3) real s = {s_real}: propagator exp(-s (Phi_t - Phi_s)) in (0, 1] for a non-negative rate (monotone forgetting); "
  f"its log is the additive bias -s (Phi_t - Phi_s) (GRAPE-AP's b_h with theta_h = s); min over pairs "
  f"{min(math.exp(-s_real * (Phi[t] - Phi[s])) for t in range(T + 1) for s in range(t + 1)):.3e} > 0")
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")
