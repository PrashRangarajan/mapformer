"""Referee checks (scratch, not part of the paper): (1) A0 inhomogeneous term; (2) softplus (Pos_r4) exact representation of a
trained signed AO ring solution via annihilated observations + omega' = 2 pi / ln 2."""
import math, numpy as np, torch
import sys; sys.path.insert(0, "/home/prashr/mapformer/docs/paper/appendix_checks")
from common import REPO, ring_batch
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4
from mapformer.model_sign import MapFormerWM_Pos_r4
torch.set_default_dtype(torch.float64)
# (1) dF/dt = alpha (-s F + f), alpha = D_u, f = f_u on (u-1, u]; RK4
rng = np.random.RandomState(0); T = 12; om = 0.7; s = 1j * om
D = rng.normal(0, 1, T); fu = rng.normal(0, 1, T) + 1j * rng.normal(0, 1, T)
F = 0j; n = 4000; h = 1.0 / n
for u in range(T):
    rhs = lambda x: D[u] * (-s * x + fu[u])
    for _ in range(n):
        k1 = rhs(F); k2 = rhs(F + h / 2 * k1); k3 = rhs(F + h / 2 * k2); k4 = rhs(F + h * k3); F = F + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
Phi = np.cumsum(D)
stated = sum(np.exp(-s * (Phi[-1] - Phi[u])) * D[u] * fu[u] for u in range(T))
fixed = sum(np.exp(-s * (Phi[-1] - Phi[u])) * (1 - np.exp(-s * D[u])) / s * fu[u] for u in range(T))
print(f"(1) A0 input term, F(0)=0: |ODE - stated (Delta_u f_u)| = {abs(F - stated):.3e}; |ODE - fixed ((1-e^(-s Delta_u))/s f_u)| = {abs(F - fixed):.3e}; |F| = {abs(F):.3f}")
# (2) softplus construction
b = torch.load(f"{REPO}/runs/wrap_mech/p0/SO_s800/SO.pt", map_location="cpu", weights_only=False)
V = b["config"]["vocab_size"]
so = MapFormerWM_SignedAO_r4(vocab_size=V, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1)).double()
so.load_state_dict(b["model_state_dict"]); so.eval()
pos = MapFormerWM_Pos_r4(vocab_size=V, d_model=128, n_heads=2, n_layers=1, grid_size=32).double().eval()
sd = {k: v for k, v in so.state_dict().items()}
pos.load_state_dict(sd, strict=False)
with torch.no_grad():
    E = so.token_emb.weight
    om = so.path_integrator.omega.reshape(-1)
    Th = so.step_table()[so.act].reshape(2, -1) * om                    # (2, C) signed move angles
    L = torch.zeros(V, 4); L[0, 0] = 1.0; L[1, 1] = 1.0                  # moves -> u1, u2; every other token -> 0
    pos.action_to_lie.w_in.weight.copy_((torch.linalg.pinv(E) @ L).T)
    omp = torch.full_like(om, 2 * math.pi / math.log(2.0))
    y = (torch.remainder(Th, 2 * math.pi) + 2 * math.pi) / omp          # target steps > 0
    Wout = torch.zeros(64, 4); Wout[:, :2] = torch.log(torch.expm1(y)).T  # softplus^{-1}
    pos.action_to_lie.w_out.weight.copy_(Wout); pos.path_integrator.omega.copy_(omp.reshape(2, 32))
    St = pos.action_to_lie(E[None])[0].reshape(V, -1) * omp           # (V, C) angles of the Pos model
    tgt = torch.zeros(V, 64); tgt[0] = Th[0]; tgt[1] = Th[1]
    err = torch.angle(torch.exp(1j * (St - tgt))).abs().max()
    toks, _, _, _ = ring_batch(64, seed=4321)
    dl = (pos(toks) - so(toks)).abs().max(); sc = so(toks).abs().max()
    print(f"(2) Pos_r4 built from SO_s800: min step {float((St / omp).min()):.4f} (> 0), max angle error mod 2 pi over all {V} tokens "
          f"{float(err):.2e} rad; max |logit diff| vs SO_s800 {float(dl):.2e} (logit scale {float(sc):.1f})")
