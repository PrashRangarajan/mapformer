# Existence: Abs (|W_out W_in x|) realises the SAME rotary angles mod 2pi as a Signed ring code.
import math, torch, numpy as np
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment_cancel import GridWorldCancel
N = 32
env = GridWorldCancel(size=N, seed=30, p_plus=0.5)
torch.manual_seed(30)
S = VARIANT_MAP["Signed_r4"](vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=N)
torch.manual_seed(30)
A = VARIANT_MAP["Abs_r4"](vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=N)
H, nb = 2, S.n_blocks
om = S.path_integrator.omega.detach()           # (H, nb), identical in both
E = S.token_emb.weight.detach()                  # identical in both
# latent: w_in rows project onto 2 directions that are zero on every obs token, distinct on the two actions
obsE = E[2:]                                     # 17 obs tokens
Q, _ = torch.linalg.qr(obsE.T)                   # basis of obs span (128 x 17)
def perp(v): return v - Q @ (Q.T @ v)
u, v = perp(E[0]), perp(E[1])
W_in = torch.zeros(4, 128); W_in[0] = u; W_in[1] = v
lat = W_in @ E.T                                  # (4, 19): obs columns ~0
# target per-channel steps: harmonic m_c, step 2*pi*m/N for '+'
m = torch.randint(1, N, (H, nb)).float()
plus = 2 * math.pi * m / N / om                   # Delta+ (positive)
minus_signed = -plus
minus_abs = 2 * math.pi / om - plus               # >= 0 since m < N
assert (minus_abs >= 0).all()
def set_wout(model, dp, dm):
    M = torch.stack([lat[:2, 0], lat[:2, 1]])     # 2x2: rows = tokens, cols = latent dims 0,1
    tgt = torch.stack([dp.reshape(-1), dm.reshape(-1)])   # 2 x (H*nb)
    sol = torch.linalg.solve(M, tgt)              # 2 x (H*nb)
    W = torch.zeros(H * nb, 4); W[:, :2] = sol.T
    model.action_to_lie.w_in.weight.data = W_in.clone(); model.action_to_lie.w_out.weight.data = W.clone()
set_wout(S, plus, minus_signed); set_wout(A, plus, minus_abs)
np.random.seed(7)
tok, _, rev = env.generate_trajectory(512)
tok = tok[None]
with torch.no_grad():
    cS, sS = S.path_integrator(S.action_to_lie(S.token_emb(tok)))
    cA, sA = A.path_integrator(A.action_to_lie(A.token_emb(tok)))
    dA = A.action_to_lie(A.token_emb(tok))
print("Abs deltas min", dA.min().item(), "(non-negative), obs-token |Delta| max", dA[0, 1::2].abs().max().item())
print("max |cos diff|", (cS - cA).abs().max().item(), "max |sin diff|", (sS - sA).abs().max().item())
# phases depend only on ring position: check two visits to the same cell have same angle in Abs
pos = np.array([p[0] for p in env.visited_locations])
ang = torch.atan2(sA, cA)[0, :, 1::2, :]          # obs positions (H, T, nb)
same = [(i, j) for i in range(len(pos)) for j in range(i) if pos[i] == pos[j]][:2000]
err = max(((torch.remainder(ang[:, i] - ang[:, j] + math.pi, 2*math.pi) - math.pi).abs().max().item()) for i, j in same)
print("revisit pairs", len(same), "max circular angle mismatch (Abs) at same cell:", err)
# raw-Delta opposition score of the Abs wrap solution (what the prereg's secondary would read)
dp, dm = plus.reshape(-1), minus_abs.reshape(-1)
print("raw opposition ||D+ + D-|| / mean||D|| for Abs WRAP solution:", ((dp+dm).norm() / ((dp.norm()+dm.norm())/2)).item())
print("signed solution:", ((plus+minus_signed).norm()).item())
# the half-turn (parity) channel: Delta+ = Delta- , exactly what 'abs of a signed code' gives
