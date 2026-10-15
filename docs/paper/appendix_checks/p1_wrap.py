"""P1 (wrapping representability).
(a) Construction, m <= r: a random rank-4 signed MapFormerWM (every token passes through W_out W_in; W_in made to annihilate
    the non-stepping embeddings so exactly m tokens step) is rewritten as a |.| model (Abs_r4) with W_in' sending the m
    stepping embeddings to e_1..e_m and the rest to 0, omega' = max(|omega|, 1), W_out'[:, j] = remainder(omega S_j, 2 pi) / omega'.
    float64 logits compared, m = 1..4; ring tokens from the registered environment.
(b) r = 1, two stepping tokens with opposite-signed latents (v1 > 0 > v2), two channels: the compatibility condition for a
    rank-1 non-negative representation mod 2 pi is AFFINE in (alpha_1, alpha_2) for each integer quadruple, and it vanishes
    identically only when the forced b2 = v2 / v1 < 0 (sympy).
(c) Approximate representation (Kronecker): with b rationally independent, a rank-1 non-negative code matches any angle table
    mod 2 pi to error eps, at angle size growing like eps^-(m-1) (m stepping tokens); measured on random tables, 64 channels.
"""
import math
import numpy as np
import sympy as sp
import torch
from common import ring_batch
from mapformer.model_sign import MapFormerWM_Signed_r4, MapFormerWM_Abs_r4

out = []
P = lambda s: (print(s), out.append(s))
toks, _, _, _ = ring_batch(32, seed=7)
torch.manual_seed(1)
for m_step in (1, 2, 3, 4):
    step_ids = [0, 1, 5, 9][:m_step]
    sg = MapFormerWM_Signed_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32).double().eval()
    ab = MapFormerWM_Abs_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32).double().eval()
    with torch.no_grad():
        sg.action_to_lie.w_out.weight.mul_(20.0)                                           # angles of order several rad
        E = sg.token_emb.weight                                                            # (V, d), rank 19
        non = [i for i in range(19) if i not in step_ids]
        Q, _ = torch.linalg.qr(E[non].T)                                                    # basis of the non-stepping span
        W = sg.action_to_lie.w_in.weight; W -= (W @ Q) @ Q.T                                # annihilate non-stepping tokens
        S = sg.step_table() if hasattr(sg, "step_table") else None
        St = sg.action_to_lie(E[None])[0].reshape(19, -1)                                  # (V, C) signed steps
        om = sg.path_integrator.omega.reshape(-1)
        ab.load_state_dict(sg.state_dict())
        L = torch.zeros(19, 4, dtype=torch.float64)
        for j, i in enumerate(step_ids):
            L[i, j] = 1.0
        ab.action_to_lie.w_in.weight.copy_((torch.linalg.pinv(E) @ L).T)                    # E W_in'^T = L
        om2 = om.abs().clamp_min(1.0)
        Wout = torch.zeros(64, 4, dtype=torch.float64)
        for j, i in enumerate(step_ids):
            Wout[:, j] = torch.remainder(om * St[i], 2 * math.pi) / om2
        ab.action_to_lie.w_out.weight.copy_(Wout)
        ab.path_integrator.omega.copy_(om2.reshape(2, 32))
        Sa = ab.action_to_lie(E[None])[0].reshape(19, -1)
        nonzero_signed = int((St.abs().max(1).values > 1e-9).sum())
        ang_err = torch.angle(torch.exp(1j * (om2 * Sa - om * St))).abs().max()
        neg_rows = int(((St[step_ids] > 0).any(0) & (St[step_ids] < 0).any(0)).sum()) if m_step > 1 else 0
        P(f"(a) m = {m_step} stepping tokens ({nonzero_signed} non-zero step rows in the signed table; {neg_rows}/64 channels "
          f"with opposite-signed steps): built |.| table min {Sa.min():.2e} (>= 0), max angle error mod 2 pi {ang_err:.2e}, "
          f"max |logit diff| {(ab(toks) - sg(toks)).abs().max():.2e} (logit scale {sg(toks).abs().max():.1f})")

# (b) r = 1, symbolic
a1, a2, v1, v2 = sp.symbols("alpha1 alpha2 v1 v2", real=True)
k1, k2, l1, l2 = sp.symbols("k1 k2 l1 l2", integer=True)
b_ch1 = (a1 * v2 + 2 * sp.pi * k2) / (a1 * v1 + 2 * sp.pi * k1)
b_ch2 = (a2 * v2 + 2 * sp.pi * l2) / (a2 * v1 + 2 * sp.pi * l1)
num = sp.expand((a1 * v2 + 2 * sp.pi * k2) * (a2 * v1 + 2 * sp.pi * l1) - (a2 * v2 + 2 * sp.pi * l2) * (a1 * v1 + 2 * sp.pi * k1))
poly = sp.Poly(num, a1, a2)
P(f"(b) r = 1: numerator of b2(channel 1) - b2(channel 2) = {sp.factor(num)}; monomials in alpha: {poly.monoms()} (no alpha1*alpha2 term: affine)")
co = poly.as_dict()
P(f"    coefficients: alpha1 -> {sp.factor(co.get((1, 0), 0))}, alpha2 -> {sp.factor(co.get((0, 1), 0))}, 1 -> {sp.factor(co.get((0, 0), 0))}")
# identically zero => k2 v1 = k1 v2 and l2 v1 = l1 v2 => b2 = v2/v1 in each channel
bsub = sp.simplify(b_ch1.subs(k2, k1 * v2 / v1))
P(f"    vanishing alpha-coefficients force k2 = k1 v2/v1 (and l2 = l1 v2/v1); then b2 = {bsub} (< 0 when v1 v2 < 0): excluded by b2 >= 0")
# numeric: no exact solution for random alpha, best mismatch over |k| <= K shrinks (approximation, not exactness)
rng = np.random.RandomState(3)
for trial in range(3):
    al = rng.normal(0, 2, 2); vv = (1.0, -0.7)
    best = {}
    for K in (5, 20, 80):
        ks = np.arange(-K, K + 1)
        bs = []
        for a in al:
            K1, K2 = np.meshgrid(ks, ks, indexing="ij")
            den = a * vv[0] + 2 * math.pi * K1
            b = (a * vv[1] + 2 * math.pi * K2) / np.where(np.abs(den) > 1e-12, den, np.nan)
            bs.append(np.sort(b[np.isfinite(b) & (b >= 0)].ravel()))
        x, y = bs
        idx = np.clip(np.searchsorted(y, x), 1, len(y) - 1)
        best[K] = float(np.min(np.minimum(np.abs(x - y[idx - 1]), np.abs(x - y[idx]))))
    P(f"    alpha = {al.round(3)}: min |b2(ch1) - b2(ch2)| over integer wraps |k| <= 5 / 20 / 80: "
      + " / ".join(f"{best[K]:.2e}" for K in best) + "  (never 0; shrinks with K -> approximation at large angles)")

# (c) Kronecker approximation at rank 1
C = 64
irr = np.array([1.0, (math.sqrt(5) - 1) / 2 + 1, math.sqrt(2), math.sqrt(3) + 0.1])
for m in (2, 3, 4):
    Th = rng.uniform(-math.pi, math.pi, (C, m)); b = irr[:m]
    row = []
    for K in (10, 100, 1000, 10000):
        ks = np.arange(-K, K + 1)
        a = (Th[:, :1] % (2 * math.pi)) + 2 * math.pi * ks[None, :]                   # a_c b_1 = Theta_c1 mod 2 pi exactly
        e = np.abs(np.angle(np.exp(1j * (a[:, :, None] * b[None, None, 1:] - Th[:, None, 1:])))).max(-1)  # (C, K)
        row.append(f"K={K}: max_c min_k err {e.min(1).max():.4f} rad, |angle| <= {2 * math.pi * K * b.max():.0f}")
    P(f"(c) m = {m}: " + "; ".join(row))
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")
