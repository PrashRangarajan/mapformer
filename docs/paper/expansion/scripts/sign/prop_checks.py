"""SEARCH_THEORY: numerical checks of Propositions S2-S4 (CPU, seconds). Readouts W (64 x 4) and omega are the default
initialisations of MECHANISM seeds 800-807 (rebuilt), so the checks use the real frequency ladder.
 (1) S3(b): at fixed (W, omega), least-squares latents for 2r and 2r + 1 channels to wrap exactly (random channel sets, sign
     regions and wrap counts n in {1, 2, 3}); the residual is ~0 for 2r channels and bounded away from 0 for 2r + 1.
 (2) S2(c) / S4 (+1/-1): gradient descent on the quadratic clock surrogate R = 1/2 sum_c q_c k_c^2 over the two latents,
     signed vs |.| (true |.|, so sign regions switch). Signed: u -> 0 with the map part unchanged; |.|: latents contract.
 (3) S4 (+1/+2): z_1 fixed (unit norm), descend R = 1/2 sum_c (a_c(+2) - 2 a_c(+1))^2 over z_2 from random starts; share of
     starts that reach a collective zero (z_2 = 2 z_1 or, under |.|, z_2 = -2 z_1) vs the share stuck elsewhere.
  python3 prop_checks.py > prop_checks_out.txt
"""
import sys, math
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from latent_geometry import build

torch.set_num_threads(2)


def readouts():
    out = []
    for seed in range(800, 808):
        m = build(True, seed)
        with torch.no_grad():
            W = m.action_to_lie.w_out.weight.double().numpy(); om = m.path_integrator.omega.double().reshape(-1).numpy()
            z = (m.token_emb.weight.double()[m.act] @ m.action_to_lie.w_in.weight.double().T).numpy()
        out.append((W, om, z))
    return out


def check1(R, rng):
    r = R[0][0].shape[1]
    res = {2 * r: [], 2 * r + 1: []}
    for W, om, _ in R:
        for _ in range(500):
            for size in res:
                S = rng.choice(len(om), size, replace=False)
                ex, ey = rng.choice([-1, 1], size), rng.choice([-1, 1], size)
                n = rng.randint(1, 4, size)
                N = np.concatenate([(om[S] * ex)[:, None] * W[S], (om[S] * ey)[:, None] * W[S]], 1)
                zeta, *_ = np.linalg.lstsq(N, 2 * math.pi * n, rcond=None)
                res[size].append(np.max(np.abs(N @ zeta - 2 * math.pi * n)))
    for size, v in res.items():
        v = np.array(v)
        print(f"(1) {size} channels: max |angle-sum error| after least squares: median {np.median(v):.3g}, min {v.min():.3g} "
              f"(n {len(v)})")


def check2(R):
    for W, om, z in R[:4]:
        Wt = torch.tensor(W); omt = torch.tensor(om)
        for mode in ("signed", "abs"):
            zx = torch.tensor(z[0], requires_grad=True); zy = torch.tensor(z[1], requires_grad=True)
            d0 = ((zy - zx) / 2).detach().clone(); n0 = (zx.norm() + zy.norm()).item()
            opt = torch.optim.SGD([zx, zy], lr=0.01)
            for _ in range(8000):
                sx, sy = Wt @ zx, Wt @ zy
                if mode == "abs":
                    sx, sy = sx.abs(), sy.abs()
                k = omt * (sx + sy)
                loss = 0.5 * (k ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                d = (zy - zx) / 2; u = zx + zy
                sx, sy = Wt @ zx, Wt @ zy
                if mode == "abs":
                    sx, sy = sx.abs(), sy.abs()
                mapn = (omt * sy).abs().mean().item()
            print(f"(2) {mode:6s}: clock surrogate {loss.item():.2e}; |u| / (|z_x| + |z_y|) {u.norm().item() / (zx.norm() + zy.norm()).item():.3f}; "
                  f"latent norm end / start {(zx.norm() + zy.norm()).item() / n0:.3f}; map part change |d - d0| / |d0| "
                  f"{(d - d0).norm().item() / d0.norm().item():.3f}; mean |map angle| {mapn:.3f}")


def check3(R, rng):
    for mode in ("signed", "abs"):
        hit, stuck = 0, 0; mirror = 0
        for W, om, z in R:
            Wt = torch.tensor(W); omt = torch.tensor(om)
            z1 = torch.tensor(z[0]); z1 = z1 / z1.norm()
            for _ in range(10):
                z2 = torch.tensor(rng.randn(W.shape[1]) * 1.0, requires_grad=True)
                opt = torch.optim.Adam([z2], lr=0.01)
                for _ in range(2000):
                    s1, s2 = Wt @ z1, Wt @ z2
                    if mode == "abs":
                        s1, s2 = s1.abs(), s2.abs()
                    loss = 0.5 * ((omt * (s2 - 2 * s1)) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    e_plus = (z2 - 2 * z1).norm().item() / 2; e_minus = (z2 + 2 * z1).norm().item() / 2
                if min(e_plus, e_minus if mode == "abs" else 9) < 0.05:
                    hit += 1; mirror += int(mode == "abs" and e_minus < e_plus)
                else:
                    stuck += 1
        print(f"(3) {mode:6s}: reached a collective zero {hit}/{hit + stuck} (via the mirror copy z_2 = -2 z_1: {mirror})", flush=True)


def main():
    rng = np.random.RandomState(0); torch.manual_seed(0)
    R = readouts()
    check1(R, rng)
    check2(R)
    check3(R, rng)


if __name__ == "__main__":
    main()
