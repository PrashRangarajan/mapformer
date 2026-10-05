"""Power of GAIN_GRAIN_PREREG.md's registered decision rules (analyze_gain_grain's own functions, N_MC lowered to 2000
for speed), by resampling stored per-seed (T=128 accuracy, SOLVED) pairs with replacement:
  P-like  MAPPOPE_PAIR's MapPoPE-Pair r2, seeds 10-25 (16/16 SOLVED)          MAPPOPE_PAIR_R2.json + checkpoints
  W-like  MAPPOPE_PAIR's MapWM r2, seeds 10-25 (10/16 SOLVED)                 same
  E-like  em_fig4's MapEM r2 (VanillaEM), seeds 0-7, same recipe except --data-workers 3 (5/8 SOLVED);
          evaluated for this script: gain_grain_emfig4_T128.json (eval_noise_refine, T=128, 100 walks, env seed 10000)
Scenarios: a coarser gain that behaves like P (null of equality), like W (the size of MapWM's deficit), a per-seed
50/50 mixture of W and P (half the deficit), or like E; the sign arm like P, a 50/50 mixture of E and P, or like E.
Output: gain_grain_power_out.txt."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
import mapformer.analyze_gain_grain as A
from mapformer.stats_core import classify_run

A.N_MC = 2000
REPO = "/home/prashr/mapformer"


def pairs(json_path, key, ck_fmt, seeds):
    J = json.load(open(json_path)); rows = dict((r[0], r[1]) for r in J[key])
    sol = [classify_run(torch.load(ck_fmt.format(s=s), map_location="cpu", weights_only=False)["losses"])["registered"] == "SOLVED"
           for s in seeds]
    return np.array([rows[s] for s in seeds]), np.array(sol)


Pp = pairs(f"{REPO}/MAPPOPE_PAIR_R2.json", "0.0|MapPoPE-Pair|128", REPO + "/runs/mappope_pair/p0/MapPoPE-Pair_s{s}/MapPoPE-Pair.pt", range(10, 26))
Wp = pairs(f"{REPO}/MAPPOPE_PAIR_R2.json", "0.0|Vanilla|128", REPO + "/runs/mappope_pair/p0/Vanilla_s{s}/Vanilla.pt", range(10, 26))
Ep = pairs(f"{REPO}/docs/audits/2026-10-05/gain_grain_emfig4_T128.json", "0.0|VanillaEM|128",
           REPO + "/runs/em_fig4/p0/VanillaEM_s{s}/VanillaEM.pt", range(8))
for nm, (a, s) in (("P-like (MapPoPE-Pair)", Pp), ("W-like (MapWM)", Wp), ("E-like (MapEM)", Ep)):
    print(f"{nm:24s} n={len(a):2d} acc {a.mean():.4f} +/- {a.std(ddof=1):.4f}  SOLVED {s.sum()}/{len(s)}")
rng = np.random.default_rng(0)


def draw(src, n):
    i = rng.integers(0, len(src[0]), n); return src[0][i], src[1][i]


def mix(s1, s2, n):
    a1, b1 = draw(s1, n); a2, b2 = draw(s2, n); m = rng.random(n) < 0.5
    return np.where(m, a1, a2), np.where(m, b1, b2)


K = 400
print(f"\n== GRANULARITY: x (coarser gain) vs P, granularity_state; {K} resamples per cell ==")
print(f"{'n':>3s} {'x behaves like':28s} {'AS GOOD':>8s} {'WORSE':>8s} {'BETTER':>8s} {'UNDET':>8s} {'CONFLICT':>8s}")
for n in (12, 16, 20):
    for nm, gen in (("P (equal: null)", lambda: draw(Pp, n)), ("W (MapWM's deficit)", lambda: draw(Wp, n)),
                    ("50/50 W/P (half the deficit)", lambda: mix(Wp, Pp, n)), ("E (MapEM)", lambda: draw(Ep, n))):
        c = {}
        for _ in range(K):
            ra, rs = draw(Pp, n); xa, xs = gen()
            lab, _ = A.granularity_state(ra, int(rs.sum()), xa, int(xs.sum()), n, n); c[lab] = c.get(lab, 0) + 1
        print(f"{n:3d} {nm:28s} " + " ".join(f"{c.get(k, 0) / K:8.2f}" for k in ("AS GOOD", "WORSE", "BETTER", "UNDETERMINED", "CONFLICT")))

print(f"\n== SIGN: N vs E, sign_state; E drawn E-like; {K} resamples per cell ==")
print(f"{'n':>3s} {'N behaves like':28s} {'NN BETTER':>10s} {'NN WORSE':>10s} {'NO DIFF':>10s} {'CEILING':>8s} {'CONFLICT':>8s}")
for n in (12, 16, 20):
    for nm, gen in (("P (non-negative fixes it)", lambda: draw(Pp, n)), ("50/50 E/P (half)", lambda: mix(Ep, Pp, n)),
                    ("E (no effect: null)", lambda: draw(Ep, n))):
        c = {}
        for _ in range(K):
            ea, es = draw(Ep, n); xa, xs = gen()
            lab, _ = A.sign_state(ea, int(es.sum()), xa, int(xs.sum()), n, n); lab = lab.split(" (")[0]
            c[lab] = c.get(lab, 0) + 1
        print(f"{n:3d} {nm:28s} " + " ".join(f"{c.get(k, 0) / K:10.2f}" for k in ("NON-NEGATIVE BETTER", "NON-NEGATIVE WORSE", "NO DIFFERENCE DETECTED"))
              + " ".join(f"{c.get(k, 0) / K:9.2f}" for k in ("CEILING", "CONFLICT")))

print(f"\n== HEADROOM control (P vs W, headroom_state), {K} resamples ==")
for n in (12, 16, 20):
    ok = 0
    for _ in range(K):
        wa, ws = draw(Wp, n); pa, ps = draw(Pp, n)
        ok += A.headroom_state(wa, int(ws.sum()), pa, int(ps.sum()), n, n)[0] == "OK"
    print(f"  n={n}: P(headroom control passes | W-like vs P-like) {ok / K:.2f}")
