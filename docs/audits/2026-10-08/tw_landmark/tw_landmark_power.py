"""Power for TW_LANDMARK_PREREG.md from text-world per-seed data (CPU). Uses the registered decision rules of
analyze_tw_landmark (contrast(): exact permutation p < .05 and |d| >= threshold; PERSISTS needs both 95% permutation
CIs above -0.10).

Per-seed data (path models without names, the proxy for P2 at r = 0 -- no 2-layer path model has been trained on
words; the pilot replaces this proxy):
  reliance and names-stripped accuracy of the 16 committed one-layer path runs (text world s0-7, TW_NORMSTEP MapWM
  s10-17) on THIS batch's eval stream (reliance_validate_out.json);
  index RoPE 2 layers: the 8 committed text-world runs' registered accuracy (eval.json, T=1024).
Scenarios for O (P2 at r* vs P2 at r = 0), each seed of the r* cell resampled from the proxy, then:
  NULL        unchanged                         -> false 'down' firings; PERSISTS rate
  LOST        every seed loses the map          (reliance ~ U(0, 0.02), stripped accuracy ~ U(0.50, 0.61))
  HALF        each seed loses it w.p. 0.5
  SHIFT10     every seed's reliance and stripped accuracy shifted down by 0.10
  (failing seeds of the proxy are kept: the r = 0 cell is itself bimodal)
For I0 (P2 - R2 at r = 0): path accuracy resampled from the proxy vs RoPE 2L resampled from the text world.
1000 simulations per cell (200 at n = 8); n per cell in {4, 5, 6, 8}.
"""
import json

import numpy as np

from mapformer.analyze_tw_landmark import contrast, O_MIN, ACC_MIN, EQUIV

V = json.load(open("/home/prashr/mapformer/docs/audits/2026-10-08/tw_landmark/reliance_validate_out.json"))
P = [d for k in ("tw_path1L", "ns_MapWM") for d in V[k].values()]
REL = np.array([d["rel"] for d in P]); ACC = np.array([d["acc"] for d in P])
ROPE2 = np.array([json.load(open(f"/home/prashr/mapformer/runs/textworld/p0/RoPE_L2_s{s}/eval.json"))["eval"]["1024"]["acc"]
                  for s in range(8)])
print(f"proxy path (n=16): reliance {REL.mean():.4f} +/- {REL.std(ddof=1):.4f}, stripped acc {ACC.mean():.4f} +/- "
      f"{ACC.std(ddof=1):.4f}; RoPE 2L (n=8) {ROPE2.mean():.4f} +/- {ROPE2.std(ddof=1):.4f}")
rng = np.random.default_rng(0)
NS = 1000


def draw(n):
    i = rng.integers(0, len(REL), n)
    return REL[i].copy(), ACC[i].copy()


def scen(name, n):
    xr, xa = draw(n); yr, ya = draw(n)
    if name == "LOST":
        yr = rng.uniform(0, 0.02, n); ya = rng.uniform(0.50, 0.61, n)
    elif name == "HALF":
        k = rng.random(n) < 0.5
        yr[k] = rng.uniform(0, 0.02, k.sum()); ya[k] = rng.uniform(0.50, 0.61, k.sum())
    elif name == "SHIFT10":
        yr -= 0.10; ya -= 0.10
    return xr, xa, yr, ya


print("\nO (per cell n): P(reliance down fires) | P(stripped acc down fires) | P(both: OVERSHADOW-type) | P(PERSISTS)")
for n in (4, 5, 6, 8):
    for name in ("NULL", "LOST", "HALF", "SHIFT10"):
        f_r = f_a = both = pers = 0
        for _ in range(NS if n <= 6 else 200):
            xr, xa, yr, ya = scen(name, n)
            cr, ca = contrast(xr, yr, O_MIN), contrast(xa, ya, O_MIN)
            dr, da = cr["fire"] and cr["d"] < 0, ca["fire"] and ca["d"] < 0
            f_r += dr; f_a += da; both += dr and da
            if not (cr["fire"] or ca["fire"]) and all(c["lo"] is not None and c["lo"] >= -EQUIV for c in (cr, ca)):
                pers += 1
        k = NS if n <= 6 else 200
        print(f"  n={n} {name:8s}: {f_r / k:.3f} | {f_a / k:.3f} | {both / k:.3f} | {pers / k:.3f}", flush=True)

print("\nI0 (P2 - R2 at r = 0, accuracy): P(PATH WINS)")
for n in (4, 5, 6, 8):
    w = 0
    for _ in range(NS if n <= 6 else 200):
        x = ROPE2[rng.integers(0, 8, n)]; y = ACC[rng.integers(0, 16, n)]
        c = contrast(x, y, ACC_MIN); w += c["fire"] and c["d"] > 0
    print(f"  n={n}: {w / (NS if n <= 6 else 200):.3f}")
