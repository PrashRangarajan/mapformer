"""Power of GAIN_PHASE's registered decisions (analyze_gain_phase.analyse, run unchanged on resampled batches).

Library of real runs (LEAK, runs/leak/p0, seeds 0-7; readouts from gain_phase_leak_validate.json, losses from the
checkpoints): W-like = a LEAK MapWM run (acc 0.984-0.992, DESCENDING, L_ms ~+0.01, S_id ~0.0045); N-like = a LEAK
NormStep run (acc >= 0.9995, SOLVED, L_ms ~0, S_id ~0.001). A simulated arm draws n runs with replacement from its
scenario's pool. GainRaw / GainPhase behaviour is unknown, so scenarios span it:
  A predicted        G W-like (gain score neither helps nor removes the leak), GP N-like
  B tolerated        G = N-like accuracy / losses / L_ms but W-like S_id (gain score tolerates the identity step)
  C unlearned        G N-like in everything
  D half             G W-like with accuracy +0.005 and L_ms halved
  E GP fails 1/8     GP: each run W-like with probability 1/8, else N-like
  F GP fails 1/4     the same with 1/4
  G GP 3x faster     GP N-like with the loss curve compressed 3x in epochs (l'[e] = l[min(3e, 899)])
  H null speed       GP N-like (= N): false-firing rate of D5 and D4
  I GP stalls 1/8    GP: each run, with probability 1/8, a hard failure (accuracy 0.60, final loss 0.5: not SOLVED,
                     no leak), else N-like
Amendment 1 (convergence gate on D3; readouts now include theta reliance, distortion share, field shift; acc_rescored =
acc in the library, so no re-score flag fires here):
  J gain unconverged  G and GP untrained-like (acc 0.14, reliance 0, S_id 1.0, L_ms 0 -- what untrained models read --
                      loss still descending at ~0.6): D3 must read UNMEASURED, never TOLERATES / ALSO REMOVES
  K GainRaw unconv.   G untrained-like, GP N-like
Output: gain_phase_power_out.txt"""
import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
import mapformer.analyze_gain_phase as A

REPO = "/home/prashr/mapformer"
A.PRINT_CI = False; A.N_MC = 20_000
V = json.load(open(f"{REPO}/docs/audits/2026-10-06/gain_phase_leak_validate.json"))
LIB = {}
for arm, key in (("MapWM", "W"), ("NormStep", "N")):
    LIB[key] = []
    for r in V[arm]:
        l = torch.load(f"{REPO}/runs/leak/p0/{arm}_s{r['seed']}/{arm}.pt", map_location="cpu", weights_only=False)["losses"]
        LIB[key].append({"acc": r["acc"], "acc_rescored": r["acc"], "L_ms": r["L_ms"], "S_id": r["S_id"],
                         "reliance": r["reliance"], "resid": r["resid"], "shift_cells": r["shift_cells"], "losses": list(l)})


def hybrid(n_run, w_run):
    return dict(n_run, S_id=w_run["S_id"], resid=w_run["resid"], shift_cells=w_run["shift_cells"])


def half(w_run):
    return dict(w_run, acc=min(1.0, w_run["acc"] + 0.005), L_ms=w_run["L_ms"] / 2)


def fast(n_run):
    l = n_run["losses"]; return dict(n_run, losses=[l[min(3 * e, len(l) - 1)] for e in range(len(l))])


def draw(rng, pool, n):
    return [pool[i] for i in rng.integers(0, len(pool), n)]


def gp_fail(rng, q, n):
    return [LIB["W"][rng.integers(8)] if rng.random() < q else LIB["N"][rng.integers(8)] for _ in range(n)]


SCEN = {
    "A predicted": lambda rng, n: {"G": draw(rng, LIB["W"], n), "GP": draw(rng, LIB["N"], n)},
    "B tolerated": lambda rng, n: {"G": [hybrid(a, b) for a, b in zip(draw(rng, LIB["N"], n), draw(rng, LIB["W"], n))],
                                   "GP": draw(rng, LIB["N"], n)},
    "C unlearned": lambda rng, n: {"G": draw(rng, LIB["N"], n), "GP": draw(rng, LIB["N"], n)},
    "D half": lambda rng, n: {"G": [half(r) for r in draw(rng, LIB["W"], n)], "GP": draw(rng, LIB["N"], n)},
    "E GP fails 1/8": lambda rng, n: {"G": draw(rng, LIB["W"], n), "GP": gp_fail(rng, 1 / 8, n)},
    "F GP fails 1/4": lambda rng, n: {"G": draw(rng, LIB["W"], n), "GP": gp_fail(rng, 1 / 4, n)},
    "G GP 3x faster": lambda rng, n: {"G": draw(rng, LIB["W"], n), "GP": [fast(r) for r in draw(rng, LIB["N"], n)]},
    "H null speed": lambda rng, n: {"G": draw(rng, LIB["W"], n), "GP": draw(rng, LIB["N"], n)},
}
STALL = {"acc": 0.60, "acc_rescored": 0.60, "L_ms": 0.0, "S_id": 0.001, "reliance": 0.5, "resid": 0.5, "shift_cells": 0.001,
         "losses": list(np.linspace(6.8, 0.5, 900))}
UNC = {"acc": 0.14, "acc_rescored": 0.14, "L_ms": 0.0, "S_id": 1.0, "reliance": 0.0, "resid": 0.6, "shift_cells": 0.5,
       "losses": list(6.8 * np.exp(-np.arange(900) / 300) + 0.3)}
SCEN["J gain unconverged"] = lambda rng, n: {"G": [UNC] * n, "GP": [UNC] * n}
SCEN["K GainRaw unconverged"] = lambda rng, n: {"G": [UNC] * n, "GP": draw(rng, LIB["N"], n)}
SCEN["I GP stalls 1/8"] = lambda rng, n: {"G": draw(rng, LIB["W"], n),
                                          "GP": [STALL if rng.random() < 1 / 8 else LIB["N"][rng.integers(8)] for _ in range(n)]}
KEYS = ("D1r", "D1", "D2_raw", "D2_norm", "D3_headline", "D4_headline", "D4_vs_N", "D5_norm")


def simulate(name, n, reps, seed=0):
    rng = np.random.default_rng(seed); tally = {k: {} for k in KEYS}
    for _ in range(reps):
        arms = {"W": draw(rng, LIB["W"], n), "N": draw(rng, LIB["N"], n), **SCEN[name](rng, n)}
        D = {(A.ARMS[i], s): arms[k][s] for i, k in enumerate(("W", "N", "G", "GP")) for s in range(n)}
        v = A.analyse(D, seeds=list(range(n)), out=lambda *a: None)
        for k in KEYS:
            lab = re.split(r":| \(| --", v[k])[0]
            tally[k][lab] = tally[k].get(lab, 0) + 1
    return tally


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for n in (8, 10, 12):
        print(f"\n######## n = {n} per arm, {reps} resampled batches per scenario")
        for name in SCEN:
            t = simulate(name, n, reps)
            print(f"== {name}")
            for k in KEYS:
                print(f"   {k:12s} " + "; ".join(f"{lab} {c / reps:.2f}" for lab, c in sorted(t[k].items(), key=lambda kv: -kv[1])))
