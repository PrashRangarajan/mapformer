"""Registered readouts for MAPPOPE_PAIR_PREREG.md (+ Amendment 1): paper torus, T=128; rank-2 arms seeds 10-25
(n=16), rank-4 arms seeds 10-17 (n=8). Inputs: MAPPOPE_PAIR_R2.json / MAPPOPE_PAIR_R4.json (eval_pair ->
eval_noise_refine, keys '0.0|<arm>|<T>' -> [[seed, acc, nll], ...]) and the checkpoints' per-epoch losses
(stats_core.classify_run: SOLVED = final-5% loss < 0.05). A contrast FIRES only on accuracy: perm p < .05 AND |d| >=
0.01; a Fisher-only difference in SOLVED is printed as convergence and never counts for a branch."""
import json
import numpy as np
import torch
from scipy import stats
from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/mappope_pair/p0"
R2, R4 = ["Vanilla", "MapPoPE-Pair", "MapPoPE-Flat"], ["Vanilla_r4", "MapPoPE-Pair_r4", "MapPoPE_r4"]
SEEDS = {a: list(range(10, 26)) for a in R2} | {a: list(range(10, 18)) for a in R4}
ARMS = ["Vanilla", "MapPoPE-Pair", "MapPoPE-Flat", "Vanilla_r4", "MapPoPE-Pair_r4", "MapPoPE_r4"]
LABEL = {"Vanilla": "MapWM r2 (32 angles)", "MapPoPE-Pair": "MapPoPE r2, 32 angles", "MapPoPE-Flat": "MapPoPE r2, 64 angles",
         "Vanilla_r4": "MapWM r4 (32)", "MapPoPE-Pair_r4": "MapPoPE r4, 32", "MapPoPE_r4": "MapPoPE r4, 64"}
MIN_D = 0.01                                            # materiality floor for an accuracy firing (prereg)
J = json.load(open(f"{REPO}/MAPPOPE_PAIR_R2.json")) | json.load(open(f"{REPO}/MAPPOPE_PAIR_R4.json"))
acc, nll = {}, {}
for a in ARMS:
    for T in (128, 512, 1024):
        rows = sorted(J[f"0.0|{a}|{T}"]); assert [r[0] for r in rows] == SEEDS[a], (a, T)
        acc[(a, T)] = np.array([r[1] for r in rows]); nll[(a, T)] = np.array([r[2] for r in rows])
cls = {a: [classify_run(torch.load(f"{R}/{a}_s{s}/{a}.pt", map_location="cpu", weights_only=False)["losses"])["registered"]
           for s in SEEDS[a]] for a in ARMS}
sol = {a: cls[a].count("SOLVED") for a in ARMS}
N = {a: len(SEEDS[a]) for a in ARMS}


def mde2(x, y):
    """Two-sample MDE (alpha .05 two-sided, power .8, df = nx + ny - 2, pooled sd)."""
    nx, ny = len(x), len(y); df = nx + ny - 2
    sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / df)
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df)) * sd * np.sqrt(1 / nx + 1 / ny)

print("== T=128 held-out accuracy (registered; floors: best n-gram 0.598, always-blank 0.507) | SOLVED | [T=512, T=1024: no verdict] ==")
for a in ARMS:
    x = acc[(a, 128)]
    print(f"  {LABEL[a]:24s} n={N[a]:2d} {x.mean():.4f} +/- {x.std(ddof=1):.4f}  {sol[a]}/{N[a]}  nll {nll[(a, 128)].mean():.4f}  [{acc[(a, 512)].mean():.4f}, {acc[(a, 1024)].mean():.4f}]  "
          + " ".join(f"{v:.3f}" for v in x))


def contrast(name, a, b):
    """b - a at T=128 -> (state, d): state 'POS' / 'NEG' (fires), 'CEILING' (both arms >= 0.999 on every seed: could not
    have gone the other way, rule 5), or 'NONE'. Fisher-only differences in SOLVED are printed, never counted."""
    x, y = acc[(a, 128)], acc[(b, 128)]; d = y.mean() - x.mean(); p = perm2_p(x, y)["p"]
    pf = fisher_solved(sol[a], N[a], sol[b], N[b]); m_ = max(mde2(x, y), MIN_D)
    if x.min() >= 0.999 and y.min() >= 0.999:
        st, tag = "CEILING", "CEILING (both arms >= 0.999 on every seed: undetermined)"
    elif p < 0.05 and abs(d) >= MIN_D:
        st = "POS" if d > 0 else "NEG"; tag = f"FIRES ({'+' if d > 0 else '-'})"
    else:
        st = "NONE"; tag = f"no (unmeasured below {m_:.4f})" + ("; SOLVED rate differs (convergence only)" if pf < 0.05 else "")
    print(f"  {name:46s} {d:+.4f}  perm p {p:.4f}  SOLVED {sol[b]}/{N[b]} vs {sol[a]}/{N[a]} Fisher p {pf:.4f}  -> {tag}")
    return st, d


print("\n== primary (rank 2, T=128, n=16 per arm) ==")
T_, dT = contrast("TOTAL   MapPoPE 64 - MapWM (paper2x2 replication)", "Vanilla", "MapPoPE-Flat")
S_, dS = contrast("SCORE   MapPoPE 32 - MapWM (score rule only)", "Vanilla", "MapPoPE-Pair")
C_, dC = contrast("COUNT   MapPoPE 64 - MapPoPE 32 (frequency count only)", "MapPoPE-Pair", "MapPoPE-Flat")
if T_ == "CEILING":
    v = "CEILING (MapWM r2 and MapPoPE r2 both at 1.000: nothing to decompose, undetermined)"
elif T_ != "POS":
    v = "NO EFFECT DETECTED (TOTAL did not fire positive; power ~0.95 for the paper2x2-sized effect at n=16)"
elif S_ == "POS" and C_ != "POS":
    v = "SCORE RULE" + (" (and the extra angles HURT)" if C_ == "NEG" else " (the frequency count does not detectably add)")
elif C_ == "POS" and S_ != "POS":
    v = "FREQUENCY COUNT" + (" (and PoPE's score alone HURTS)" if S_ == "NEG" else " (the score rule alone does not detectably help)")
elif S_ == "POS" and C_ == "POS":
    v = "BOTH (each change contributes detectably)"
else:
    v = "SPLIT UNMEASURED (TOTAL fires, neither part does; power for a 50/50 split is ~0.21)"
print(f"\n  REGISTERED: {v}")

print("\n== secondaries (no verdict) ==")
contrast("rank 4: MapPoPE 32 - MapWM", "Vanilla_r4", "MapPoPE-Pair_r4")
contrast("rank 4: MapPoPE 64 - MapPoPE 32", "MapPoPE-Pair_r4", "MapPoPE_r4")
for fam, (a2, a4) in {"MapWM": ("Vanilla", "Vanilla_r4"), "MapPoPE 32": ("MapPoPE-Pair", "MapPoPE-Pair_r4"),
                      "MapPoPE 64": ("MapPoPE-Flat", "MapPoPE_r4")}.items():
    g = {T: acc[(a4, T)].mean() - acc[(a2, T)].mean() for T in (128, 1024)}
    p = perm2_p(acc[(a2, 1024)], acc[(a4, 1024)])["p"]
    print(f"  rank-4 upgrade within {fam:10s} (r2 seeds 10-25 vs r4 seeds 10-17): T=128 {g[128]:+.4f}; T=1024 (OOD, rule 10) {g[1024]:+.4f} (perm p {p:.4f})")
for T in (512, 1024):
    for name, a, b in (("SCORE", "Vanilla", "MapPoPE-Pair"), ("COUNT", "MapPoPE-Pair", "MapPoPE-Flat")):
        x, y = acc[(a, T)], acc[(b, T)]
        print(f"  T={T} (OOD) {name}: {y.mean() - x.mean():+.4f} (perm p {perm2_p(x, y)['p']:.4f})")
for name, a, b in (("TOTAL", "Vanilla", "MapPoPE-Flat"), ("SCORE", "Vanilla", "MapPoPE-Pair"), ("COUNT", "MapPoPE-Pair", "MapPoPE-Flat")):
    x, y = nll[(a, 128)], nll[(b, 128)]
    print(f"  T=128 revisit NLL {name}: {y.mean() - x.mean():+.4f} (perm p {perm2_p(x, y)['p']:.4f}; lower is better)")
tails = [classify_run(torch.load(f"{R}/{a}_s{s}/{a}.pt", map_location="cpu", weights_only=False)["losses"])["tail"] for a in ARMS for s in SEEDS[a]]
print(f"  r(final loss, acc@128) over {len(tails)} runs: {np.corrcoef(tails, np.concatenate([acc[(a, 128)] for a in ARMS]))[0, 1]:+.3f}")
