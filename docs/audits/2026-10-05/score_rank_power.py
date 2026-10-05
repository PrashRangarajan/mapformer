"""Power for SCORE_RANK_PREREG.md's primary (MapPoPE-Pair r2 vs MapWM r2 at T=1024), computed before the batch.

(1) SOLVED: exact power of the two-sided Fisher test (alpha .05) for n per arm in {8, 12, 16}, MapWM r2's true solve
    rate in {0, 0.05, 0.125, 0.25} (observed: 0/8 in RANK_MI, seeds 0-7) and MapPoPE-Pair r2's in {0.25 .. 1}.
(2) Accuracy firing (perm p < .05 AND d >= MIN_D), Monte Carlo: MapWM r2 seeds drawn from RANK_MI's 8 stored T=1024
    accuracies (Vanilla); MapPoPE-Pair r2 seeds are 'solved' with probability q (accuracy drawn from Vanilla_r4mi's 8
    stored values) and otherwise drawn from the MapWM r2 values. Permutation p from 2000 relabellings.
(3) The joint RESCUE condition (both fire) under the same model.
Output: score_rank_power_out.txt next to this file."""
import json, os
import numpy as np
from scipy import stats

REPO = "/home/prashr/mapformer"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_rank_power_out.txt")
J = json.load(open(f"{REPO}/RANK_MI.json"))
A = np.array([x[1] for x in sorted(J["0.0|Vanilla|1024"])]); C = np.array([x[1] for x in sorted(J["0.0|Vanilla_r4mi|1024"])])
MIN_D = 0.02
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def fisher_power(n, p0, p1, alpha=0.05):
    k = np.arange(n + 1); pa, pb = stats.binom.pmf(k, n, p0), stats.binom.pmf(k, n, p1)
    rej = np.array([[stats.fisher_exact([[j, n - j], [i, n - i]])[1] < alpha and j > i for j in k] for i in k])
    return float((pa[:, None] * pb[None, :] * rej).sum())


log(f"stored inputs: MapWM r2 T=1024 acc {np.round(A, 4).tolist()} (mean {A.mean():.4f}); r4mi {np.round(C, 4).tolist()}")
log("\n(1) Fisher power (two-sided .05, PoPE > MapWM) for SOLVED")
log("   n   p_WM  " + "  ".join(f"q={q:.2f}" for q in (0.25, 0.375, 0.5, 0.625, 0.75, 1.0)))
for n in (8, 12, 16):
    for p0 in (0.0, 0.05, 0.125, 0.25):
        log(f"  {n:2d}  {p0:.3f} " + "  ".join(f"{fisher_power(n, p0, q):6.3f}" for q in (0.25, 0.375, 0.5, 0.625, 0.75, 1.0)))
log("   min solved count that fires vs 0/n: " + ", ".join(
    f"n={n}: {min(j for j in range(n + 1) if stats.fisher_exact([[j, n - j], [0, n]])[1] < 0.05)}/{n}" for n in (8, 12, 16)))

rng = np.random.default_rng(0)


def perm_p(x, y, nperm=2000):
    z = np.concatenate([x, y]); n = len(x); obs = abs(y.mean() - x.mean())
    idx = np.argsort(rng.random((nperm, len(z))), axis=1)
    zz = z[idx]; d = np.abs(zz[:, n:].mean(1) - zz[:, :n].mean(1))
    return (np.sum(d >= obs - 1e-12) + 1) / (nperm + 1)


log(f"\n(2)/(3) accuracy firing (perm p < .05 AND d >= {MIN_D}) and joint RESCUE (Fisher AND accuracy), MapWM r2 rate 0 "
    f"(its seeds resampled from RANK_MI), 1000 simulations per cell")
log("   n    q   P(acc fires)  P(Fisher fires)  P(both)")
for n in (8, 12, 16):
    for q in (0.25, 0.5, 0.75, 1.0):
        fa = ff = fb = 0
        for _ in range(1000):
            x = rng.choice(A, n); sol = rng.random(n) < q
            y = np.where(sol, rng.choice(C, n), rng.choice(A, n))
            a_ = perm_p(x, y) < 0.05 and y.mean() - x.mean() >= MIN_D
            f_ = stats.fisher_exact([[sol.sum(), n - sol.sum()], [0, n]])[1] < 0.05
            fa += a_; ff += f_; fb += a_ and f_
        log(f"  {n:2d}  {q:.2f}   {fa / 1000:.3f}         {ff / 1000:.3f}           {fb / 1000:.3f}")
open(OUT, "w").write("\n".join(lines) + "\n")
