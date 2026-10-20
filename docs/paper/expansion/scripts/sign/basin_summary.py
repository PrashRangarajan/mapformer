"""Summarise runs_basin/ (step_basin.py): per arm, final training loss and fresh-stream accuracy per seed; counts below a
loss of 0.05 (SOLVED, the house threshold) and with accuracy >= 0.95 (ESCAPED); exact tests for the main contrasts.
  python3 basin_summary.py > basin_summary_out.txt"""
import sys, json, glob, math
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact

D = Path(__file__).with_name("runs_basin")
ARMS = ["S_def", "S_abs", "S_abs_fz", "A_abs", "A_high", "A_seg_t0.75", "A_seg_t0.5", "A_seg_t0.25", "S_seg_t0.75"]


def perm_p(a, b, n=20000, rng=np.random.RandomState(0)):
    a, b = np.asarray(a), np.asarray(b); d0 = abs(a.mean() - b.mean()); x = np.concatenate([a, b]); k = 0
    for _ in range(n):
        rng.shuffle(x); k += abs(x[:len(a)].mean() - x[len(a):].mean()) >= d0 - 1e-12
    return (k + 1) / (n + 1)


def main():
    res = {}
    for arm in ARMS:
        rows = []
        for f in sorted(glob.glob(str(D / f"{arm}_s*.json"))):
            j = json.load(open(f))
            if j.get("skipped"):
                continue
            rows.append((j["args"]["seed"], j["final_loss"], j["acc"], j["acc0"], j["wrap_end"][1]))
        res[arm] = rows
        if not rows:
            print(f"{arm:12s} (no runs)"); continue
        fl = np.array([r[1] for r in rows]); ac = np.array([r[2] for r in rows])
        print(f"{arm:12s} n {len(rows)} | SOLVED (loss < 0.05) {int((fl < 0.05).sum())} | ESCAPED (acc >= 0.95) {int((ac >= 0.95).sum())}, acc >= 0.9 {int((ac >= 0.9).sum())} | "
              f"acc mean {ac.mean():.3f} (start {np.mean([r[3] for r in rows]):.3f}) | final loss median {np.median(fl):.3f} | "
              f"wrap share end {np.mean([r[4] for r in rows]):.2f}")
        print("     per seed acc/loss: " + " ".join(f"s{r[0]}:{r[2]:.2f}/{r[1]:.2f}" for r in rows))
    print("\n== contrasts (accuracy: permutation p; ESCAPED counts: Fisher p) ==")
    for a, b in (("S_abs", "A_abs"), ("S_abs", "S_abs_fz"), ("S_def", "S_abs"), ("A_high", "A_abs"), ("S_abs_fz", "A_abs"),
                 ("A_seg_t0.25", "A_abs"), ("S_seg_t0.75", "A_seg_t0.75")):
        if not res.get(a) or not res.get(b):
            continue
        xa = [r[2] for r in res[a]]; xb = [r[2] for r in res[b]]
        ea = sum(x >= 0.95 for x in xa); eb = sum(x >= 0.95 for x in xb)
        p = fisher_exact([[ea, len(xa) - ea], [eb, len(xb) - eb]])[1]
        print(f"{a} vs {b}: acc {np.mean(xa):.3f} vs {np.mean(xb):.3f} (d {np.mean(xa) - np.mean(xb):+.3f}, perm p {perm_p(xa, xb):.4f}); "
              f"ESCAPED {ea}/{len(xa)} vs {eb}/{len(xb)} (Fisher p {p:.4f})")


if __name__ == "__main__":
    main()
