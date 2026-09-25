"""Readouts for RANK_MATCHED_PREREG.md (Amendment 2), from committed JSONs and checkpoints.

    python3 -m mapformer.analyze_rank_matched                 # 300-epoch batch
    python3 -m mapformer.analyze_rank_matched --tag _e900     # 900-epoch batch
    python3 -m mapformer.analyze_rank_matched --tag _e900 --classify-only --seeds 0 1   # pilot
"""
import argparse, json
import numpy as np
import torch

from mapformer.stats_core import classify_run, fisher_solved, perm2_ci, perm2_p

REPO = "/home/prashr/mapformer"
V2, V4 = "Vanilla", "Vanilla_r4"
STRATA = ("all", "plain_lag<128", "plain_lag>=128", "wrap")
SOLVED_LOSS, STALL_TOL = 0.05, 0.05


def classify(losses):
    """SOLVED / STALLED / DESCENDING from per-epoch loss (Amendment 2): the REGISTERED
    label, from stats_core.classify_run (which also flags a RISING run; see main())."""
    r = classify_run(losses, SOLVED_LOSS, STALL_TOL)
    return r["registered"], r["tail"], r["ratio"]


def paired(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return d.mean(), 2.8 * d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())


def perm(a, b, shift=0.0):
    """Exact two-sample permutation p (two-sided) for mean(b) - mean(a) - shift.
    Delegates to stats_core.perm2_p (vectorised, audit 2026-09-24): the same arithmetic on
    the same C(N, n) relabellings as the per-combination loop it replaced, so the p-values
    and the committed *_ANALYSIS.txt files reproduce byte for byte."""
    return perm2_p(a, b, shift)["p"]


def perm_ci(a, b, lo=-0.6, hi=0.6, step=0.005):
    """95% test-inversion CI on the registered fixed grid [lo, hi] (kept so committed
    readouts reproduce). Returns (lo, hi, clipped): `clipped` means an END of the grid was
    accepted, so the true interval extends past it -- main() then says so."""
    r = perm2_ci(a, b, level=0.95, step=step, grid=(lo, hi))
    return r["lo"], r["hi"], r["clipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--classify-only", action="store_true",
                    help="pilot mode: training-loss classes only, reads no evaluation")
    a = ap.parse_args()
    S = a.seeds
    runs = f"{REPO}/runs/rank_matched{a.tag}/p0"

    print(f"== run classes (Amendment 2), runs/rank_matched{a.tag} ==")
    C, loss = {}, {}
    for v in (V2, V4):
        for s in S:
            b = torch.load(f"{runs}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
            assert b["config"].get("n_steps") == 1024, b["config"]
            C[(v, s)] = classify(b["losses"]); loss[(v, s)] = float(b["losses"][-1])
            cls, tail, ratio = C[(v, s)]
            print(f"  {v:11s} s{s}  epochs {len(b['losses'])}  tail loss {tail:.4f}  "
                  f"last10%/prev10% {'' if ratio is None else f'{ratio:.3f}'}  {cls}")
    for v in (V2, V4):
        n = {k: sum(C[(v, s)][0] == k for s in S) for k in ("SOLVED", "STALLED", "DESCENDING")}
        print(f"  {v:11s} {n}")
    # Added 2026-09-24 (audit B5): the registered rule files a run whose loss went UP by >5%
    # under DESCENDING, which counts toward "unreadable". Printed as an extra line only.
    rising = [(v, s, C[(v, s)][2]) for v in (V2, V4) for s in S
              if C[(v, s)][0] == "DESCENDING" and C[(v, s)][2] > 1]
    if rising:
        print("  note: loss RISING (final 10% above the 10% before; the registered rule files "
              "these under DESCENDING): " + ", ".join(f"{v} s{s} ({r:.3f})" for v, s, r in rising))
    if a.classify_only:
        desc = [k for k, c in C.items() if c[0] == "DESCENDING"]
        print("PILOT RULE:", "no run DESCENDING -> launch seeds 2-7 at this budget" if not desc
              else f"{len(desc)} run(s) DESCENDING {desc} -> do NOT launch; decide next budget with the user")
        return

    M = json.load(open(f"{REPO}/RANK_MATCHED{a.tag}.json"))
    ST = json.load(open(f"{REPO}/RANK_MATCHED{a.tag}_STRATA.json"))
    acc = lambda v, T: [dict((x[0], x) for x in M[f"0.0|{v}|{T}"])[s][1] for s in S]
    nll = lambda v, T: [dict((x[0], x) for x in M[f"0.0|{v}|{T}"])[s][2] for s in S]

    a2, a4 = acc(V2, 1024), acc(V4, 1024)
    print("\n== primary: T=1024 overall accuracy ==")
    d, mde, npos = paired(a2, a4)
    p = perm(a2, a4); lo, hi, clipped = perm_ci(a2, a4)
    print(f"  r2 {np.mean(a2):.3f}  r4 {np.mean(a4):.3f}  d {d:+.3f}  paired MDE {mde:.3f} ({npos}/{len(S)})"
          f"  exact permutation p {p:.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if clipped:
        print("  WARNING: the CI reaches an end of its fixed [-0.6, +0.6] grid -- the interval is "
              "CLIPPED there (stats_core.perm2_ci(grid=None) gives the unclipped one)")
    s2 = sum(C[(V2, s)][0] == "SOLVED" for s in S); s4 = sum(C[(V4, s)][0] == "SOLVED" for s in S)
    pf = fisher_solved(s2, len(S), s4, len(S))
    print(f"== co-primary: SOLVED  r2 {s2}/{len(S)}  r4 {s4}/{len(S)}  Fisher p {pf:.4f}")
    sa2 = [x for x, s in zip(a2, S) if C[(V2, s)][0] == "SOLVED"]
    sa4 = [x for x, s in zip(a4, S) if C[(V4, s)][0] == "SOLVED"]
    print(f"  accuracy among SOLVED: r2 {np.mean(sa2) if sa2 else float('nan'):.3f} (n={len(sa2)})"
          f"  r4 {np.mean(sa4) if sa4 else float('nan'):.3f} (n={len(sa4)})")
    ndesc = {v: sum(C[(v, s)][0] == "DESCENDING" for s in S) for v in (V2, V4)}
    ci_excl = lo is not None and not (lo <= 0.085 <= hi)
    if max(ndesc.values()) > 2:
        verdict = f"UNREADABLE ({ndesc} DESCENDING; more than 2 in an arm)"
    elif (p < 0.05 and d > 0) or (pf < 0.05 and s4 > s2):
        verdict = "R2 -- learnability deficit at r=2"
    elif (p < 0.05 and d < 0) or (pf < 0.05 and s2 > s4):
        verdict = "R3 -- reversal"
    elif ci_excl:
        verdict = "R1 -- no difference at matched length; +0.085 excluded"
    else:
        verdict = "UNMEASURED -- no difference found, +0.085 not excluded"
    print(f"== BRANCH: {verdict}")

    print("\n== secondary: other lengths (paired MDE for continuity) ==")
    for T in (512, 1024, 2048):
        for name, f in (("acc", acc), ("NLL", nll)):
            x2, x4 = f(V2, T), f(V4, T); d, mde, npos = paired(x2, x4)
            print(f"  {name} T={T:4d}  r2 {np.mean(x2):.3f}  r4 {np.mean(x4):.3f}  d {d:+.3f}  "
                  f"MDE {mde:.3f}  {npos}/{len(S)} r4>r2  perm p {perm(x2, x4):.4f}")

    print("\n== secondary: strata (accuracy and NLL; floor = best constant) ==")
    for T in (1024, 2048):
        for k in STRATA:
            g = lambda v, q: [ST[f"{v}|{s}|{T}"][k][q] for s in S]
            fl = g(V2, "floor")
            below = sum(x < f for x, f in zip(g(V2, "acc"), fl)), sum(x < f for x, f in zip(g(V4, "acc"), fl))
            da, ma, _ = paired(g(V2, "acc"), g(V4, "acc")); dn, mn, _ = paired(g(V2, "nll"), g(V4, "nll"))
            print(f"  T={T} {k:15s} floor {np.mean(fl):.3f}  acc r2 {np.mean(g(V2, 'acc')):.3f} r4 "
                  f"{np.mean(g(V4, 'acc')):.3f} d {da:+.3f} (MDE {ma:.3f})  NLL d {dn:+.3f} (MDE {mn:.3f})"
                  f"  runs below floor r2 {below[0]}/{len(S)} r4 {below[1]}/{len(S)}")
    print("\n== exploratory: within-run stratum profile at T=1024 (stratum acc - own gap<128 acc) ==")
    for k in ("plain_lag>=128", "wrap"):
        pr = lambda v: [ST[f"{v}|{s}|1024"][k]["acc"] - ST[f"{v}|{s}|1024"]["plain_lag<128"]["acc"] for s in S]
        d, mde, npos = paired(pr(V2), pr(V4))
        print(f"  {k:15s} r2 {np.mean(pr(V2)):+.3f}  r4 {np.mean(pr(V4)):+.3f}  d {d:+.3f}  MDE {mde:.3f}  {npos}/{len(S)}")

    print("\n== losses (descriptive: at matched length loss-matching cannot separate speed from solution) ==")
    l2 = [loss[(V2, s)] for s in S]; l4 = [loss[(V4, s)] for s in S]
    print(f"  final loss r2 {min(l2):.4f}-{max(l2):.4f}  r4 {min(l4):.4f}-{max(l4):.4f}  "
          f"r4 lower on {sum(y < x for x, y in zip(l2, l4))}/{len(S)} seeds")
    x = np.log(np.array([*l2, *l4])); y = np.array([*a2, *a4]); arm = np.r_[np.zeros(len(S)), np.ones(len(S))]
    for nm, sl in (("r2", slice(0, len(S))), ("r4", slice(len(S), None))):
        print(f"  within {nm}: r(log loss, acc) {np.corrcoef(x[sl], y[sl])[0, 1]:+.3f}  "
              f"slope {np.polyfit(x[sl], y[sl], 1)[0]:+.3f}")
    X = np.c_[np.ones_like(x), arm, x]; beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  ANCOVA acc ~ arm + log loss: arm coefficient {beta[1]:+.3f}, loss slope {beta[2]:+.3f}")


if __name__ == "__main__":
    main()
