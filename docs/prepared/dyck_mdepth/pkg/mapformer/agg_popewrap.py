"""Does PoPE's gain scale with how much the phase wraps?

THE PREDICTION (mapformer_math.tex sec 3.1). PoPE's two changes are one trade:
deleting the intrinsic phase is what buys one frequency per ELEMENT instead of one
per PAIR, and that ceiling is forced -- a rotation acts on a 2-plane and has one
angle, so d_head coordinates admit exactly d_head/2 rotary frequencies.

So the doubled count should pay exactly when the frequency spectrum is stretched
thin. omega is initialised over [2*pi/N, 2*pi] at grid size N with a FIXED 32
blocks, so the range spans log2(N) octaves and the log-spacing between adjacent
frequencies coarsens as N grows:

    grid  16   32   64   128
    oct    4    5    6     7

PRE-REGISTERED: the MapPoPE - Vanilla gain increases monotonically with grid size,
and increases with evaluation length at fixed grid (path length is the other
wrapping knob). Falsified if flat, or decreasing.

CEILING CAVEAT, stated before the numbers exist. A null at small grid is only
informative if BOTH arms are off ceiling there -- conditioning on convergence can
select into a ceiling that shows ~0 by construction (rule 11). Absolute means are
printed beside every contrast so this is checkable rather than assumed. Grid 8 is
excluded outright: it yields 82 scored positions per trajectory against ~29
elsewhere and is known solvable without position at all (ALIASING_CONTROLLED.md).
"""
import argparse, json, os
import numpy as np

MDE_K = 2.8


def load(runs, grid, arm, T):
    f = os.path.join(runs, f"g{grid}", "acc.json")
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))
    key = f"0.0|{arm}|{T}"
    return {int(s): a for s, a, _ in d.get(key, [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--grids", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--arms", nargs="+", default=["Vanilla", "MapPoPE"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 1024])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, wide = a.arms

    o = ["# Does PoPE's gain scale with how much the phase wraps?", "",
         "`MapPoPE` minus `Vanilla`: same path-integrated phase, PoPE's magnitude and",
         "one frequency per element instead of one per pair. +320 parameters at every",
         "grid size. 8 seeds, one batch.", "",
         "omega spans [2pi/N, 2pi] over a fixed 32 blocks, so the spectrum covers",
         "log2(N) octaves and thins as N grows -- the quantity the prediction is about.", ""]
    for T in a.lengths:
        o += [f"## T = {T}", "",
              "| grid | octaves | " + f"`{base}`" + " | " + f"`{wide}`"
              + " | gain | sd | MDE | seeds + | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for g in a.grids:
            A, B = load(a.runs_dir, g, base, T), load(a.runs_dir, g, wide, T)
            c = sorted(set(A) & set(B))
            if len(c) < 2:
                o.append(f"| {g} | {int(np.log2(g))} | — | — | — | — | — | — | — |")
                continue
            va = np.array([A[s] for s in c]); vb = np.array([B[s] for s in c])
            d = vb - va; sd = d.std(ddof=1); m = MDE_K * sd / np.sqrt(len(d))
            v = "DETECTABLE" if abs(d.mean()) > m else "unmeasured"
            if v == "DETECTABLE" and d.mean() < 0:
                v = "DETECTABLE NEGATIVE"
            ceil = "  *(ceiling)*" if min(va.mean(), vb.mean()) > 0.99 else ""
            o.append(f"| {g} | {int(np.log2(g))} | {va.mean():.3f} ± {va.std(ddof=1):.3f} "
                     f"| {vb.mean():.3f} ± {vb.std(ddof=1):.3f} | **{d.mean():+.3f}** "
                     f"| {sd:.3f} | {m:.3f} | {int((d>0).sum())}/{len(d)} | {v}{ceil} |")
        o.append("")
    o += ["## Reading it", "",
          "The prediction is a **trend**, not a single contrast: the gain should rise",
          "with octaves and with T. A row marked *(ceiling)* carries no information",
          "either way -- both arms above 0.99 leaves nothing for a mechanism to buy."]
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
