"""Aggregate the Match-Query rank x loop 2x2.

The PRE-REGISTERED primary quantity is the MINIMUM and the seed sd, not the mean.
The mechanism under test says r=2 learns a skewed basis, which shows up as bimodal
seeds, so the prediction is that r=4 raises the FLOOR -- the same profile the loop
has (LOOP_HEADROOM: 8/8 seeds >= 0.77 against a 1-layer spread of 0.11-0.80). A
mean-only read would miss it, and reporting the mean alone after the fact would be
choosing the metric once the numbers are visible.
"""
import argparse, json, os
import numpy as np

ARMS = [("Vanilla", "r=2, no loop"), ("Vanilla_r4", "r=4, no loop"),
        ("Looped", "r=2, loop x4"), ("Looped_r4", "r=4, loop x4")]
MDE_K = 2.8
CHANCE = 0.0625


def load(runs, arm, TQ):
    out = {}
    for s in range(8):
        f = f"{runs}/{arm}/s{s}/{arm}_matchquery.json"
        if os.path.exists(f):
            d = json.load(open(f))
            if TQ in d:
                out[s] = d[TQ]["match_acc"]
    return out


def paired(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = np.array([b[s] - a[s] for s in common])
    sd = d.std(ddof=1)
    return dict(mean=d.mean(), sd=sd, mde=MDE_K * sd / np.sqrt(len(d)),
                pos=int((d > 0).sum()), n=len(d), d=d)


def fmt(p):
    if p is None:
        return "| — | — | — | — | underpowered |"
    v = "DETECTABLE" if abs(p["mean"]) > p["mde"] else "unmeasured"
    if v == "DETECTABLE" and p["mean"] < 0:
        v = "DETECTABLE NEGATIVE"
    return (f"| {p['mean']:+.3f} | {p['sd']:.3f} | {p['mde']:.3f} "
            f"| {p['pos']}/{p['n']} | {v} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--tq", default="256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {arm: load(a.runs_dir, arm, a.tq) for arm, _ in ARMS}
    o = ["# Match-Query: does r=4 remove the same failure mode as the loop?", "",
         f"128^2, TQ={a.tq}, chance {CHANCE}. All four arms in one batch, "
         "LOOP_HEADROOM's recipe. The loop is free on both rows and rank costs "
         "exactly +384 on both, so the interaction is parameter-matched.", "",
         "| arm | params | mean | sd | **min** | per-seed |",
         "|---|---|---|---|---|---|"]
    P = {"Vanilla": "204,373", "Vanilla_r4": "204,757",
         "Looped": "204,373", "Looped_r4": "204,757"}
    for arm, lbl in ARMS:
        v = np.array([A[arm][s] for s in sorted(A[arm])])
        if not len(v):
            o.append(f"| {lbl} | {P[arm]} | — | — | — | — |"); continue
        o.append(f"| {lbl} | {P[arm]} | **{v.mean():.3f}** | {v.std(ddof=1):.3f} "
                 f"| **{v.min():.3f}** | " + " ".join(f"{x:.2f}" for x in v) + " |")
    o += ["", "## Effects", "",
          "| contrast | delta | sd | MDE | seeds + | verdict |",
          "|---|---|---|---|---|---|"]
    rows = [("rank main effect, no loop  (r4 - r2)", "Vanilla", "Vanilla_r4"),
            ("rank main effect, with loop (r4 - r2)", "Looped", "Looped_r4"),
            ("loop main effect, r=2", "Vanilla", "Looped"),
            ("loop main effect, r=4", "Vanilla_r4", "Looped_r4")]
    for lbl, x, y in rows:
        o.append(f"| {lbl} " + fmt(paired(A[x], A[y])))

    # interaction, paired per seed
    common = sorted(set(A["Vanilla"]) & set(A["Vanilla_r4"]) &
                    set(A["Looped"]) & set(A["Looped_r4"]))
    if len(common) >= 2:
        inter = np.array([(A["Looped_r4"][s] - A["Looped"][s]) -
                          (A["Vanilla_r4"][s] - A["Vanilla"][s]) for s in common])
        sd = inter.std(ddof=1); mde = MDE_K * sd / np.sqrt(len(inter))
        v = "DETECTABLE" if abs(inter.mean()) > mde else "unmeasured"
        o.append(f"| **interaction (rank x loop)** | {inter.mean():+.3f} | {sd:.3f} "
                 f"| {mde:.3f} | {int((inter>0).sum())}/{len(inter)} | {v} |")

    o += ["", "## The pre-registered reading", "",
          "The prediction was that r=4 **compresses variance more than it lifts "
          "the mean** -- raising the floor, the way the loop does. Read the `min` "
          "and `sd` columns first; the mean is secondary here by prior "
          "commitment, not by hindsight.", "",
          "- Negative interaction with both mains positive -> the two remove the "
          "**same** failure mode, and $384$ parameters is the cheaper route.",
          "- Interaction near zero with both mains holding -> independent fixes, "
          "and `r=4, loop x4` should be the best arm measured on this task.",
          "- No rank effect at all -> the skewed-basin account does not transfer "
          "off the torus, and the D x r optimisation half is torus-specific."]
    open(a.out, "w").write("\n".join(o) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
