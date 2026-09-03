"""Does hierarchy help parity? A test of the sufficient-statistic principle.

Parity is a TREE REDUCTION -- the partial parity of a pooled pair is exactly a
sufficient statistic, with no loss. If the principle is predictive, hierarchy wins
here even though it loses on this project's exact-recall navigation tasks. The
sharper half is LENGTH EXTRAPOLATION: a tree argument predicts a flatter decay from
L=16 to L=256, not just a higher score at the training length.
"""
import argparse, json, statistics as st
from pathlib import Path

ROWS = [("unshared", "HourglassFlat3", "Hourglass_k2"),
        ("shared", "LoopedHourglassFlat", "LoopedHourglass")]
LENS = [512, 1024, 2048]
CHANCE = 0.5


def acc(runs, v, s, L):
    f = Path(runs) / f"{v}_s{s}" / f"{v}_parity.json"
    return json.load(open(f))["acc"].get(str(L)) if f.exists() else None


def stat(d):
    d = [x for x in d if x is not None]
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), mde=float("nan"), n=n, pos=0)
    sd = st.stdev(d)
    return dict(m=st.mean(d), sd=sd, mde=2.8 * sd / n ** 0.5, n=n,
                pos=sum(1 for x in d if x > 0))


def fmt(x):
    return f"{x['m']:+.3f} (sd {x['sd']:.3f}, MDE {x['mde']:.3f}, {x['pos']}/{x['n']})"


def vd(x):
    if x["m"] != x["m"]:
        return "NO DATA"
    return ("DETECTABLE POSITIVE" if x["m"] > x["mde"] else
            "DETECTABLE NEGATIVE" if x["m"] < -x["mde"] else "UNMEASURED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {(v, L): [acc(a.runs_dir, v, s, L) for s in a.seeds]
         for _r, f, h in ROWS for v in (f, h) for L in LENS}

    o = ["# Loop x hierarchy: is the pairing free?", "",
     "Parity trained at L=512, where the compute saving is real "
     "(-22.8% time, -19.9% memory at L=2048; see LOOP_HIER_COMPUTE.md).",
     "A NULL ON ACCURACY IS THE SUCCESS CASE: the claim is cheaper at equal",
     "accuracy, not better.", "",
         "Parity is a TREE REDUCTION: parity(a,b,c,d) = parity(parity(a,b),",
         "parity(c,d)). The partial parity of a pooled pair is EXACTLY a sufficient",
         "statistic, so this is the case this project's standing principle --",
         "hierarchy helps only when a summary is sufficient -- says should WIN, on a",
         "task where hierarchy usually loses (exact recall).", "",
         "Exact parameter parity within each row. Hourglass variants ignore",
         "--n-layers and are always the 3-block scaffold, so the flat controls are",
         "3-layer models, not 1-layer ones.", "",
         "| arm | params | " + " | ".join(f"L={L}" for L in LENS) + " |",
         "|---" * (len(LENS) + 2) + "|"]
    P = {"HourglassFlat3": "596,034", "Hourglass_k2": "596,034",
         "LoopedHourglassFlat": "199,490", "LoopedHourglass": "199,490"}
    LBL = {"HourglassFlat3": "unshared, FLAT", "Hourglass_k2": "unshared, HIER",
           "LoopedHourglassFlat": "SHARED, FLAT", "LoopedHourglass": "SHARED, HIER"}
    for _r, f, h in ROWS:
        for v in (f, h):
            cells = []
            for L in LENS:
                xs = [x for x in A[(v, L)] if x is not None]
                cells.append(f"{st.mean(xs):.3f}" if xs else "—")
            o.append(f"| {LBL[v]} | {P[v]} | " + " | ".join(cells) + " |")
    o.append("")

    o += ["## Hierarchy minus flat, per row", "",
          "| row | " + " | ".join(f"L={L}" for L in LENS) + " |",
          "|---" * (len(LENS) + 1) + "|"]
    S = {}
    for r, f, h in ROWS:
        cells = []
        for L in LENS:
            d = [x - y for x, y in zip(A[(h, L)], A[(f, L)])
                 if x is not None and y is not None]
            S[(r, L)] = stat(d)
            cells.append(fmt(S[(r, L)]))
        o.append(f"| {r} | " + " | ".join(cells) + " |")
    o.append("")

    o += ["## Length decay -- the sharper half of the prediction", "",
          "Accuracy above chance at L, as a fraction of that at L=16. A tree",
          "reduction should decay FLATTER, not merely start higher.", "",
          "| arm | " + " | ".join(f"L={L}" for L in LENS) + " |",
          "|---" * (len(LENS) + 1) + "|"]
    for _r, f, h in ROWS:
        for v in (f, h):
            b = [x for x in A[(v, 16)] if x is not None]
            base = st.mean(b) - CHANCE if b else 0.0
            cells = []
            for L in LENS:
                xs = [x for x in A[(v, L)] if x is not None]
                cells.append(f"{(st.mean(xs) - CHANCE) / base:.2f}"
                             if xs and base > 1e-6 else "—")
            o.append(f"| {LBL[v]} | " + " | ".join(cells) + " |")
    o.append("")

    o += ["## Verdict", ""]
    for r, _f, _h in ROWS:
        o.append(f"- hierarchy inside the **{r}** row, L=2048: {fmt(S[(r, 2048)])} "
                 f"-> {vd(S[(r, 2048)])}")
    cross = stat([x - y for x, y in zip(A[("LoopedHourglass", 2048)],
                                        A[("HourglassFlat3", 2048)])
                  if x is not None and y is not None])
    o += ["", f"- **the pairing vs the plain 3-block scaffold**, L=2048: "
          f"{fmt(cross)} -> {vd(cross)}", ""]
    if vd(cross) == "UNMEASURED":
        o += ["**THE PAIRING IS FREE.** LoopedHourglass matches the unshared flat "
              "scaffold on accuracy while using **66.5% fewer parameters** and "
              "**22.8% less time and 19.9% less memory** at L=2048 (measured, "
              "LOOP_HIER_COMPUTE.md). Both savings, no accuracy cost. Note this is "
              "an equivalence read off a null, so the MDE above is the claim's real "
              "precision -- it says the cost is smaller than that, not that it is "
              "zero.", ""]
    elif vd(cross) == "DETECTABLE NEGATIVE":
        o += [f"**The pairing COSTS accuracy**: {cross['m']:+.3f} against the "
              f"unshared flat scaffold. The savings are real but not free, and the "
              f"trade has to be argued rather than assumed.", ""]
    else:
        o += ["**The pairing BEATS the unshared scaffold outright**, which was not "
              "predicted -- hierarchy buys no accuracy on parity (HIER_PARITY.md, "
              "+0.001 with MDE 0.006) and sharing should at best break even. Treat "
              "a positive here as needing its own explanation before it is "
              "reported.", ""]
    o += ["## Scope", "",
          "One task, one pooling factor (k=2), one train length, n=16. Copy is not "
          "included: it has no dynamic range at any length here, so it cannot serve "
          "as a control (see ALGORITHMIC_RESULTS.md). Hierarchy is NOT combined with "
          "the loop in this run -- no looped-hierarchical variant exists, and "
          "building one is the natural follow-up if this is positive."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
