"""Does hierarchy help parity? A test of the sufficient-statistic principle.

Parity is a TREE REDUCTION -- the partial parity of a pooled pair is exactly a
sufficient statistic, with no loss. If the principle is predictive, hierarchy wins
here even though it loses on this project's exact-recall navigation tasks. The
sharper half is LENGTH EXTRAPOLATION: a tree argument predicts a flatter decay from
L=16 to L=256, not just a higher score at the training length.
"""
import argparse, json, statistics as st
from pathlib import Path

ROWS = [("index", "RoPE", "PlainHourglass"),
        ("path-int", "HourglassFlat3", "Hourglass_k2")]
LENS = [16, 32, 64, 128, 256]
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
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(16)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {(v, L): [acc(a.runs_dir, v, s, L) for s in a.seeds]
         for _r, f, h in ROWS for v in (f, h) for L in LENS}

    o = ["# Does hierarchy help parity?", "",
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
    P = {"RoPE": "595,586", "PlainHourglass": "595,586",
         "HourglassFlat3": "596,034", "Hourglass_k2": "596,034"}
    LBL = {"RoPE": "index, FLAT (3 layers)", "PlainHourglass": "index, HIERARCHICAL",
           "HourglassFlat3": "path-int, FLAT (3 blocks)",
           "Hourglass_k2": "path-int, HIERARCHICAL"}
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
    hits = [r for r, _f, _h in ROWS if vd(S[(r, 128)]) == "DETECTABLE POSITIVE"]
    miss = [r for r, _f, _h in ROWS if vd(S[(r, 128)]) == "DETECTABLE NEGATIVE"]
    for r, _f, _h in ROWS:
        o.append(f"- **{r}** at L=128: {fmt(S[(r, 128)])} -> {vd(S[(r, 128)])}")
    o.append("")
    if len(hits) == 2:
        o += ["**The sufficient-statistic principle is PREDICTIVE.** Hierarchy wins "
              "in both position codes on the one task where the pooled summary is "
              "provably lossless, having lost on this project's exact-recall "
              "navigation benchmarks. The navigation losses were about the TASK, not "
              "about the mechanism.", ""]
    elif not hits and not miss:
        o += ["**Unmeasured in both rows.** Hierarchy neither helps nor hurts on a "
              "task where the summary is provably lossless and the algorithm is "
              "literally a tree reduction. That is a hard case for the principle: it "
              "predicted a win here specifically. Report the MDE and treat the "
              "principle as unsupported rather than refuted (rule 11).", ""]
    elif miss and not hits:
        o += ["**The principle FAILS.** Hierarchy loses on the one task where the "
              "pooled summary is exactly sufficient. It should be retired rather "
              "than narrowed -- 'helps when the summary is sufficient' has now been "
              "tested at its most favourable case and lost.", ""]
    else:
        o += [f"**Row-dependent**: helps {hits or 'neither'}, hurts "
              f"{miss or 'neither'}. Hierarchy interacts with the POSITION CODE, "
              f"which nothing in this project predicts and which needs its own "
              f"mechanism test before it is reported as a finding.", ""]
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
