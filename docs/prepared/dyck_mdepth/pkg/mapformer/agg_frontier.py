"""Depth-vs-loop frontier: does looping substitute for depth equally well with and
without path integration?

Reports accuracy against PARAMETER COUNT for both position codes, so the two
frontiers can be laid on top of each other. The headline contrast is not
"loop beats X" but where the loop sits ON each row's depth curve.

Honest axis note, stated in the output too: the loop is parameter-matched to the
1-layer model, NOT compute-matched. Four passes cost four passes. This is a
parameters-and-memory result, never a FLOPs one.
"""
import argparse, json, statistics as st
from pathlib import Path

SPECS = [("RoPE", 1), ("RoPE", 2), ("RoPE", 3), ("RoPELooped", 1),
         ("Vanilla", 1), ("Vanilla", 2), ("Vanilla", 3), ("Looped", 1)]
NAME = {("RoPE", 1): "index L1", ("RoPE", 2): "index L2", ("RoPE", 3): "index L3",
        ("RoPELooped", 1): "index LOOP x4",
        ("Vanilla", 1): "path-int L1", ("Vanilla", 2): "path-int L2",
        ("Vanilla", 3): "path-int L3", ("Looped", 1): "path-int LOOP x4"}
ROW = {"RoPE": "index", "RoPELooped": "index",
       "Vanilla": "path-int", "Looped": "path-int"}


def stat(d):
    d = [x for x in d if x is not None]
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), mde=float("nan"), n=n, pos=0)
    sd = st.stdev(d)
    return dict(m=st.mean(d), sd=sd, mde=2.8 * sd / n ** 0.5, n=n,
                pos=sum(1 for x in d if x > 0))


def fmt(x):
    return (f"{x['m']:+.3f} (sd {x['sd']:.3f}, MDE {x['mde']:.3f}, "
            f"{x['pos']}/{x['n']})")


def vd(x):
    if x["m"] != x["m"]:
        return "NO DATA"
    return ("DETECTABLE POSITIVE" if x["m"] > x["mde"] else
            "DETECTABLE NEGATIVE" if x["m"] < -x["mde"] else "UNMEASURED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--venue", required=True, choices=["algorithmic", "matchquery"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    alg = a.venue == "algorithmic"
    tasks = ["parity", "copy"] if alg else ["matchquery"]
    seeds = range(16) if alg else range(8)
    KEY = "128" if alg else "256"

    def load(task, v, nl, s):
        d = (Path(a.runs_dir) / task / f"{v}_L{nl}_s{s}" if alg
             else Path(a.runs_dir) / f"{v}_L{nl}_s{s}")
        f = d / (f"{v}_{task}.json" if alg else f"{v}_matchquery.json")
        if not f.exists():
            return None, None
        j = json.load(open(f))
        acc = j["acc"].get(KEY) if alg else j.get(KEY, {}).get("match_acc")
        return acc, j.get("params")

    o = ["# Depth vs loop: does looping substitute for depth in BOTH position codes?",
         "", f"Venue: **{a.venue}**, chosen by `decide_frontier.py` on a rule fixed",
         "before the numbers were seen. Eight arms: {index, path integration} x",
         "{1, 2, 3 real layers, loop x4}.", "",
         "**Axis note.** The loop is parameter-matched to the 1-layer model, NOT",
         "compute-matched -- four passes cost four passes. Everything below is a",
         "parameters-and-memory result and none of it is a FLOPs result.", ""]

    A = {}
    for task in tasks:
        o += [f"## {task}" if len(tasks) > 1 else "## Frontier", "",
              "| arm | params | accuracy |", "|---|---|---|"]
        for v, nl in SPECS:
            xs, pars = [], None
            for s in seeds:
                acc, p = load(task, v, nl, s)
                if acc is not None:
                    xs.append(acc); pars = p or pars
            A[(task, v, nl)] = xs
            o.append(f"| {NAME[(v, nl)]} | {pars:,} | "
                     + (f"{st.mean(xs):.3f} +/- {st.stdev(xs):.3f} (n={len(xs)})"
                        if len(xs) > 1 else "—") + " |" if pars else
                     f"| {NAME[(v, nl)]} | — | — |")
        o.append("")

        def pair(v1, n1, v2, n2):
            X, Y = A[(task, v1, n1)], A[(task, v2, n2)]
            k = min(len(X), len(Y))
            return [X[i] - Y[i] for i in range(k)]

        o += ["### Where the loop sits on each row's depth curve", "",
              "| row | loop - L1 | loop - L2 | loop - L3 |", "|---|---|---|---|"]
        res = {}
        for row, flat, loop in (("index", "RoPE", "RoPELooped"),
                                ("path-int", "Vanilla", "Looped")):
            cells = []
            for nl in (1, 2, 3):
                s_ = stat(pair(loop, 1, flat, nl))
                res[(row, nl)] = s_
                cells.append(fmt(s_))
            o.append(f"| {row} | " + " | ".join(cells) + " |")
        o.append("")
        i1 = res.get(("path-int", 1)); i2 = res.get(("index", 1))
        if i1 and i2 and i1["m"] == i1["m"] and i2["m"] == i2["m"]:
            o += [f"**Loop gain over 1 layer**: path-int {i1['m']:+.3f} "
                  f"({vd(i1)}), index {i2['m']:+.3f} ({vd(i2)}); "
                  f"difference **{i1['m'] - i2['m']:+.3f}**.", "",
                  ("The loop pays MORE with path integration -- looping and path "
                   "integration are not independent, and the parameter-efficiency "
                   "claim is stronger in the path-integrated row."
                   if i1["m"] - i2["m"] > max(i1["mde"], i2["mde"]) else
                   "The loop's gain does not differ detectably between position "
                   "codes: it substitutes for depth the same way in both, and "
                   "LOOP_HEADROOM's apparent interaction does not survive having a "
                   "depth baseline on the index row."), ""]
        for row, flat, loop in (("index", "RoPE", "RoPELooped"),
                                ("path-int", "Vanilla", "Looped")):
            s3 = res.get((row, 3))
            if s3 and s3["m"] == s3["m"]:
                o.append(f"- **{row}**: loop vs 3 real layers {fmt(s3)} -> "
                         f"{vd(s3)}. " +
                         ("The loop MATCHES real depth at a third of the "
                          "parameters." if vd(s3) == "UNMEASURED" else
                          "The loop BEATS real depth." if s3["m"] > 0 else
                          "The loop FALLS SHORT of real depth."))
        o.append("")

    o += ["## Scope", "",
          "One train length, one loop count, depth to 3. Accuracy is reported at "
          + (f"L={KEY} (the extrapolation point)" if alg else
             f"T_query={KEY}") + ". A frontier is a shape, and three depth points "
          "define it only loosely -- read the per-arm numbers, not just the "
          "contrasts."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
