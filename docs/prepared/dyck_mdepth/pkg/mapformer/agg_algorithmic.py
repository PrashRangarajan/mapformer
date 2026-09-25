"""Aggregate the algorithmic-task runs, with the three hypotheses fixed in advance.

H1  path integration helps PARITY much more than COPY -> the effect tracks the
    task's structure rather than our pipeline.
H2  the loop improves LENGTH GENERALIZATION (L=128 relative to L=16), the
    literature's actual claim.
H3  LoopedSampled beats fixed Looped at long L -- the adaptivity claim, and the one
    place our navigation data already agrees with the field.

Verdicts compare against the MDE (2.8*sd/sqrt(n)), never against zero.
"""
import argparse, json, statistics as st
from pathlib import Path

ARMS = ["RoPE", "Vanilla", "RoPELooped", "Looped", "LoopedSampled"]
LABEL = {"RoPE": "index, flat", "Vanilla": "path-int, flat",
         "RoPELooped": "index, loop x4", "Looped": "path-int, loop x4",
         "LoopedSampled": "path-int, loop SAMPLED"}
TASKS = ["parity", "copy"]
LENS = [16, 32, 64, 128, 256]


def load(runs_dir, task, v, s):
    f = Path(runs_dir) / task / f"{v}_s{s}" / f"{v}_{task}.json"
    return json.load(open(f)) if f.exists() else None


def stat(d):
    d = [x for x in d if x is not None]
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), t=float("nan"),
                    mde=float("nan"), n=n, pos=0)
    sd = st.stdev(d)
    return dict(m=st.mean(d), sd=sd, t=st.mean(d) / (sd / n ** 0.5) if sd else 0.0,
                mde=2.8 * sd / n ** 0.5, n=n, pos=sum(1 for x in d if x > 0))


def fmt(x):
    return (f"{x['m']:+.3f} (sd {x['sd']:.3f}, t {x['t']:+.2f}, "
            f"MDE {x['mde']:.3f}, {x['pos']}/{x['n']})")


def vd(x):
    if x["m"] != x["m"]:
        return "NO DATA"
    return ("DETECTABLE POSITIVE" if x["m"] > x["mde"] else
            "DETECTABLE NEGATIVE" if x["m"] < -x["mde"] else "UNMEASURED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = {(t, v, s): load(a.runs_dir, t, v, s)
         for t in TASKS for v in ARMS for s in a.seeds}
    acc = {(t, v, L): [D[(t, v, s)]["acc"][str(L)] if D[(t, v, s)] else None
                       for s in a.seeds]
           for t in TASKS for v in ARMS for L in LENS}
    chance = {t: next((D[(t, v, s)]["chance"] for v in ARMS for s in a.seeds
                       if D[(t, v, s)]), None) for t in TASKS}
    par = {v: next((D[(t, v, s)]["params"] for t in TASKS for s in a.seeds
                    if D[(t, v, s)]), None) for v in ARMS}

    o = ["# Path integration on the looped literature's own tasks", "",
         "Trained at L=16, evaluated to L=256. 5 arms x 2 tasks x 8 seeds, one",
         "batch. Gated before training: every trivial baseline sits at chance at",
         "every length, worst excess +0.0122 (`ALGORITHMIC_GATES.md`).", ""]
    for t in TASKS:
        o += [f"## {t} (chance {chance[t]:.4f})", "",
              "| arm | params | " + " | ".join(f"L={L}" for L in LENS) + " |",
              "|---" * (len(LENS) + 2) + "|"]
        for v in ARMS:
            cells = []
            for L in LENS:
                xs = [x for x in acc[(t, v, L)] if x is not None]
                cells.append(f"{st.mean(xs):.3f}" if xs else "—")
            o.append(f"| {v} <br><sub>{LABEL[v]}</sub> | "
                     f"{par[v]:,} | " + " | ".join(cells) + " |" if par[v]
                     else f"| {v} | — | " + " | ".join(cells) + " |")
        o.append("")

    def pair(t, v1, v2, L):
        A, B = acc[(t, v1, L)], acc[(t, v2, L)]
        return [x - y for x, y in zip(A, B) if x is not None and y is not None]

    o += ["## H1 -- does path integration help PARITY more than COPY?", "",
          "| length | parity: path-int - index | copy: path-int - index | "
          "difference |", "|---" * 4 + "|"]
    h1 = {}
    for L in LENS:
        p = stat(pair("parity", "Vanilla", "RoPE", L))
        c = stat(pair("copy", "Vanilla", "RoPE", L))
        h1[L] = (p, c)
        o.append(f"| L={L} | {fmt(p)} | {fmt(c)} | {p['m'] - c['m']:+.3f} |")
    o.append("")

    o += ["## H2 -- does the loop improve LENGTH GENERALIZATION?", "",
          "Retention = accuracy above chance at L, as a fraction of that at L=16.", "",
          "| task | arm | " + " | ".join(f"L={L}" for L in LENS) + " |",
          "|---" * (len(LENS) + 2) + "|"]
    for t in TASKS:
        for v in ARMS:
            b = [x for x in acc[(t, v, 16)] if x is not None]
            if not b:
                continue
            base = st.mean(b) - chance[t]
            cells = []
            for L in LENS:
                xs = [x for x in acc[(t, v, L)] if x is not None]
                cells.append(f"{(st.mean(xs) - chance[t]) / base:.2f}"
                             if xs and base > 1e-6 else "—")
            o.append(f"| {t} | {v} | " + " | ".join(cells) + " |")
    o.append("")

    o += ["## H3 -- does SAMPLING the loop count beat a fixed count at long L?", "",
          "| task | length | LoopedSampled - Looped | verdict |", "|---" * 4 + "|"]
    h3 = {}
    for t in TASKS:
        for L in (64, 128, 256):
            s_ = stat(pair(t, "LoopedSampled", "Looped", L))
            h3[(t, L)] = s_
            o.append(f"| {t} | L={L} | {fmt(s_)} | {vd(s_)} |")
    o.append("")

    o += ["## Verdict", ""]
    pL, cL = h1[128]
    gap = pL["m"] - cL["m"]
    o.append(f"**H1.** At L=128, path integration minus index is {pL['m']:+.3f} on "
             f"parity ({vd(pL)}) and {cL['m']:+.3f} on copy ({vd(cL)}); the "
             f"difference is **{gap:+.3f}**.")
    if vd(pL) == "DETECTABLE POSITIVE" and abs(gap) > max(pL["mde"], cL["mde"]):
        o += ["", "Path integration helps the ITERATIVE task specifically. The "
              "cumsum-of-a-learned-per-token-value mechanism is doing what the "
              "mechanistic prediction said it would, and copy is the control that "
              "rules out a pipeline artifact.", ""]
    elif vd(pL) == "DETECTABLE POSITIVE":
        o += ["", "Path integration helps parity, but NOT distinguishably more than "
              "copy -- so the task-structure story is not supported and something "
              "more generic is doing the work.", ""]
    else:
        o += ["", "Path integration does NOT detectably help parity. The mechanistic "
              "prediction -- theta as a running sum mod 2*pi being a natural parity "
              "register -- is not borne out, and the navigation results do not "
              "transfer to the literature's own tasks.", ""]
    best = max(h3, key=lambda k: h3[k]["m"] if h3[k]["m"] == h3[k]["m"] else -9)
    o.append(f"**H3.** Best sampled-minus-fixed contrast: {best[0]} at L={best[1]}, "
             f"{fmt(h3[best])} -> {vd(h3[best])}.")
    o += ["", "## Scope", "",
          "Two tasks, one train length, one loop count for the fixed arm, n=8. "
          "Binary addition -- the third task in arXiv 2409.15647 -- is not "
          "included; its formatting is fiddly and two clean tasks beat three "
          "sloppy ones. These models are 1 layer at d=128, far smaller than the "
          "literature's, so absolute numbers are not comparable to theirs; the "
          "CONTRASTS are what this run is for."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
