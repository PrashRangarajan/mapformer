"""Does the filter pay where its premise finally holds?

Every correction result in this project has been measured where the correction had
nothing to correct: Match-Query's actions are clean, and the torus is at ceiling.
Stochastic explore transitions supply drift (0 -> 13 cells at p=0.10, gated before
training) while the observations still reflect TRUE cells, so there is both
something to correct and something to correct with.

PRIMARY: Level15 - Vanilla at p=0.10 against the SAME contrast at p=0.0 in the same
batch. The effect must GROW with p; a flat effect says the InEKF does not buy drift
correction even when drift is what is being measured.
SECONDARY: the interaction (L15Loop - Loop) - (L15 - Vanilla) at p=0.10, re-asking
last night's question on a task that has a premise.
"""
import argparse, json, re, statistics as st
from pathlib import Path

ARMS = ["Vanilla", "Level15", "Looped", "Level15Looped"]
TAGS = [("p0", 0.0), ("p010", 0.10)]
SEEDS = list(range(8))
CHANCE = 0.0625


def parse(runs_dir, tag, v, s):
    f = Path(runs_dir) / f"{tag}_{v}_s{s}.log"
    if not f.exists():
        return None
    txt = f.read_text()
    accs = {int(m[0]): float(m[1])
            for m in re.findall(r"T_query=(\d+): acc=([0-9.]+)", txt)}
    ls = [float(x) for x in re.findall(r"loss=([0-9.]+)", txt)]
    fin = re.search(r"final_loss=([0-9.]+)", txt)
    return {"acc": accs, "final_loss": float(fin.group(1)) if fin else
            (ls[-1] if ls else None)}


def stat(d):
    d = [x for x in d if x is not None]
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), t=float("nan"),
                    mde=float("nan"), n=n, pos=0)
    sd = st.stdev(d); se = sd / n ** 0.5
    return dict(m=st.mean(d), sd=sd, t=st.mean(d) / se if se else float("nan"),
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
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--tq", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = {(tag, v, s): parse(a.runs_dir, tag, v, s)
         for tag, _p in TAGS for v in ARMS for s in SEEDS}
    acc = {(tag, v): [D[(tag, v, s)]["acc"].get(a.tq)
                      if D[(tag, v, s)] else None for s in SEEDS]
           for tag, _p in TAGS for v in ARMS}
    loss = {(tag, v): [D[(tag, v, s)]["final_loss"] if D[(tag, v, s)] else None
                       for s in SEEDS] for tag, _p in TAGS for v in ARMS}

    o = ["# Does the filter pay where its premise holds?", "",
         f"Match-Query 128^2, T_explore=512, T_query={a.tq}, chance {CHANCE:.4f}.",
         "4 arms x 2 noise levels x 8 seeds, one batch. Stochastic transitions apply",
         "to the EXPLORE phase only, and evaluation uses the SAME noise as training.",
         "", "Gated before training: every shortcut gate stays at its clean-task",
         "level while drift rises 0 -> 13.05 cells at p=0.10.", "",
         "## Accuracy (mean +/- sd)", "",
         "| arm | p=0.0 | p=0.10 | drop |", "|---|---|---|---|"]
    for v in ARMS:
        c = [x for x in acc[("p0", v)] if x is not None]
        n_ = [x for x in acc[("p010", v)] if x is not None]
        o.append(f"| {v} | "
                 + (f"{st.mean(c):.3f} +/- {st.stdev(c):.3f}" if len(c) > 1 else "—")
                 + " | "
                 + (f"{st.mean(n_):.3f} +/- {st.stdev(n_):.3f}" if len(n_) > 1 else "—")
                 + " | "
                 + (f"{st.mean(n_) - st.mean(c):+.3f}"
                    if len(c) > 1 and len(n_) > 1 else "—") + " |")
    o += ["", "## Convergence (re-measured here; recipe transfer across tasks is an",
          "assumption, not a result)", "",
          "| arm | p=0.0 final loss | p=0.10 final loss |", "|---|---|---|"]
    for v in ARMS:
        cells = []
        for tag, _p in TAGS:
            ls = [x for x in loss[(tag, v)] if x is not None]
            cells.append(f"{st.mean(ls):.3f}  ({min(ls):.3f} – {max(ls):.3f})"
                         if ls else "—")
        o.append(f"| {v} | " + " | ".join(cells) + " |")

    def pair(v1, v2, tag):
        A, B = acc[(tag, v1)], acc[(tag, v2)]
        return [x - y for x, y in zip(A, B) if x is not None and y is not None]

    def inter(tag):
        z = zip(acc[(tag, "Level15Looped")], acc[(tag, "Looped")],
                acc[(tag, "Level15")], acc[(tag, "Vanilla")])
        return [(a1 - b1) - (c1 - d1) for a1, b1, c1, d1 in z
                if None not in (a1, b1, c1, d1)]

    o += ["", "## Contrasts", "",
          "| contrast | p=0.0 | p=0.10 | verdict at p=0.10 |", "|---|---|---|---|"]
    tests = [("Level15", "Vanilla", "THE PRIMARY: filter, no loop"),
             ("Level15Looped", "Looped", "filter, inside the loop"),
             ("Looped", "Vanilla", "loop, no filter"),
             ("Level15Looped", "Level15", "loop, inside the filter")]
    for v1, v2, lab in tests:
        s0, s1 = stat(pair(v1, v2, "p0")), stat(pair(v1, v2, "p010"))
        o.append(f"| {v1} - {v2} <br><sub>{lab}</sub> | {fmt(s0)} | {fmt(s1)} | "
                 f"{vd(s1)} |")
    i0, i1 = stat(inter("p0")), stat(inter("p010"))
    o.append(f"| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | "
             f"**{fmt(i0)}** | **{fmt(i1)}** | **{vd(i1)}** |")

    p0, p1 = stat(pair("Level15", "Vanilla", "p0")), stat(pair("Level15", "Vanilla", "p010"))
    growth = p1["m"] - p0["m"]
    o += ["", "## Verdict against the pre-registration", "",
          f"**Primary.** Level15 - Vanilla is {p0['m']:+.3f} at p=0 and "
          f"{p1['m']:+.3f} at p=0.10; the effect changes by **{growth:+.3f}** as "
          f"drift goes from 0 to 13 cells. At p=0.10 it is {vd(p1)} "
          f"(MDE {p1['mde']:.3f}, {p1['pos']}/{p1['n']} seeds).", ""]
    if vd(p1) == "DETECTABLE POSITIVE" and growth > p1["mde"]:
        o += ["**The filter pays where its premise holds.** The effect both clears "
              "its noise floor and GROWS with drift, which is the signature a "
              "correction mechanism should have and which no previous test in this "
              "project produced. Every prior null was then a premise failure rather "
              "than a mechanism failure -- a claim that now needs the dose-response "
              "curve at a third noise level before it is more than two points and a "
              "line (the H12 budget-curve error).", ""]
    elif vd(p1) == "DETECTABLE POSITIVE":
        o += ["**The filter helps under drift, but the effect does NOT grow with "
              "drift.** A correction mechanism whose benefit is flat in the amount "
              "to be corrected is not doing correction; this is the stabilisation "
              "reading again, now on a task built to give inference its best case.", ""]
    else:
        o += ["**The filter does not pay even here.** Drift is present and "
              "measurable, observations carry the correction signal, the task has "
              "headroom, and the contrast is still inside its noise floor. That is "
              "the sharpest test the InEKF has been given, and the "
              "'stabilisation, not inference' reading survives it. Per rule 11 this "
              "is UNMEASURED rather than zero -- report the MDE beside it.", ""]
    o += [f"**Secondary.** Interaction at p=0.10: {fmt(i1)} -> {vd(i1)}. Last "
          f"night's clean-torus 2x2 found no complementarity on a task with no "
          f"premise; this re-asks it on one that has a premise.", "",
          "## Scope", "",
          "One task, one map size, two noise levels, n=8, one loop count. Two noise "
          "levels are two points -- a dose-response claim needs a third. Query "
          "transitions stay clean by design: scoring is keyed on the TRUE cell, so "
          "noisy query transitions would make the answer unknowable rather than the "
          "task harder."]
    Path(a.out).write_text("\n".join(o) + "\n")
    json.dump({f"{t_}|{v}": acc[(t_, v)] for t_, _p in TAGS for v in ARMS},
              open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
