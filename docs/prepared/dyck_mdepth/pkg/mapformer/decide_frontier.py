"""Pick the venue for the depth-vs-loop frontier from the algorithmic result.

THE QUESTION the frontier answers: does looping substitute for real depth EQUALLY
WELL with and without path integration? LOOP_HEADROOM hints it does not -- the loop
buys +0.099 on the index arm and +0.414 on the path-integrated one, interaction
+0.315 -- but there is no depth baseline for the index row, so the two frontiers
cannot be compared. Eight arms fix that: {index, path-int} x {1, 2, 3 real layers,
loop x4}.

THE VENUE depends on whether path integration does anything on the algorithmic
tasks. If it does, they are a 40x cheaper venue (about 90 s a run against 62 min on
Match-Query) and the frontier can afford 16 seeds instead of 8. If it does not,
there is no path-integration effect there to interact with, the question is moot on
those tasks, and Match-Query is the only venue -- worth its 6.6 hours precisely
because it would then be the sole evidence.

Decision rule, fixed here rather than chosen after seeing the numbers:
  parity (path-int - index) at L=128 DETECTABLE POSITIVE  -> algorithmic venue
  otherwise                                               -> Match-Query venue
"""
import json, statistics as st, sys
from pathlib import Path

RUNS = Path("mapformer/runs/algorithmic")
SEEDS = range(8)


def acc(task, v, s, L=128):
    f = RUNS / task / f"{v}_s{s}" / f"{v}_{task}.json"
    if not f.exists():
        return None
    return json.load(open(f))["acc"].get(str(L))


def main():
    d = [x - y for x, y in ((acc("parity", "Vanilla", s), acc("parity", "RoPE", s))
                            for s in SEEDS) if x is not None and y is not None]
    c = [x - y for x, y in ((acc("copy", "Vanilla", s), acc("copy", "RoPE", s))
                            for s in SEEDS) if x is not None and y is not None]
    if len(d) < 3:
        print("INSUFFICIENT DATA -- defaulting to the Match-Query venue")
        print("VENUE=matchquery"); return
    m, sd = st.mean(d), st.stdev(d)
    mde = 2.8 * sd / len(d) ** 0.5
    cm = st.mean(c) if len(c) > 1 else float("nan")
    print(f"parity  path-int - index @L=128: {m:+.3f} (sd {sd:.3f}, MDE {mde:.3f}, "
          f"{sum(1 for x in d if x > 0)}/{len(d)})")
    print(f"copy    path-int - index @L=128: {cm:+.3f}   (the control)")
    if m > mde:
        print("-> path integration is DETECTABLE on parity. The algorithmic tasks "
              "are a live venue and are 40x cheaper, so the frontier runs there at "
              "16 seeds.")
        print("VENUE=algorithmic")
    else:
        print("-> path integration is NOT detectable on parity, so there is no "
              "effect there for looping to interact with. The frontier runs on "
              "Match-Query, where the path-integration effect is large and "
              "established.")
        print("VENUE=matchquery")


if __name__ == "__main__":
    main()
