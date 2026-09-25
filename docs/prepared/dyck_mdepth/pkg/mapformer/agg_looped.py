"""Aggregate the looped-transformer pilot into one horizon table.

Answers the two questions the depth grid could not separate, with the verdict
rules written before the runs finished:

  Q1  index arm: is horizon about EFFECTIVE DEPTH or about DISTINCT layers?
  Q2  path-integrated arm: does recursion inherit the penalty real depth carried
      at long range (Vanilla L2 d256 0.976 -> L4 d256 0.782 at interval 65+)?

Reads the per-config JSONs written by probe_revisit_distance and reports the
horizon (largest interval bucket clearing the blank floor by >0.10), which is the
same definition HORIZON_RESULTS.md used.
"""
import argparse
import json
import os

import numpy as np

MARGIN = 0.10       # a bucket "clears" if it beats the floor by this much
FLOOR = 0.50        # blank rate on this task; the probe reports it per bucket


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def horizon(buckets, keys, floor=FLOOR):
    """Largest bucket clearing the floor by >MARGIN, as a label."""
    best = None
    for k in keys:
        vals = buckets.get(k) or []
        if vals and np.mean(vals) > floor + MARGIN:
            best = k
    return best or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = {}
    for lbl, fn in (("L1", "LOOPED_L1.json"), ("L4", "LOOPED_L4.json"),
                    ("Loop4", "LOOPED_Loop4.json")):
        cfg[lbl] = load(os.path.join(a.repo, fn))

    o = ["# Does recursion buy attention horizon cheaply?", "",
         "Weight-shared looped block (1 block x4, ALBERT-style, no per-iteration",
         "depth embedding) against 1 real layer and 4 real layers. Torus paper task,",
         "d=128, 2 heads, 300 epochs, 5% warmup + cosine, n=3 seeds.", "",
         "**All three configs were retrained in this batch.** The published horizon",
         "table used 16 epochs of LinearLR; this uses 300 of warmup+cosine, so those",
         "numbers are not a valid baseline for these (rules 3 and 10).", "",
         "Parameter parity is exact: Looped 207,457 = L1 207,457, against L4 802,273.", ""]

    keys = None
    for lbl in ("L1", "L4", "Loop4"):
        d = cfg[lbl]
        if d:
            for v in d:
                keys = list(d[v].keys())
                break
        if keys:
            break
    if not keys:
        o.append("**No probe output found -- nothing to aggregate.**")
        open(a.out, "w").write("\n".join(o) + "\n"); print("\n".join(o)); return

    name = {("L1", "RoPE"): "RoPE L1 (204K, depth 1)",
            ("L4", "RoPE"): "RoPE L4 (802K, depth 4)",
            ("Loop4", "RoPELooped"): "RoPE Looped x4 (204K, depth 4)",
            ("L1", "Vanilla"): "MapFormer L1 (204K, depth 1)",
            ("L4", "Vanilla"): "MapFormer L4 (802K, depth 4)",
            ("Loop4", "Looped"): "MapFormer Looped x4 (204K, depth 4)"}

    got = {}
    for grp, arms in (("INDEX (RoPE)", [("L1", "RoPE"), ("L4", "RoPE"),
                                        ("Loop4", "RoPELooped")]),
                      ("PATH-INTEGRATED (MapFormer-WM)", [("L1", "Vanilla"), ("L4", "Vanilla"),
                                                          ("Loop4", "Looped")])):
        o += [f"## {grp}", "",
              "| config | " + " | ".join(keys) + " | horizon |",
              "|---" * (len(keys) + 2) + "|"]
        for lbl, v in arms:
            d = (cfg[lbl] or {}).get(v)
            if not d:
                o.append(f"| {name[(lbl,v)]} | " + " | ".join("—" for _ in keys) + " | — |")
                continue
            got[(lbl, v)] = d
            o.append(f"| {name[(lbl,v)]} | " +
                     " | ".join(f"{np.mean(d[k]):.3f}" if d.get(k) else "—" for k in keys) +
                     f" | **{horizon(d, keys)}** |")
        o.append("")

    o += ["## Verdict", ""]
    long_keys = [k for k in keys if k in ("33-64", "65+")] or keys[-2:]

    def longmean(lbl, v):
        d = got.get((lbl, v))
        if not d:
            return None
        vals = [np.mean(d[k]) for k in long_keys if d.get(k)]
        return float(np.mean(vals)) if vals else None

    # Q1 -- index arm
    r1, r4, rl = (horizon(got.get((l, v), {}), keys)
                  for l, v in (("L1", "RoPE"), ("L4", "RoPE"), ("Loop4", "RoPELooped")))
    o += [f"**Q1 index arm.** horizon: L1 {r1}, L4 {r4}, Looped x4 {rl}.", ""]
    if rl == r4 and r4 != r1:
        o += ["Recursion buys DEPTH'S HORIZON AT A QUARTER OF THE PARAMETERS. Effective "
              "depth, not layer specialisation, is what the depth grid was measuring.", ""]
    elif rl == r1 and r4 != r1:
        o += ["Recursion buys NOTHING. A shared block does not extend the horizon the way "
              "distinct layers do, so what depth bought was SPECIALISATION, not iteration. "
              "Looping is not a cheap substitute.", ""]
    else:
        o += ["Partial or ambiguous -- read the buckets rather than the summary label.", ""]

    # Q2 -- path-integrated arm
    v1, v4, vl = (longmean(*x) for x in (("L1", "Vanilla"), ("L4", "Vanilla"),
                                         ("Loop4", "Looped")))
    if None not in (v1, v4, vl):
        o += [f"**Q2 path-integrated arm**, mean over {'/'.join(long_keys)}: "
              f"L1 {v1:.3f}, L4 {v4:.3f}, Looped x4 {vl:.3f}.", ""]
        depth_hurt = v4 < v1 - 0.05
        loop_hurt = vl < v1 - 0.05
        if depth_hurt and loop_hurt:
            o += ["Recursion INHERITS the long-range penalty that real depth carried. "
                  "Stacking recursion on MapFormer is counterproductive, and a recursive "
                  "MapFormer is dead before it is built.", ""]
        elif depth_hurt and not loop_hurt:
            o += ["Real depth hurts at long range but RECURSION DOES NOT -- the penalty is "
                  "about distinct-layer capacity, not iteration. Recursion composes with "
                  "path integration where depth does not. This is the interesting outcome.", ""]
        elif not depth_hurt:
            o += ["Depth did NOT hurt at long range under this schedule and budget, which "
                  "does not reproduce the earlier grid's non-monotonicity. That grid ran at "
                  "16 epochs of LinearLR, so the earlier finding was plausibly an "
                  "optimisation artifact -- as its own caveat allowed. Q2 is answered by "
                  "dissolving it.", ""]
    else:
        o += ["**Q2 incomplete** -- missing arms.", ""]

    o += ["## Scope", "",
          "One task (torus, T=128), one width, one loop count (4), n=3. A shared block",
          "with no depth embedding is the most conservative form of recursion; Universal",
          "Transformer's per-iteration timestep embedding and recomputing theta each",
          "iteration (iterative position refinement) are both untested here and are the",
          "natural follow-ons IF this pilot is positive."]

    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
