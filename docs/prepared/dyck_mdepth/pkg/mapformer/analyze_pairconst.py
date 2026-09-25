"""PAIRCONST_PREREG.md analysis. Written before any arm finished.

    python3 -m mapformer.analyze_pairconst
"""
from __future__ import annotations

import json

import numpy as np

from mapformer.ckpt_guard import REPO, load_checkpoint
from mapformer.stats_guard import paired, rule9, table

SEEDS = range(8)
# arm -> (variant, run dir)  -- comparators are the stored PAIRORIGIN arms, reused only if the
# determinism re-check in MANIPULATION.txt is bitwise identical.
ARMS = {"EMPairConst_r4": (   "EMPairConst_r4", "runs/pairconst"),
        "EMPair_r4": (        "EMPair_r4",      "runs/pairorigin"),
        "VanillaEM_P0_r4": (  "VanillaEM_P0_r4", "runs/pairorigin")}


def main():
    rows = {}
    for arm, (v, d) in ARMS.items():
        for s in SEEDS:
            js = REPO / d / f"{v}_s{s}" / f"{v}_recency.json"
            if not js.exists():
                raise FileNotFoundError(js)
            r = json.load(open(js))
            ck = load_checkpoint(REPO / d / f"{v}_s{s}" / f"{v}_recency.pt")
            rows[(arm, s)] = dict(acc=r["1024"]["acc"], acc2048=r["2048"]["acc"],
                                  loss=float(ck.losses[-1]))
    g = lambda a, k: {s: rows[(a, s)][k] for s in SEEDS}

    L = ["# PAIRCONST report (PAIRCONST_PREREG.md)\n", "## Per-arm\n",
         "| arm | acc T=1024 | acc T=2048 | final loss |", "|---|---|---|---|"]
    for a in ARMS:
        A = np.array([rows[(a, s)]["acc"] for s in SEEDS])
        L.append(f"| {a} | **{A.mean():.3f} +/- {A.std(ddof=1):.3f}** | "
                 f"{np.mean([rows[(a,s)]['acc2048'] for s in SEEDS]):.3f} | "
                 f"{np.mean([rows[(a,s)]['loss'] for s in SEEDS]):.3f} |")

    c1 = paired(g("EMPair_r4", "acc"), g("EMPairConst_r4", "acc"), "EMPair - EMPairConst (C1: freedom)")
    c2 = paired(g("EMPairConst_r4", "acc"), g("VanillaEM_P0_r4", "acc"), "EMPairConst - P0 (C2/C3: parameters)")
    c3 = paired(g("EMPair_r4", "acc"), g("VanillaEM_P0_r4", "acc"), "EMPair - P0 (the effect under test)")
    L += ["", "## Contrasts (paired by seed, n=8)\n", table([c1, c2, c3])]

    acc = [rows[(a, s)]["acc"] for a in ARMS for s in SEEDS]
    loss = [rows[(a, s)]["loss"] for a in ARMS for s in SEEDS]
    L += ["", "## Rule 9\n", str(rule9(acc, loss))]

    C1 = c1.delta >= 0.20 and c1.verdict == "DETECTABLE"
    C3 = c2.delta >= 0.20 and c2.verdict == "DETECTABLE"
    L += ["", "## Registered verdicts\n",
          f"- **C1 (per-pair freedom does the work)**: {c1.delta:+.3f} (MDE {c1.mde:.3f}; needs "
          f">= +0.20 AND detectable) -> **{'CONFIRMED' if C1 else 'NOT CONFIRMED'}**",
          f"- **C2 (the parameters buy nothing)**: EMPairConst - P0 = {c2.delta:+.3f} "
          f"(MDE {c2.mde:.3f}) -> **{'MET' if abs(c2.delta) < c2.mde else 'NOT MET'}**",
          f"- **C3 (it was capacity)**: -> **{'CONFIRMED -- PAIRORIGIN kernel reading WITHDRAWN' if C3 else 'not confirmed'}**"]
    if not C1 and not C3:
        L.append("- C1 and C3 both fail: unresolved; report both contrasts with their MDEs.")
    man = REPO / "runs/pairconst/MANIPULATION.txt"
    L += ["", "## Manipulation checks\n", "```",
          man.read_text().strip() if man.exists() else "MISSING", "```"]
    txt = "\n".join(L)
    print(txt)
    (REPO / "runs/pairconst/REPORT.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
