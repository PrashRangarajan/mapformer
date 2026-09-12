"""PAIRSPLIT_PREREG.md analysis: EMPair vs EMPairConst at n=48.

Seeds 0-7 come from the PAIRORIGIN / PAIRCONST batches (reuse licensed by the in-batch bitwise
determinism re-check), seeds 8-47 from runs/pairsplit. The fresh seeds are reported ALONE beside
the pooled figure, because the first eight seeds in this line have over-estimated three times.

    python3 -m mapformer.analyze_pairsplit
"""
from __future__ import annotations

import json

import numpy as np

from mapformer.ckpt_guard import REPO, load_checkpoint
from mapformer.stats_guard import paired, replication_split, rule9, table

N = 48
SRC = {  # (variant, dir for seeds 0-7, dir for seeds 8+)
    "EMPair_r4": ("runs/pairorigin", "runs/pairsplit"),
    "EMPairConst_r4": ("runs/pairconst", "runs/pairsplit"),
}
P0 = ("VanillaEM_P0_r4", "runs/dof/recency", 24)


def load(variant, d0, d8, n):
    acc, loss = {}, {}
    for s in range(n):
        d = REPO / (d0 if s < 8 else d8) / f"{variant}_s{s}"
        js = d / f"{variant}_recency.json"
        if not js.exists():
            raise FileNotFoundError(js)
        acc[s] = json.load(open(js))["1024"]["acc"]
        loss[s] = float(load_checkpoint(d / f"{variant}_recency.pt").losses[-1])
    return acc, loss


def main():
    A, Lo = {}, {}
    for v, (d0, d8) in SRC.items():
        A[v], Lo[v] = load(v, d0, d8, N)
    A[P0[0]], Lo[P0[0]] = load(P0[0], P0[1], P0[1], P0[2])

    L = ["# PAIRSPLIT report (PAIRSPLIT_PREREG.md)\n", "## Per-arm\n",
         "| arm | n | acc T=1024 | final loss |", "|---|---|---|---|"]
    for v in list(SRC) + [P0[0]]:
        a = np.array(list(A[v].values()))
        L.append(f"| {v} | {len(a)} | **{a.mean():.3f} +/- {a.std(ddof=1):.3f}** | "
                 f"{np.mean(list(Lo[v].values())):.3f} |")

    c1 = paired(A["EMPair_r4"], A["EMPairConst_r4"], "EMPair - EMPairConst (C1: freedom)")
    c2 = paired({s: A["EMPairConst_r4"][s] for s in range(24)},
                A[P0[0]], "EMPairConst - P0 (C2: pathway), n=24")
    c3 = paired({s: A["EMPair_r4"][s] for s in range(24)}, A[P0[0]], "EMPair - P0, n=24")
    L += ["", "## Contrasts\n", table([c1, c2, c3]),
          "", "## Fresh-seed split on C1 (the registered guard)\n", replication_split(c1, first_k=8).report()]

    acc = [A[v][s] for v in SRC for s in range(N)]
    loss = [Lo[v][s] for v in SRC for s in range(N)]
    L += ["", "## Rule 9\n", str(rule9(acc, loss))]

    band = 0.05 <= c1.delta <= 0.15 and c1.verdict == "DETECTABLE"
    L += ["", "## Registered verdicts\n",
          f"- **S1 (freedom is real but small: detectable, 0.05-0.15)**: {c1.delta:+.3f} "
          f"(MDE {c1.mde:.3f}) -> **{'CONFIRMED' if band else 'not confirmed'}**",
          f"- **S2 (freedom buys no accuracy)**: -> "
          f"**{'CONFIRMED -- the gain is the pathway' if abs(c1.delta) < c1.mde else 'not confirmed'}**",
          f"- **S3 (the n=8 estimate was low: > +0.15)**: -> "
          f"**{'CONFIRMED' if c1.delta > 0.15 and c1.verdict == 'DETECTABLE' else 'not confirmed'}**",
          f"- **C2 at n=24**: {c2.delta:+.3f} (MDE {c2.mde:.3f}) -> **{c2.verdict}**"]
    det = REPO / "runs/pairsplit/DETERMINISM.txt"
    L += ["", "## Determinism re-check\n", "```",
          det.read_text().strip() if det.exists() else "MISSING", "```"]
    txt = "\n".join(L)
    print(txt)
    (REPO / "runs/pairsplit/REPORT.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
