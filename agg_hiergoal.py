"""Aggregate multi-seed hierarchical-goal checkpoints into mean +/- std tables
(held-out action accuracy and NLL, by eval explore-length)."""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

DISPLAY = {"Vanilla": "MapWM-Flat", "Hourglass_k2": "MapWM-Hier",
           "PlainFlat": "Plain-Flat", "PlainHourglass": "Plain-Hier"}


def ms(xs):
    if not xs:
        return (float("nan"), float("nan"), 0)
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0, len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="HIERGOAL_MULTISEED.md")
    args = ap.parse_args()

    acc = {v: {t: [] for t in args.lengths} for v in args.variants}
    nll = {v: {t: [] for t in args.lengths} for v in args.variants}
    found = {v: [] for v in args.variants}
    for v in args.variants:
        for s in args.seeds:
            cp = Path(args.runs_dir) / f"seed{s}" / f"{v}_hiergoal.pt"
            if not cp.exists():
                print(f"MISSING {cp}"); continue
            c = torch.load(cp, map_location="cpu", weights_only=False)
            found[v].append(s)
            for t in args.lengths:
                r = c["results"].get(t) or c["results"].get(str(t))
                if r:
                    acc[v][t].append(r["acc"]); nll[v][t].append(r["nll"])

    def disp(v): return DISPLAY.get(v, v)

    lines = ["# Hierarchical goal-directed navigation — multi-seed (mean ± std)\n",
             f"Train T_explore=64; eval at listed T_explore (>64 = OOD). "
             f"Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.\n",
             "Seeds found: " + ", ".join(f"{disp(v)}={found[v]}" for v in args.variants) + "\n"]
    for label, tbl in [("Held-out action accuracy", acc), ("Held-out NLL (lower better)", nll)]:
        lines.append(f"\n## {label}\n")
        lines.append("| variant | " + " | ".join(f"T_exp={t}" for t in args.lengths) + " |")
        lines.append("|" + "---|" * (len(args.lengths) + 1))
        for v in args.variants:
            cells = []
            for t in args.lengths:
                m, sd, n = ms(tbl[v][t])
                cells.append(f"{m:.3f} ± {sd:.3f}")
            lines.append(f"| {disp(v)} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {disp(v): {str(t): ms(acc[v][t]) for t in args.lengths} for v in args.variants}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
