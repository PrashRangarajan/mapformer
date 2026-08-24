"""Aggregate Compositional Match-Query runs: mean±std per variant, per category
(exact / cross / all), per eval T_query. Reads runs/cmq_sweep/s*/<variant>_cmq.json.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "cmq_sweep"))
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(_REPO, "COMPOSITIONAL_MATCH_QUERY_RESULTS.md"))
    args = ap.parse_args()

    # data[variant][TQ][cat][metric] -> list over seeds
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    seeds_found = defaultdict(set)
    for v in args.variants:
        for f in sorted(glob.glob(os.path.join(args.runs_dir, "s*", f"{v}_cmq.json"))):
            seed = os.path.basename(os.path.dirname(f))
            res = json.load(open(f))
            seeds_found[v].add(seed)
            for TQ, r in res.items():
                for cat in ("exact", "cross", "all"):
                    if cat in r:
                        data[v][int(TQ)][cat]["acc"].append(r[cat]["acc"])
                        data[v][int(TQ)][cat]["nll"].append(r[cat]["nll"])

    def cell(vals):
        a = np.array(vals)
        if len(a) == 0:
            return "—"
        return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a.mean():.3f}"

    TQs = sorted({tq for v in data for tq in data[v]})
    lines = ["# Compositional Match-Query — results (mean ± std over seeds)", "",
             "Blind continuation in a repeated-motif world. chance = 0.0625. "
             "`exact` = path-integration matching; `cross` = path integration AND "
             "motif abstraction (the synergy target). Held-out env (seed=10000).", ""]
    for cat in ("cross", "exact", "all"):
        lines.append(f"## {cat}_acc")
        header = "| variant | " + " | ".join(f"TQ={tq}" for tq in TQs) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(TQs) + 1))
        for v in args.variants:
            if v not in data:
                continue
            row = [f"{v} (n={len(seeds_found[v])})"]
            for tq in TQs:
                row.append(cell(data[v][tq][cat]["acc"]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
