"""Ring 'neither' run bins split by plain / wrap-only revisits (POST HOC, CPU, seconds); docs/paper/review_draft5/
ANALYSIS_SCRIPT_AUDIT.md depth MINOR (the ring segment rows of ring_depth_out.txt include wrap-only revisits, the torus
rows do not). Reads $DEPTH_SCRATCH/ring_depth.npz (written by ring_depth.py; not committed, regenerates in a few minutes).
Fresh seeds 2-7, mean of per-seed accuracies, as summarize_ring.py.
  DEPTH_SCRATCH=... python3 ring_plain_bins.py > out/ring_plain_bins_out.txt"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import OUTDIR  # noqa: E402

S = os.environ.get("DEPTH_SCRATCH", OUTDIR)
R = dict(np.load(f"{S}/ring_depth.npz"))
SEEDS = range(2, 8)
for p in (0.5, 0.75, 0.9):
    print(f"== p_plus {p}: 'neither' revisits (not retrace, not lap) by runs spanned")
    print("| runs | wrap-only share | arm: all / plain only / wrap-only only |")
    print("|---|---|---|")
    for lo, hi in ((3, 5), (5, 9), (9, 17)):
        base = (R["p"] == p) & ~R["retr"] & ~R["lap"] & (R["nruns"] >= lo) & (R["nruns"] < hi)
        a0 = base & (R["arm"] == "RoPE_L1")
        cells = []
        for arm in ("RoPE_L1", "RoPE_L2", "RoPE_L3"):
            parts = []
            for sub in (np.ones_like(base), ~R["wo"], R["wo"]):
                v = [R["c"][base & sub & (R["arm"] == arm) & (R["seed"] == s)].mean() for s in SEEDS
                     if (base & sub & (R["arm"] == arm) & (R["seed"] == s)).any()]
                parts.append(f"{np.mean(v):.3f}" if v else "--")
            cells.append(f"{arm} {' / '.join(parts)}")
        sh = R["wo"][a0 & np.isin(R["seed"], list(SEEDS))].mean() if a0.any() else float("nan")
        print(f"| {lo}-{hi - 1} | {sh:.2f} | " + "; ".join(cells) + " |")
