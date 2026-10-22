"""Per-seed accuracy by direction runs spanned, for fig_segments (POST HOC, CPU, seconds). Reads the per-revisit rows that
torus_depth.py and ring_depth.py write to $DEPTH_SCRATCH (not committed; they regenerate in minutes) and writes
out/segments_bins.json (small). Same labels and seeds as summarize_torus.py / summarize_ring.py / ring_plain_bins.py:
  torus (FACTORIAL, seeds 500-503, arms L1 / L4 / LOOP / PATH):
    'short'   plain revisits at gap < 128 moves; runs 2 = retrace (on the torus every plain 2-run revisit is a retrace and
              every retrace spans 2 runs), runs 3+ = not retrace -- the rows of Table 3 / torus_depth_out.txt
    'gap32'   plain revisits at gaps 32-127 moves, by runs spanned -- torus_segments_out.txt
  ring (CANCEL, fresh seeds 2-7, arms RoPE_L1/L2/L3, Vanilla_L1; p_plus 0.5 / 0.75 / 0.9):
    'plain'   runs 2 = retrace (all retraces are plain), runs 3+ = 'neither' (not retrace, not a one-lap return) AND not
              wrap-only -- ring_plain_bins.py's 'plain only' column (ANALYSIS_SCRIPT_AUDIT depth MINOR)
Accuracy per seed = mean correctness over that seed's revisits in the bin; n = revisits pooled over the seeds.
  DEPTH_SCRATCH=... python3 segments_bins.py   -> out/segments_bins.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import OUTDIR  # noqa: E402

S = os.environ.get("DEPTH_SCRATCH", OUTDIR)
BINS = [("2", 2, 3), ("3-4", 3, 5), ("5-8", 5, 9), ("9-16", 9, 17), ("17+", 17, 10 ** 6)]


def per_seed(c, seed, sel, seeds):
    return [float(c[sel & (seed == s)].mean()) if (sel & (seed == s)).any() else None for s in seeds]


out = {"torus": {}, "ring": {}, "bins": [b for b, _, _ in BINS]}
TS = (500, 501, 502, 503)
for arm in ("L1", "L4", "LOOP", "PATH"):
    R = dict(np.load(f"{S}/torus_depth_{arm}.npz"))
    plain = ~R["wrap"]
    pops = {"short": plain & (R["lag"] < 128), "gap32": plain & (R["lag"] >= 32) & (R["lag"] < 128)}
    out["torus"][arm] = {}
    for pn, base in pops.items():
        rows = {}
        for b, lo, hi in BINS:
            if pn == "short":
                sel = base & R["retr"] if b == "2" else base & ~R["retr"] & (R["nruns"] >= lo) & (R["nruns"] < hi)
            else:
                sel = base & (R["nruns"] >= lo) & (R["nruns"] < hi)
            rows[b] = {"acc": per_seed(R["c"], R["seed"], sel, TS), "n": int(sel.sum())}
        out["torus"][arm][pn] = rows
R = dict(np.load(f"{S}/ring_depth.npz"))
RS = range(2, 8)
for p in (0.5, 0.75, 0.9):
    out["ring"][str(p)] = {}
    for arm in ("RoPE_L1", "RoPE_L2", "RoPE_L3", "Vanilla_L1"):
        base = (R["p"] == p) & (R["arm"] == arm) & np.isin(R["seed"], list(RS))
        rows = {}
        for b, lo, hi in BINS:
            if b == "2":
                sel = base & R["retr"]
            else:
                sel = base & ~R["retr"] & ~R["lap"] & ~R["wo"] & (R["nruns"] >= lo) & (R["nruns"] < hi)
            rows[b] = {"acc": per_seed(R["c"], R["seed"], sel, RS), "n": int(sel.sum())}
        out["ring"][str(p)][arm] = rows
out["seeds"] = {"torus": list(TS), "ring": list(RS)}
json.dump(out, open(f"{OUTDIR}/segments_bins.json", "w"), indent=0)
for arm, d in out["torus"].items():
    for pn, rows in d.items():
        print(f"torus {arm:5s} {pn:6s} " + "  ".join(
            f"{b}: {np.mean([v for v in r['acc'] if v is not None]):.3f} (n {r['n']})" if r["n"] else f"{b}: --"
            for b, r in rows.items()))
for p, d in out["ring"].items():
    for arm, rows in d.items():
        print(f"ring p{p} {arm:10s} " + "  ".join(
            f"{b}: {np.mean([v for v in r['acc'] if v is not None]):.3f} (n {r['n']})" if r["n"] else f"{b}: --"
            for b, r in rows.items()))
