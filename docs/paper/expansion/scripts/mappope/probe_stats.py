"""Unpaired permutation tests (stats_core.perm2_p) on per-seed probe readouts from kernel_probe_<batch>.json, and the
decomposition of the PoPE - RoPE accuracy gap over retrace strata (share x delta acc). ANALYSIS, post hoc.
Run from /home/prashr: python3 mapformer/docs/paper/expansion/scripts/mappope/probe_stats.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import perm2_p  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SFX = os.environ.get("SUFFIX", "")   # SUFFIX=_hstratum reads kernel_probe_<batch>_hstratum.json (ANALYSIS_SCRIPT_AUDIT B2)
COL = ["share", "acc", "massCell", "massRT", "massOff1", "C", "C0", "mAct", "mObsX"]
lines = []


def L(s):
    print(s); lines.append(s)


def get(J, v, sn, col):
    k = str((v, sn))
    return np.array([r[COL.index(col)] for r in J[k]], dtype=float) if k in J else None


def contrast(J, a, b, sn, col):
    x, y = get(J, a, sn, col), get(J, b, sn, col)
    if x is None or y is None:
        return
    L(f"  {col:9s} {sn:16s} {b} {np.mean(y):.3f} vs {a} {np.mean(x):.3f}: d {np.mean(y) - np.mean(x):+.3f}  perm p {perm2_p(x, y)['p']:.4f}"
      f"  (seeds {b} > {a} max: {int((y > x.max()).sum())}/{len(y)})")


for batch in ("paper2x2", "score_rank"):
    f = f"{HERE}/kernel_probe_{batch}{SFX}.json"
    if not os.path.exists(f):
        continue
    J = json.load(open(f))[batch]
    L(f"\n=== {batch} ===")
    if batch == "paper2x2":
        L("index arms, PoPE-Flat vs RoPE:")
        for sn in ("all", "R1 (1-step reversal)", "R2", "R3+", "not retrace"):
            contrast(J, "RoPE", "PoPE-Flat", sn, "acc")
        contrast(J, "RoPE", "PoPE-Flat", "all", "massOff1")
        contrast(J, "RoPE", "PoPE-Flat", "R3+", "massRT")
        tot = 0.0
        L("decomposition of PoPE - RoPE accuracy gap (mean share x mean delta acc):")
        for sn in ("R1 (1-step reversal)", "R2", "R3+", "not retrace"):
            sh = np.mean(get(J, "RoPE", sn, "share")); d = np.mean(get(J, "PoPE-Flat", sn, "acc")) - np.mean(get(J, "RoPE", sn, "acc"))
            tot += sh * d; L(f"  {sn:22s} share {sh:.3f} x d {d:+.3f} = {sh * d:+.4f}")
        L(f"  sum {tot:+.4f} (total gap {np.mean(get(J, 'PoPE-Flat', 'all', 'acc')) - np.mean(get(J, 'RoPE', 'all', 'acc')):+.4f})")
        L("path arms, MapPoPE-Flat vs Vanilla (MapWM), r2:")
        for col in ("acc", "massCell", "C", "C0", "mAct", "mObsX"):
            contrast(J, "Vanilla", "MapPoPE-Flat", "all", col)
        c, c0 = get(J, "Vanilla", "all", "C"), get(J, "Vanilla", "all", "C0")
        L(f"  MapWM C0 - C (content phase's cost to target coherence): {np.mean(c0 - c):+.3f}, positive on {int((c0 > c).sum())}/{len(c)} seeds")
    else:
        for a, b in (("Vanilla", "MapPoPE-Pair"), ("Vanilla_r4mi", "MapPoPE-Pair_r4mi"), ("Vanilla", "Vanilla_r4mi"),
                     ("MapPoPE-Pair", "MapPoPE-Pair_r4mi")):
            L(f"{b} vs {a}:")
            for sn in ("plain gap<128", "plain gap>=128", "wrap-only"):
                for col in ("acc", "massCell", "C", "C0"):
                    contrast(J, a, b, sn, col)
        for v in ("Vanilla", "MapPoPE-Pair", "Vanilla_r4mi", "MapPoPE-Pair_r4mi"):
            for sn in ("plain gap<128", "plain gap>=128", "wrap-only"):
                c, c0 = get(J, v, sn, "C"), get(J, v, sn, "C0")
                if c is not None:
                    L(f"  {v:18s} {sn:15s} C {np.mean(c):.3f}  C0 {np.mean(c0):.3f}  C0-C {np.mean(c0 - c):+.3f}")
open(f"{HERE}/probe_stats{SFX}_out.txt", "w").write("\n".join(lines) + "\n")
