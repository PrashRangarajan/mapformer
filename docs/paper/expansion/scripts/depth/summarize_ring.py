"""Tables from ring_depth.py (POST HOC): accuracy by revisit class, attention, probes, for index RoPE 1-3 layers and
path 1 layer on the biased ring, FRESH seeds 2-7 only (0-1 are the pilot).
  python3 summarize_ring.py   (reads $DEPTH_SCRATCH/ring_depth.npz and out/ring_depth.json); writes out/ring_depth_out.txt"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import OUTDIR  # noqa: E402

S = os.environ.get("DEPTH_SCRATCH", OUTDIR)
SEEDS = (2, 3, 4, 5, 6, 7)
ARMS = ("RoPE_L1", "RoPE_L2", "RoPE_L3", "Vanilla_L1")
PS = (0.5, 0.75, 0.9, 1.0)


def classes(R):
    plain = ~R["wo"]
    out = {"all": np.ones(len(R["c"]), bool), "retrace": R["retr"], "lap, not retrace": R["lap"] & ~R["retr"],
           "neither": ~R["retr"] & ~R["lap"]}
    for lo, hi in [(3, 5), (5, 9), (9, 17), (17, 10 ** 6)]:
        out[f"neither, runs {lo}-{hi - 1 if hi < 10 ** 6 else '+'}"] = ~R["retr"] & ~R["lap"] & (R["nruns"] >= lo) & \
            (R["nruns"] < hi)
    for lo, hi in [(2, 8), (8, 32), (32, 129)]:
        out[f"plain gap {lo}-{hi - 1}"] = plain & (R["gap"] >= lo) & (R["gap"] < hi)
    out["wrap-only"] = R["wo"]
    return out


def acc(R, sel, arm, p):
    a = []
    for s in SEEDS:
        m = sel & (R["arm"] == arm) & (R["p"] == p) & (R["seed"] == s)
        if m.any():
            a.append(R["c"][m].mean())
    return a


def main():
    R = dict(np.load(f"{S}/ring_depth.npz")); J = json.load(open(f"{OUTDIR}/ring_depth.json"))
    C = classes(R); L = []
    for p in PS:
        ref = (R["arm"] == "RoPE_L1") & (R["p"] == p) & np.isin(R["seed"], SEEDS)
        L.append(f"\n== p_plus {p}: accuracy by revisit class, fresh seeds 2-7 (mean, range); share of revisits")
        L.append("| class | share | " + " | ".join(ARMS) + " |")
        L.append("|---|---|" + "---|" * len(ARMS))
        for k, sel in C.items():
            cells = []
            for arm in ARMS:
                a = acc(R, sel, arm, p)
                cells.append(f"{np.mean(a):.3f} ({min(a):.2f}-{max(a):.2f})" if a else "--")
            L.append(f"| {k} | {sel[ref].mean():.3f} | " + " | ".join(cells) + " |")
    # share-mix account of index 1-layer accuracy across p (does a p-invariant per-class accuracy predict a1(p)?)
    L.append("\n== index 1 layer: observed a1(p) vs a mixture prediction from p-pooled per-class accuracies "
             "(classes retrace / lap-not-retrace / neither)")
    ks = ("retrace", "lap, not retrace", "neither")
    pooled = {k: np.mean([np.mean(acc(R, C[k], "RoPE_L1", p)) for p in PS if acc(R, C[k], "RoPE_L1", p)]) for k in ks}
    L.append("  pooled per-class accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in pooled.items()))
    for p in PS:
        ref = (R["arm"] == "RoPE_L1") & (R["p"] == p) & np.isin(R["seed"], SEEDS)
        sh = {k: C[k][ref].mean() for k in ks}
        pred = sum(sh[k] * pooled[k] for k in ks)
        obs = np.mean(acc(R, C["all"], "RoPE_L1", p))
        L.append(f"  p {p}: shares " + ", ".join(f"{k} {sh[k]:.3f}" for k in ks) + f"  predicted {pred:.3f}  observed {obs:.3f}")
    # attention, last pass, p 0.5
    L.append("\n== attention at revisit queries (p 0.5, fresh seeds): per pass, same-cell obs mass h0/h1, normalised entropy")
    for arm in ARMS:
        for k in ("retrace", "neither", "wrap-only"):
            sel = C[k] & (R["arm"] == arm) & (R["p"] == 0.5) & np.isin(R["seed"], SEEDS)
            npass = int(arm[-1])
            parts = []
            for q in range(npass):
                m = R["st"][sel, q].mean(0)
                parts.append(f"p{q + 1}: same {m[0, 0]:.2f}/{m[1, 0]:.2f} ent {m[0, 2]:.2f}/{m[1, 2]:.2f} dist {m[0, 3]:.0f}/{m[1, 3]:.0f}")
            L.append(f"  {arm:10s} {k:10s} [chance {R['chance'][sel].mean():.3f}] " + " | ".join(parts))
    # probes
    L.append("\n== probes (ridge R^2, out of sample), mean over fresh seeds; residual after pass q (q0 = embedding)")
    cols = J["probe_cols"]
    for p in PS:
        for arm in ARMS:
            pr = np.array([J["probes"][f"{arm.replace('_', '_')}|{p}|{s}"] for s in SEEDS])
            m = pr.mean(0)
            L.append(f"  p {p} {arm}: " + " ".join(f"{c:>6s}" for c in cols))
            for q in range(m.shape[0]):
                L.append(f"      q{q}: " + " ".join(f"{x:6.3f}" for x in m[q]))
    txt = "\n".join(L); print(txt)
    open(f"{OUTDIR}/ring_depth_out.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
