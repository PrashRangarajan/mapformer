"""Tables from torus_depth.py's rows (POST HOC). Accuracy by stratum / retrace / runs spanned / gap, per arm (mean over
the 4 seeds of per-seed accuracies, range), attention summaries per pass, and probe R^2.
  python3 summarize_torus.py   (reads $DEPTH_SCRATCH/torus_depth_*.npz and out/torus_depth_*.json)
writes out/torus_depth_out.txt"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import OUTDIR  # noqa: E402

S = os.environ.get("DEPTH_SCRATCH", OUTDIR)
ARMS = ("L1", "L4", "LOOP", "PATH")
SEEDS = (500, 501, 502, 503)


def classes(R):
    short = ~R["wrap"] & (R["lag"] < 128); long_ = ~R["wrap"] & (R["lag"] >= 128)
    out = {"all": np.ones(len(R["c"]), bool), "short": short, "long": long_, "wrap-only": R["wrap"],
           "short, retrace": short & R["retr"], "short, not retrace": short & ~R["retr"]}
    for lo, hi in [(2, 3), (3, 5), (5, 9), (9, 17), (17, 10 ** 6)]:
        out[f"short non-retrace, runs {lo}-{hi - 1 if hi < 10 ** 6 else '+'}"] = short & ~R["retr"] & \
            (R["nruns"] >= lo) & (R["nruns"] < hi)
    for lo, hi in [(2, 8), (8, 32), (32, 128), (128, 512), (512, 1025)]:
        out[f"plain gap {lo}-{hi - 1}"] = ~R["wrap"] & (R["lag"] >= lo) & (R["lag"] < hi)
    for lo, hi in [(64, 128), (128, 512), (512, 1025)]:
        out[f"wrap-only gap {lo}-{hi - 1}"] = R["wrap"] & (R["lag"] >= lo) & (R["lag"] < hi)
    return out


def main():
    L = []
    D = {a: dict(np.load(f"{S}/torus_depth_{a}.npz")) for a in ARMS if os.path.exists(f"{S}/torus_depth_{a}.npz")}
    J = {a: json.load(open(f"{OUTDIR}/torus_depth_{a}.json")) for a in D}
    R0 = D[next(iter(D))]
    C0 = classes(R0)
    L.append("== accuracy by revisit class (mean of per-seed accuracies, n = 4 seeds; share = fraction of revisits)")
    L.append("| class | share | n (seed 500) | " + " | ".join(D) + " |")
    L.append("|---|---|---|" + "---|" * len(D))
    for k in C0:
        cells = []
        for a, R in D.items():
            C = classes(R)
            accs = [R["c"][C[k] & (R["seed"] == s)].mean() for s in SEEDS if (C[k] & (R["seed"] == s)).any()]
            cells.append(f"{np.mean(accs):.3f} ({min(accs):.2f}-{max(accs):.2f})" if accs else "--")
        L.append(f"| {k} | {C0[k].mean():.3f} | {int((C0[k] & (R0['seed'] == 500)).sum())} | " + " | ".join(cells) + " |")
    # attention
    L.append("\n== attention at revisit queries, per pass: mass on earlier same-cell observation tokens (same), on the last "
             "visit's two tokens (last), normalised entropy (ent), mean distance in moves (dist); uniform-attention chance "
             "of 'same' in brackets; means over seeds and queries; h0 / h1")
    for a, R in D.items():
        C = classes(R); st = R["st"]                       # (n, P, H, 4)
        for k in ("short, retrace", "short, not retrace", "long", "wrap-only"):
            sel = C[k]
            parts = []
            for p in range(st.shape[1]):
                m = st[sel, p].mean(0)                       # (H, 4)
                parts.append(f"p{p + 1}: same {m[0, 0]:.2f}/{m[1, 0]:.2f} last {m[0, 1]:.2f}/{m[1, 1]:.2f} "
                             f"ent {m[0, 2]:.2f}/{m[1, 2]:.2f} dist {m[0, 3]:.0f}/{m[1, 3]:.0f}")
            L.append(f"  {a:5s} {k:20s} [chance {R['chance'][sel].mean():.3f}]  " + " | ".join(parts))
    # max-over-heads same-cell mass in the LAST pass, by correctness (does retrieval track accuracy?)
    L.append("\n== last pass, max over heads of same-cell mass, split by correct / wrong prediction")
    for a, R in D.items():
        C = classes(R); m = R["st"][:, -1, :, 0].max(1)
        for k in ("short, retrace", "short, not retrace", "long", "wrap-only"):
            sel = C[k]
            L.append(f"  {a:5s} {k:20s} correct {m[sel & R['c']].mean():.2f} (n {int((sel & R['c']).sum())})  wrong "
                     f"{m[sel & ~R['c']].mean() if (sel & ~R['c']).any() else float('nan'):.2f} (n {int((sel & ~R['c']).sum())})")
    # probes
    L.append("\n== linear probes (ridge, out-of-sample R^2; train walks 0-29, test 30-39), mean over seeds; rows = residual "
             "after pass p (p0 = embedding); x and y averaged")
    for a in D:
        cols = J[a]["probe_cols"]; pr = np.array([J[a]["probe"][str(s)] for s in SEEDS])   # (S, P, ncol)
        m = pr.mean(0)
        names = ["P (unwrapped)", "cos/sin 2piP/64"] + [f"d{w}" for w in (4, 16, 64, 256)]
        idx = [[0, 1], [2, 3, 4, 5]] + [[6 + 2 * i, 7 + 2 * i] for i in range(4)]
        L.append(f"  {a}: " + "  ".join(names))
        for p in range(m.shape[0]):
            L.append(f"    p{p}: " + "  ".join(f"{m[p, ix].mean():6.3f}" for ix in idx))
    txt = "\n".join(L); print(txt)
    open(f"{OUTDIR}/torus_depth_out.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
