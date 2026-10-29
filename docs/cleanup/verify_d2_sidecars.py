"""Re-derive the checkpoint-only numbers from the CKPT_META.jsonl sidecars (no torch, no .pt).

Run before and after the D2 deletion (2026-10-29); output stored in
`docs/cleanup/verify_d2_sidecars_out.txt`. Checks:

  1. CLAUDE.md rule 2: "r = -0.996 over 57 runs, acc = 1.039 - 0.555*loss, residual sd 0.021"
     (MINIWORLD_FRESH_FINDINGS.md; commit 882d7f9, 2026-08-27). The 57-run set was never
     recorded. The 51 runs of runs/mw_grid_sweep are certain (all finished before that
     commit); the sidecars are searched for the 6-run additions (two 3-seed cells of the other
     MiniWorld dirs) that reproduce all four documented numbers to their printed precision.
  2. The map-extent row: MINIWORLD_ENDPOINTS.md's per-seed final loss and final-10% slope
     (runs/alias_follow, the +0.305 / +0.015 endpoints), and ALIASING_CONTROLLED.md's
     training-loss table (runs/alias_sweep + the runs/rope_converge anchor).
Exit status 1 if any documented value is not reproduced.
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAIL = []


def side(top):
    with open(os.path.join(REPO, "runs", top, "CKPT_META.jsonl")) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert rows[0]["n_entries"] == len(rows) - 1
    return {r["path"]: r["meta"] for r in rows[1:]}


def check(label, got, want):
    ok = got == want
    print(f"  {'OK ' if ok else 'BAD'} {label}: sidecar {got}  documented {want}")
    if not ok:
        FAIL.append(label)


def slope(L):
    tail = L[int(0.9 * len(L)):]
    return (tail[-1] - tail[0]) / max(1, len(tail) - 1)


def fit(rows):
    L = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows])
    b = np.polyfit(L, a, 1)
    res = a - np.polyval(b, L)
    return np.corrcoef(L, a)[0, 1], b[1], b[0], res.std(ddof=2)


def rule2():
    print("== 1. rule 2: accuracy (nb_acc, T=512) vs final training loss, MiniWorld")
    sweep = [(m["losses"][-1], m["results"]["512"]["nb_acc"])
             for m in side("mw_grid_sweep").values()]
    r, a, b, sd = fit(sweep)
    print(f"  mw_grid_sweep alone: n={len(sweep)} r={r:.4f} acc = {a:.3f} {b:+.3f}*loss, resid sd {sd:.4f}")
    cells = defaultdict(list)
    for top in ("mw_grid_sweep_hier", "mw_hier_ablation", "miniworld_fixed", "miniworld_fresh",
                "miniworld_oracle", "miniworld"):
        for p, m in side(top).items():
            cells[(top, os.path.basename(p))].append((m["losses"][-1], m["results"]["512"]["nb_acc"]))
    hits = []
    for c1, c2 in itertools.combinations(sorted(cells), 2):
        if len(cells[c1]) + len(cells[c2]) != 6:
            continue
        r, a, b, sd = fit(sweep + cells[c1] + cells[c2])
        if (f"{r:.3f}", f"{a:.3f}", f"{b:.3f}", f"{sd:.3f}") == ("-0.996", "1.039", "-0.555", "0.021"):
            hits.append((c1, c2, r, a, b, sd))
    print(f"  6-run additions reproducing r -0.996, intercept 1.039, slope -0.555, sd 0.021: {len(hits)}")
    for c1, c2, r, a, b, sd in hits:
        print(f"    + {c1[0]}/*/{c1[1]} + {c2[0]}/*/{c2[1]}: n=57 r={r:.4f} a={a:.4f} b={b:.4f} sd={sd:.4f}")
    check("rule 2 reproduced by at least one 57-run set", len(hits) > 0, True)
    print("  (every hit contains miniworld_oracle RoPE_oracle; which second cell was used is not "
          "recorded. Residual sd uses ddof=2.)")


ENDPOINTS = {  # MINIWORLD_ENDPOINTS.md: (final loss, slope) per (dir, seed, variant)
    ("n256_800", 0, "Vanilla"): ("0.0156", "-1.8e-04"), ("n256_800", 0, "RoPE"): ("0.3981", "-2.5e-04"),
    ("n256_800", 1, "Vanilla"): ("0.0102", "-6.0e-05"), ("n256_800", 1, "RoPE"): ("0.3440", "-1.7e-04"),
    ("n256_800", 2, "Vanilla"): ("0.0157", "-4.1e-05"), ("n256_800", 2, "RoPE"): ("0.3712", "-2.7e-04"),
    ("g16", 0, "Vanilla"): ("0.0022", "-6.0e-05"), ("g16", 0, "RoPE"): ("0.1317", "-3.3e-04"),
    ("g16", 1, "Vanilla"): ("0.0016", "-1.7e-05"), ("g16", 1, "RoPE"): ("0.1439", "-3.2e-04"),
    ("g16", 2, "Vanilla"): ("0.0188", "-3.1e-04"), ("g16", 2, "RoPE"): ("0.1498", "-4.7e-04"),
}
# ALIASING_CONTROLLED.md, training-loss table: mean [min-max] per arm, gap
ALIAS = {16: ("0.120 [0.004-0.430]", "0.438 [0.406-0.499]", "-0.318"),
         64: ("0.026 [0.006-0.040]", "0.560 [0.512-0.613]", "-0.534"),
         256: ("0.073 [0.007-0.286]", "0.664 [0.599-0.704]", "-0.591")}


def map_extent():
    print("== 2a. MINIWORLD_ENDPOINTS.md: final loss / final-10% slope (runs/alias_follow)")
    S = side("alias_follow")
    for (d, s, v), want in ENDPOINTS.items():
        L = S[f"{d}/s{s}/{v}_oracle.pt"]["losses"]
        check(f"{d} s{s} {v}", (f"{L[-1]:.4f}", f"{slope(L):.1e}"), want)
    print("== 2b. ALIASING_CONTROLLED.md: training-loss table (runs/alias_sweep, anchor runs/rope_converge)")
    A, R = side("alias_sweep"), side("rope_converge")

    def loss(nobs, s, v):
        k = f"n{nobs}/s{s}/{v}_oracle.pt"
        if k in A:
            return A[k]["losses"][-1]
        if nobs == 16 and f"s{s}/{v}_oracle.pt" in R:
            return R[f"s{s}/{v}_oracle.pt"]["losses"][-1]
        return None
    for nobs, want in ALIAS.items():
        vl, rl, gap = [], [], []
        for s in range(5):
            a, b = loss(nobs, s, "Vanilla"), loss(nobs, s, "RoPE")
            if a is not None:
                vl.append(a)
            if b is not None:
                rl.append(b)
            if a is not None and b is not None:
                gap.append(a - b)
        f = lambda x: f"{np.mean(x):.3f} [{min(x):.3f}-{max(x):.3f}]"
        check(f"n_obs {nobs}", (f(vl), f(rl), f"{np.mean(gap):+.3f}"), want)


if __name__ == "__main__":
    rule2()
    map_extent()
    print(f"\n{'ALL REPRODUCED' if not FAIL else 'NOT REPRODUCED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
