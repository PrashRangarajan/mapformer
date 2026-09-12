"""PAIRORIGIN_PREREG.md analysis. Written before any arm finished.

    python3 -m mapformer.analyze_pairorigin
"""
from __future__ import annotations

import json

import numpy as np
import torch

from mapformer.ckpt_guard import REPO, load_checkpoint
from mapformer.stats_guard import paired, rule9, table

ARMS = ["EMPair_r4", "VanillaEM_P0_r4", "Vanilla_r4"]
SEEDS = range(8)
RUNS = REPO / "runs/pairorigin"


def load():
    rows = {}
    for a in ARMS:
        for s in SEEDS:
            d = RUNS / f"{a}_s{s}"
            pt, js = d / f"{a}_recency.pt", d / f"{a}_recency.json"
            if not js.exists():
                raise FileNotFoundError(js)
            r = json.load(open(js))
            ck = load_checkpoint(pt)
            rows[(a, s)] = dict(acc=r["1024"]["acc"], acc2048=r["2048"]["acc"],
                                per_k=r["1024"]["per_k"], loss=float(ck.losses[-1]))
    return rows


def rewind_and_phase(dev="cuda:0"):
    """P4 and manipulation check 3, on the EM arms only."""
    from mapformer.probe_anatomy import anatomy
    from mapformer.probe_phase_spread import pair_phases, stats
    from mapformer.probe_rewind import load_model
    out = {}
    for a in ["EMPair_r4", "VanillaEM_P0_r4"]:
        fr, sp = [], []
        for s in SEEDS:
            pt = RUNS / f"{a}_s{s}" / f"{a}_recency.pt"
            m, ck, env = load_model(pt, a)
            an = anatomy(m, env, s, device=dev)
            S, S0 = np.array(an["sel"]), np.array(an["sel0"])
            M, M0 = np.array(an["selmin"]), np.array(an["selmin0"])
            solved = [k for k in range(8, 65) if an["n"][k - 1] and an["acc"][k - 1] >= 0.9]
            if solved:
                fr.append(float(np.mean([max(S[k - 1].max() - S0[k - 1].max(),
                                             M[k - 1].max() - M0[k - 1].max()) >= 0.5
                                         for k in solved])))
            sp.append(stats(*pair_phases(pt, a, s))["circ_sd"])
        out[a] = dict(rewind_frac=fr, phase_sd=sp)
    return out


def main():
    rows = load()
    g = lambda a, k: {s: rows[(a, s)][k] for s in SEEDS}
    L = ["# PAIRORIGIN report (PAIRORIGIN_PREREG.md)\n", "## Per-arm\n",
         "| arm | acc T=1024 | acc T=2048 | final loss |", "|---|---|---|---|"]
    for a in ARMS:
        A = np.array([rows[(a, s)]["acc"] for s in SEEDS])
        L.append(f"| {a} | **{A.mean():.3f} +/- {A.std(ddof=1):.3f}** | "
                 f"{np.mean([rows[(a, s)]['acc2048'] for s in SEEDS]):.3f} | "
                 f"{np.mean([rows[(a, s)]['loss'] for s in SEEDS]):.3f} |")

    cs = [paired(g("EMPair_r4", "acc"), g("VanillaEM_P0_r4", "acc"), "EMPair - P0 (P1/P3)"),
          paired(g("EMPair_r4", "acc"), g("Vanilla_r4", "acc"), "EMPair - WM (P2)"),
          paired(g("VanillaEM_P0_r4", "acc"), g("Vanilla_r4", "acc"), "P0 - WM (the gap being closed)"),
          paired(g("EMPair_r4", "acc2048"), g("VanillaEM_P0_r4", "acc2048"), "EMPair - P0 at T=2048")]
    L += ["", "## Contrasts (paired by seed, n=8)\n", table(cs)]

    acc = [rows[(a, s)]["acc"] for a in ARMS for s in SEEDS]
    loss = [rows[(a, s)]["loss"] for a in ARMS for s in SEEDS]
    L += ["", "## Rule 9\n", str(rule9(acc, loss))]

    c1, c2 = cs[0], cs[1]
    p1 = c1.delta >= 0.20 and c1.verdict == "DETECTABLE"
    p3 = abs(c1.delta) < c1.mde
    L += ["", "## Registered verdicts\n",
          f"- **P1 (kernel sharing)**: EMPair - P0 = {c1.delta:+.3f} (MDE {c1.mde:.3f}, "
          f"needs >= +0.20 AND detectable) -> **{'CONFIRMED' if p1 else 'NOT CONFIRMED'}**",
          f"- **P2 (recovery to WM)**: EMPair - WM = {c2.delta:+.3f} (MDE {c2.mde:.3f}) -> "
          f"**{'MET (within MDE)' if abs(c2.delta) < c2.mde else 'NOT MET'}**",
          f"- **P3 (per-token search)**: |EMPair - P0| below MDE -> **{'CONFIRMED' if p3 else 'NOT CONFIRMED'}**"]
    if not p1 and not p3:
        L.append("- P1 and P3 both fail: the contrast landed between them; reported as unresolved.")

    try:
        rp = rewind_and_phase()
        L += ["", "## P4 (mechanism) and manipulation check 3\n",
              "| arm | rewind fraction of solved cells | phase spread across pairs |", "|---|---|---|"]
        for a, v in rp.items():
            L.append(f"| {a} | {np.mean(v['rewind_frac']):.3f} | {np.mean(v['phase_sd']):.3f} |")
        d = np.mean(rp["EMPair_r4"]["rewind_frac"]) - np.mean(rp["VanillaEM_P0_r4"]["rewind_frac"])
        L.append(f"\nP4: EMPair's solved cells use the per-token rewind {d:+.3f} vs P0. Lower is what "
                 "P1 predicts (per-pair freedom should retrieve WITHOUT moving the query token).")
        L.append(f"Check 3: EMPair phase spread must be > 0 (P0 is 0.000 by construction) -> "
                 f"{'PASS' if np.mean(rp['EMPair_r4']['phase_sd']) > 0.01 else 'FAIL'}")
    except Exception as e:                       # analysis of the levels must not be lost to it
        L.append(f"\nP4/check-3 probe failed: {type(e).__name__}: {e}")

    man = RUNS / "MANIPULATION.txt"
    L += ["", "## Manipulation checks 1-2 (from the batch)\n", "```",
          man.read_text().strip() if man.exists() else "MISSING", "```"]
    txt = "\n".join(L)
    print(txt)
    (RUNS / "REPORT.md").write_text(txt + "\n")
    json.dump({f"{a}_s{s}": r for (a, s), r in rows.items()}, open(REPO / "_PAIRORIGIN.json", "w"))


if __name__ == "__main__":
    main()
