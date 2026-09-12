"""SPREAD2_PREREG.md analysis. Written before any arm finished.

Reads the results json only -- no model, no GPU. Primary readout: mean accuracy over
k in {4, 16, 64} at T=1024 (k=1 excluded; the most-recent shortcut floor scales as 1/m).

    python3 -m mapformer.analyze_spread2
"""
from __future__ import annotations

import json

import numpy as np

from mapformer.ckpt_guard import REPO, load_checkpoint
from mapformer.stats_guard import paired, rule9, table

COMMON = [4, 16, 64]
SEEDS = range(8)
RUNS = REPO / "runs/spread2"
# arm -> (epochs, m, queries per token)
ARMS = {"m4_e60": (60, 4, 80_600), "m16_e240": (240, 16, 80_600),
        "m16_e60": (60, 16, 20_200), "m64_e60": (60, 64, 5_000)}
PAIR = ("m16_e240", "m4_e60")          # exposure-matched
CEILING = 0.98


def load():
    rows = {}
    for arm in ARMS:
        for s in SEEDS:
            d = RUNS / f"{arm}_s{s}"
            js = d / "VanillaEM_P0_r4_recency.json"
            if not js.exists():
                raise FileNotFoundError(js)
            r = json.load(open(js))["1024"]
            pk = {int(k): v for k, v in r["per_k"].items() if v is not None}
            ck = load_checkpoint(d / "VanillaEM_P0_r4_recency.pt")
            n_ep = len(ck.losses)
            want = ARMS[arm][0]
            if n_ep != want:
                raise AssertionError(f"{d}: {n_ep} epochs, arm {arm} expects {want}")
            rows[(arm, s)] = dict(primary=float(np.mean([pk[k] for k in COMMON if k in pk])),
                                  whole=r["acc"], loss=float(ck.losses[-1]))
    return rows


def main():
    rows = load()
    g = lambda a, k: {s: rows[(a, s)][k] for s in SEEDS}
    L = ["# SPREAD2 report (SPREAD2_PREREG.md)\n", "## Per-arm\n",
         "| arm | queries/token | primary (k in {4,16,64}) | whole trained set | final loss | at ceiling? |",
         "|---|---|---|---|---|---|"]
    ceil = {}
    for a, (ep, m, q) in ARMS.items():
        P = np.array([rows[(a, s)]["primary"] for s in SEEDS])
        ceil[a] = P.mean() >= CEILING
        L.append(f"| {a} | {q:,} | **{P.mean():.3f} +/- {P.std(ddof=1):.3f}** | "
                 f"{np.mean([rows[(a,s)]['whole'] for s in SEEDS]):.3f} | "
                 f"{np.mean([rows[(a,s)]['loss'] for s in SEEDS]):.3f} | "
                 f"{'**YES**' if ceil[a] else 'no'} |")

    design_ok = 0.60 <= np.mean([rows[("m4_e60", s)]["primary"] for s in SEEDS]) <= 0.95
    L += ["", "## Design check (read FIRST)\n",
          f"m4_e60 must land in 0.60-0.95: measured "
          f"{np.mean([rows[('m4_e60', s)]['primary'] for s in SEEDS]):.3f} -> "
          f"**{'PASS -- the exposure contrast is interpretable' if design_ok else 'FAIL -- ceilinged again; exposure contrast NOT read'}**"]

    cs = [paired(g(PAIR[0], "primary"), g(PAIR[1], "primary"), "m16_e240 - m4_e60 (exposure-matched)"),
          paired(g("m4_e60", "primary"), g("m16_e60", "primary"), "m4_e60 - m16_e60 (fixed budget)"),
          paired(g("m16_e60", "primary"), g("m64_e60", "primary"), "m16_e60 - m64_e60 (fixed budget)"),
          paired(g("m4_e60", "primary"), g("m64_e60", "primary"), "m4_e60 - m64_e60 (fixed budget)")]
    L += ["", "## Contrasts (paired by seed, n=8)\n", table(cs)]

    acc = [rows[(a, s)]["primary"] for a in ARMS for s in SEEDS]
    loss = [rows[(a, s)]["loss"] for a in ARMS for s in SEEDS]
    L += ["", "## Rule 9\n", str(rule9(acc, loss))]

    c = cs[0]
    means = {a: np.mean([rows[(a, s)]["primary"] for s in SEEDS]) for a in ARMS}
    grad = means["m4_e60"] > means["m16_e60"] > means["m64_e60"]
    p1 = abs(c.delta) < c.mde and design_ok
    p2 = c.verdict == "DETECTABLE" and c.delta < 0
    L += ["", "## Registered verdicts\n",
          f"- **S5-P1 (exposure is the currency)**: {c.delta:+.3f} (MDE {c.mde:.3f}) -> "
          f"**{'CONFIRMED' if p1 else 'NOT CONFIRMED' if design_ok else 'NOT READ -- design check failed'}**"
          + ("" if design_ok else "  (an agreement between ceilinged arms does not count)"),
          f"- **S5-P2 (token count costs beyond exposure)**: -> **{'CONFIRMED' if p2 and design_ok else 'not confirmed'}**",
          f"- **S5-P3 (gradient replicates at a lower budget)**: ordering "
          f"{means['m4_e60']:.3f} > {means['m16_e60']:.3f} > {means['m64_e60']:.3f}: "
          f"{'holds' if grad else 'FAILS'}; m4 - m64 {cs[3].delta:+.3f} ({cs[3].verdict}) -> "
          f"**{'MET' if grad and cs[3].verdict == 'DETECTABLE' else 'NOT MET'}**"]
    if not (p1 or p2) and design_ok:
        L.append("- P1 and P2 both fail: the contrast landed between them; reported as unresolved.")
    txt = "\n".join(L)
    print(txt)
    (RUNS / "REPORT.md").write_text(txt + "\n")
    json.dump({f"{a}_s{s}": r for (a, s), r in rows.items()}, open(REPO / "_SPREAD2.json", "w"))


if __name__ == "__main__":
    main()
