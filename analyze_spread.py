"""SPREAD_PREREG.md analysis. Written before any SPREAD checkpoint existed.

Primary readout: mean accuracy over k in {4, 16, 64} at T=1024, from `probe_anatomy` on
held-out episodes (the same 256-episode stream every arm sees, and a much larger per-k sample
than the trainer's own eval, which gives ~7 queries per k at m=64). The trainer's per-k numbers
are reported beside it.

    python3 -m mapformer.analyze_spread
"""
from __future__ import annotations

import json

import numpy as np
import torch

from mapformer.ckpt_guard import REPO
from mapformer.probe_anatomy import anatomy
from mapformer.probe_rewind import load_model
from mapformer.stats_guard import paired, table

K4 = [1, 4, 16, 64]
K16 = [1, 2, 3, 4, 6, 8, 11, 16, 22, 26, 32, 38, 45, 52, 58, 64]
COMMON = [4, 16, 64]
SEEDS = range(8)
# arm -> (run dir, k_set for the env, exposure in queries per token over training)
ARMS = {
    "m4_e300":   ("runs/spread/m4_e300_s{s}",  K4,   403_000),
    "m16_e300":  ("runs/spread/m16_e300_s{s}", K16,  101_000),
    "m16_e1200": ("runs/spread/m16_e1200_s{s}", K16, 403_000),
    "m64_e1200": ("runs/spread/m64_e1200_s{s}", None, 101_000),
    "m64_e300":  ("runs/dof/recency/VanillaEM_P0_r4_s{s}", None, 25_000),
}
PAIRS = [("m16_e1200", "m4_e300"), ("m64_e1200", "m16_e300")]


def rewind_fraction(an, ks):
    """Fraction of the arm's trained query tokens carrying a wrapped rewind (peak or trough)."""
    S, S0 = np.array(an["sel"]), np.array(an["sel0"])
    M, M0 = np.array(an["selmin"]), np.array(an["selmin0"])
    hit = [max(S[k - 1].max() - S0[k - 1].max(), M[k - 1].max() - M0[k - 1].max()) >= 0.5
           for k in ks if an["n"][k - 1] > 0]
    return float(np.mean(hit)), len(hit)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = {}
    for arm, (tmpl, kset, exp) in ARMS.items():
        for s in SEEDS:
            d = REPO / tmpl.format(s=s)
            pt, js = d / "VanillaEM_P0_r4_recency.pt", d / "VanillaEM_P0_r4_recency.json"
            if not pt.exists():
                raise FileNotFoundError(f"missing {pt}")
            res = json.load(open(js))
            over = dict(k_set=kset) if kset else {}
            m, ck, env = load_model(pt, "VanillaEM_P0_r4", **over)
            # the env here is built from the ARM definition, so check it against what the run
            # actually trained on: a mismatched launch would otherwise be read silently.
            stored = ck.config.get("k_set")
            stored = [int(v) for v in stored.split(",")] if stored else None
            if stored != kset:
                raise AssertionError(f"{d}: checkpoint k_set {stored} != arm {arm}'s {kset}")
            n_ep = len(ck.losses)
            want_ep = 1200 if "e1200" in arm else 300
            if n_ep != want_ep:
                raise AssertionError(f"{d}: {n_ep} epochs trained, arm {arm} expects {want_ep}")
            an = anatomy(m, env, s, device=dev)
            trained = kset or list(range(1, 65))
            frac, ntok = rewind_fraction(an, trained)
            rows[(arm, s)] = dict(
                arm=arm, seed=s, exposure=exp,
                primary=float(np.mean([an["acc"][k - 1] for k in COMMON])),
                per_k={k: an["acc"][k - 1] for k in COMMON},
                trainer_acc=res["1024"]["acc"], trainer_acc2048=res["2048"]["acc"],
                full_set=float(np.average([an["acc"][k - 1] for k in trained],
                                          weights=[an["n"][k - 1] for k in trained])),
                rewind_frac=frac, n_tokens=ntok,
                final_loss=float(ck.losses[-1]), acc_k=an["acc"], n_k=an["n"])
            print(arm, s, "primary", round(rows[(arm, s)]["primary"], 3),
                  "rewind frac", round(frac, 3), flush=True)

    g = lambda arm, key: {s: rows[(arm, s)][key] for s in SEEDS}
    L = ["# SPREAD report (SPREAD_PREREG.md)\n", "## Per-arm\n",
         "| arm | queries per token | primary (k in {4,16,64}) | k=4 | k=16 | k=64 | whole trained set | rewind fraction | final loss |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        P = np.array([rows[(arm, s)]["primary"] for s in SEEDS])
        pk = {k: np.mean([rows[(arm, s)]["per_k"][k] for s in SEEDS]) for k in COMMON}
        L.append(f"| {arm} | {ARMS[arm][2]:,} | **{P.mean():.3f} +/- {P.std(ddof=1):.3f}** | "
                 + " | ".join(f"{pk[k]:.3f}" for k in COMMON)
                 + f" | {np.mean([rows[(arm, s)]['full_set'] for s in SEEDS]):.3f} | "
                 f"{np.mean([rows[(arm, s)]['rewind_frac'] for s in SEEDS]):.3f} | "
                 f"{np.mean([rows[(arm, s)]['final_loss'] for s in SEEDS]):.3f} |")

    cs = [paired(g(a, "primary"), g(b, "primary"), f"{a} - {b} (exposure-matched)") for a, b in PAIRS]
    cs += [paired(g("m4_e300", "primary"), g("m16_e300", "primary"), "m4_e300 - m16_e300 (fixed budget)"),
           paired(g("m16_e300", "primary"), g("m64_e300", "primary"), "m16_e300 - m64_e300 (fixed budget)"),
           paired(g("m4_e300", "primary"), g("m64_e300", "primary"), "m4_e300 - m64_e300 (fixed budget)")]
    L += ["", "## Contrasts (primary readout, paired by seed, n=8)\n", table(cs)]

    p1 = [c for c in cs[:2]]
    conf = all(c.verdict != "DETECTABLE" for c in p1)
    below = p1[0].delta < 0 and p1[0].verdict == "DETECTABLE"
    L += ["", "## Registered verdicts\n",
          f"- **S4-P1 (exposure)**: matched pairs {p1[0].delta:+.3f} (MDE {p1[0].mde:.3f}) and "
          f"{p1[1].delta:+.3f} (MDE {p1[1].mde:.3f}) -> "
          f"**{'CONFIRMED' if conf else 'REFUTED -- window, not total exposure' if below else 'SPLIT'}**",
          f"- **S4-P2 (spread gradient)**: ordering m4 {np.mean([rows[('m4_e300', s)]['primary'] for s in SEEDS]):.3f} "
          f"> m16 {np.mean([rows[('m16_e300', s)]['primary'] for s in SEEDS]):.3f} "
          f"> m64 {np.mean([rows[('m64_e300', s)]['primary'] for s in SEEDS]):.3f}: "
          f"{'holds' if np.mean([rows[('m4_e300', s)]['primary'] for s in SEEDS]) > np.mean([rows[('m16_e300', s)]['primary'] for s in SEEDS]) > np.mean([rows[('m64_e300', s)]['primary'] for s in SEEDS]) else 'FAILS'}; "
          f"m4 - m64 {cs[4].delta:+.3f} ({cs[4].verdict}) -> "
          f"**{'MET' if cs[4].verdict == 'DETECTABLE' else 'NOT MET -- P1 not interpreted'}**"]
    x = np.array([rows[k]["rewind_frac"] for k in rows])
    y = np.array([rows[k]["primary"] for k in rows])
    r = float(np.corrcoef(x, y)[0, 1])
    L.append(f"- **S4-P3 (mechanism)**: r(rewind fraction, primary) = {r:+.3f} over {len(x)} "
             f"arm-seed cells (>= 0.70 predicted) -> **{'MET' if r >= 0.7 else 'NOT MET'}**")
    det = (REPO / "runs/spread/DETERMINISM.txt")
    L += ["", "## Determinism re-check (licenses reuse of stored m64_e300)\n", "```",
          det.read_text().strip() if det.exists() else "MISSING", "```"]
    txt = "\n".join(L)
    print(txt)
    (REPO / "runs/spread/S4_report.md").write_text(txt + "\n")
    json.dump({f"{a}_s{s}": r_ for (a, s), r_ in rows.items()}, open(REPO / "_SPREAD.json", "w"))


if __name__ == "__main__":
    main()
