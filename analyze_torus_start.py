"""Registered analysis for TORUS_START_PREREG.md (TS1-TS4, then declared secondaries). CPU.
  python3 -m mapformer.analyze_torus_start [--root runs/torus_start/p0] [--seeds 1200-1207] [--no-json]
"""
import argparse, json

import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
ARMS = ("SO4", "AO4", "AO4_high", "SO4_high", "AO4_mid")


def compare(R, seeds, x, ref):
    n = len(seeds)
    a = np.array([R[x, s]["acc"] for s in seeds]); r = np.array([R[ref, s]["acc"] for s in seeds])
    ka = sum(R[x, s]["solved"] for s in seeds); kr = sum(R[ref, s]["solved"] for s in seeds)
    pf = fisher_solved(ka, n, kr, n); pp = perm2_p(r, a)["p"]; d = float(a.mean() - r.mean())
    ci = perm2_ci(r, a, step=0.005, grid=(-1.0, 1.0))
    if (pf < 0.05 and ka > kr) or (pp < 0.05 and d >= 0.05):
        lab = "BETTER"
    elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
        lab = "WORSE"
    elif abs(ka - kr) <= 1 and abs(d) <= 0.01:
        lab = "NO EFFECT DETECTED"
    else:
        lab = "UNMEASURED"
    return lab, (f"{x} {a.mean():.4f} SOLVED {ka}/{n} vs {ref} {r.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} [95% CI "
                 f"{ci['lo']:+.3f}, {ci['hi']:+.3f}] (perm p {pp:.4f}); Fisher p {pf:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/torus_start/p0")
    ap.add_argument("--seeds", default="1200-1207")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {}
    for x in ARMS:
        for s in seeds:
            d = f"{a.root}/{x}_s{s}"
            ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{x}.pt", map_location="cpu", weights_only=False)
            c = classify_run(b["losses"])
            R[x, s] = {"acc": ev["eval"]["1024"]["acc"], "acc2": ev["eval"]["2048"]["acc"], "solved": c["registered"] == "SOLVED",
                       "cls": c["cls"], "wrap": ev["wrap"][1], "strata": ev["strata"], "auc": float(np.mean(b["losses"]))}
    print("== per seed: accuracy T = 1024 (held-out map) and class ==")
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{x} {R[x, s]['acc']:.4f} {R[x, s]['cls'][:4]}" for x in ARMS))
    for x in ARMS:
        print(f"{x:9s}: mean {np.mean([R[x, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[x, s]['solved'] for s in seeds)}/{n}")
    k_so = sum(R["SO4", s]["solved"] for s in seeds)
    V = {}
    k_soh = sum(R["SO4_high", s]["solved"] for s in seeds)
    # Amendment 1: every contrast gated (TS1, TS2, TS4 on SO4 SOLVED >= 4; TS3 on SO4_high SOLVED >= 4)
    names = {"TS1": ("AO4_high", "AO4", {"BETTER": "RESCUES ON THE TORUS"}, ("SO4", k_so)),
             "TS2": ("AO4_high", "AO4_mid", {"BETTER": "SUM PAST PI BEATS A LARGE SUB-PI START"}, ("SO4", k_so)),
             "TS3": ("AO4_high", "SO4_high", {"WORSE": "SIGN COST AT THE LARGE START"}, ("SO4_high", k_soh)),
             "TS4": ("AO4", "SO4", {"WORSE": "SIGNED LEARNS BETTER"}, ("SO4", k_so))}
    for key, (x, ref, ren, (garm, gk)) in names.items():
        lab, txt = compare(R, seeds, x, ref); lab = ren.get(lab, lab)
        if gk < 4:
            lab = f"UNREADABLE ({garm} SOLVED {gk} < 4)"
        V[key] = lab
        print(f"{key} ({x} vs {ref}): {lab} -- {txt}")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/TORUS_START_VERDICTS.json", "w"), indent=2)
    print("\n== secondaries (declared; no verdict) ==")
    for x in ARMS:
        st = [R[x, s]["strata"] for s in seeds]
        sm = lambda k: np.mean([t[k]["acc"] for t in st if t[k]["acc"] is not None])
        print(f"{x:9s}: wrap share {np.mean([R[x, s]['wrap'] for s in seeds]):.2f}; strata lag<128 {sm('plain_lag<128'):.3f}, "
              f"lag>=128 {sm('plain_lag>=128'):.3f}, wrap-only {sm('wrap'):.3f}; acc 2048 {np.mean([R[x, s]['acc2'] for s in seeds]):.4f}; "
              f"loss AUC {np.mean([R[x, s]['auc'] for s in seeds]):.3f}")


if __name__ == "__main__":
    main()
