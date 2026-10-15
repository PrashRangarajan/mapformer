"""Registered analysis for GAUGE_PREREG.md (G1, G2, reading; then declared secondaries). CPU.
  python3 -m mapformer.analyze_gauge [--root runs/gauge/p0] [--seeds 1300-1307] [--no-json]
"""
import argparse, json

import numpy as np
import torch

from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, action_angles
from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
CELLS = ("SO_m", "AO_m", "SO_p", "AO_p", "RoPE_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/gauge/p0")
    ap.add_argument("--seeds", default="1300-1307")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {}
    for c in CELLS:
        arm = c.split("_")[0]
        for s in seeds:
            d = f"{a.root}/{c}_s{s}"
            ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
            k = classify_run(b["losses"])
            R[c, s] = {"acc": ev["eval"]["128"]["acc"], "acc4": ev["eval"]["512"]["acc"], "solved": k["registered"] == "SOLVED",
                       "cls": k["cls"], "auc": float(np.mean(b["losses"])), "blob": b}
    print("== per seed: accuracy T = 128 and class ==")
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{c} {R[c, s]['acc']:.4f} {R[c, s]['cls'][:4]}" for c in CELLS))
    for c in CELLS:
        print(f"{c:7s}: mean {np.mean([R[c, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[c, s]['solved'] for s in seeds)}/{n}, "
              f"acc 4T {np.mean([R[c, s]['acc4'] for s in seeds]):.4f}, loss AUC {np.mean([R[c, s]['auc'] for s in seeds]):.3f}")
    V = {}
    for key, x, ref in (("G1", "AO_p", "SO_p"), ("G2", "AO_m", "SO_m")):
        av = np.array([R[x, s]["acc"] for s in seeds]); rv = np.array([R[ref, s]["acc"] for s in seeds])
        ka = sum(R[x, s]["solved"] for s in seeds); kr = sum(R[ref, s]["solved"] for s in seeds)
        pf = fisher_solved(ka, n, kr, n); pp = perm2_p(rv, av)["p"]; d = float(av.mean() - rv.mean())
        ci = perm2_ci(rv, av, step=0.005, grid=(-1.0, 1.0))
        if kr < 4:
            lab = f"UNREADABLE ({ref} SOLVED {kr} < 4)"
        elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
            lab = "SIGNED LEARNS BETTER"
        elif ka >= kr - 1 and av.mean() >= rv.mean() - 0.01:
            lab = "NO SIGN COST"
        else:
            lab = "UNMEASURED"
        V[key] = lab
        print(f"{key} ({x} vs {ref}): {lab} -- {av.mean():.4f} SOLVED {ka}/{n} vs {rv.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} "
              f"[95% CI {ci['lo']:+.3f}, {ci['hi']:+.3f}] (perm p {pp:.4f}); Fisher p {pf:.4f}")
    V["reading"] = ("GAUGE PREDICTION CONFIRMED" if (V["G1"] == "NO SIGN COST" and V["G2"] == "SIGNED LEARNS BETTER") else
                    ("REFUTED" if V["G1"] == "SIGNED LEARNS BETTER" else "AS FALLS"))
    print(f"Reading: {V['reading']}")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/GAUGE_VERDICTS.json", "w"), indent=2)
    print("\n== secondaries (declared; no verdict) ==")
    for c in ("SO_p", "SO_m"):
        same = []
        for s in seeds:
            b = R[c, s]["blob"]
            m = MapFormerWM_SignedAO_r4(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
                                        action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
            m.load_state_dict(b["model_state_dict"]); A = action_angles(m).double().numpy()
            sel = np.abs(np.angle(np.exp(1j * A[0]))) > 0.05
            same.append(float((np.sign(A[0]) == np.sign(A[1]))[sel].mean()) if sel.any() else float("nan"))
        print(f"{c}: share of channels whose two move angles have the SAME sign at the end {np.nanmean(same):.2f} "
              f"(per seed {' '.join(f'{v:.2f}' for v in same)})")


if __name__ == "__main__":
    main()
