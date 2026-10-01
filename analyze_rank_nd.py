"""Readouts for RANK_ND_PREREG.md: does the hard per-head rank track the torus's dimension?"""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/rank_nd"; S = list(range(8))
CELLS = {"A2": (2, "Vanilla_r2ph"), "B2": (2, "Vanilla_r3ph"), "A3": (3, "Vanilla_r3ph"), "B3": (3, "Vanilla_r4ph")}


def main():
    J = json.load(open(f"{REPO}/RANK_ND.json"))
    solved, acc = {}, {}
    print("== cells (SOLVED = final-5% loss < 0.05) ==")
    for k, (D, v) in CELLS.items():
        cl = [classify_run(torch.load(f"{R}/D{D}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"]) for s in S]
        solved[k] = sum(c["registered"] == "SOLVED" for c in cl)
        acc[k] = [J[f"D{D}"]["acc"][v]["1024"][str(s)] for s in S]
        a2048 = np.mean([J[f"D{D}"]["acc"][v]["2048"][str(s)] for s in S])
        print(f"  {k} D={D} {v:13s} SOLVED {solved[k]}/8  acc@1024 {np.mean(acc[k]):.3f} +/- {np.std(acc[k], ddof=1):.3f}"
              f"  [2048 {a2048:.3f}]  " + " ".join(f"{c['cls'][:4]}({c['tail']:.3f})" for c in cl))
    fires = {}
    for lo, hi in (("A2", "B2"), ("A3", "B3")):
        pf = fisher_solved(solved[lo], 8, solved[hi], 8); pp = perm2_p(acc[lo], acc[hi])["p"]
        d = np.mean(acc[hi]) - np.mean(acc[lo])
        fires[hi] = (pf < 0.05 and solved[hi] > solved[lo]) or (pp < 0.05 and d > 0)
        print(f"\n  {hi} - {lo}: SOLVED {solved[hi]}/8 vs {solved[lo]}/8 Fisher p {pf:.4f} | acc {d:+.3f} perm p {pp:.4f}"
              f" | {'FIRES' if fires[hi] else 'does not fire'}")
    if solved["A2"] >= 6:
        v = "CONTROL FAILED (rank = D is not hard here even in 2D)"
    elif solved["A2"] <= 2 and solved["A3"] <= 2 and fires["B3"]:
        v = "THRESHOLD TRACKS DIMENSION"
    elif solved["A2"] <= 2 and solved["A3"] >= 6 and not fires["B3"]:
        v = "RANK 3 IS ENOUGH"
    else:
        v = "no registered branch -- reported as it falls"
    print(f"\n== REGISTERED: {v}")


if __name__ == "__main__":
    main()
