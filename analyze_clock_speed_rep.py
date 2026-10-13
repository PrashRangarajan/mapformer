"""Registered analysis for CLOCK_SPEED_REP_PREREG.md.  python3 -m mapformer.analyze_clock_speed_rep (from /home/prashr)."""
import json, re
import numpy as np
import torch

from mapformer.analyze_clock_speed import branch, first

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/clock_speed_rep/p0"
ARMS = ("Signed_r4", "Abs_r4"); SEEDS = list(range(230, 246)); T = 1024; K_MAX = 128; EPOCHS = 300


def load(runs, arm, sd):
    b = torch.load(f"{runs}/{arm}_s{sd}/{arm}_recency.pt", map_location="cpu", weights_only=False)
    res = json.load(open(f"{runs}/{arm}_s{sd}/{arm}_recency.json")); L = np.asarray(b["losses"], float)
    assert b["variant"] == arm and b["seed"] == sd and b["k_max"] == K_MAX and b["min_gap"] == K_MAX and len(L) == EPOCHS, (arm, sd)
    hdr = re.search(r"T=(\d+) k_max=(\d+) min_gap=(\d+)", open(f"{runs}/{arm}_s{sd}.log").read())
    assert hdr and tuple(int(x) for x in hdr.groups()) == (T, K_MAX, K_MAX), (arm, sd)
    return L, res


def main():
    auc = {}
    print(f"== CLOCK_SPEED_REP: fresh seeds 230-245, Signed_r4 vs Abs_r4, AUC = mean training loss over {EPOCHS} epochs ==")
    for arm in ARMS:
        auc[arm] = []; rows = []
        for sd in SEEDS:
            L, res = load(R, arm, sd); auc[arm].append(float(L.mean()))
            rows.append((first(L, 2.0), first(L, 1.0), L[99], L[199], L[299], res[str(T)]["acc"], res[str(2 * T)]["acc"]))
        r = np.array(rows, float)
        print(f"  {arm:10s} AUC {np.mean(auc[arm]):.3f} +/- {np.std(auc[arm]):.3f} | epoch to <2.0 / <1.0 median {np.median(r[:, 0]):.0f} / "
              f"{np.median(r[:, 1]):.0f} | loss 100/200/300 {np.median(r[:, 2]):.2f}/{np.median(r[:, 3]):.2f}/{np.median(r[:, 4]):.2f} | "
              f"acc {r[:, 5].mean():.3f} (2T {r[:, 6].mean():.3f})")
    print(f"  REGISTERED S1 (Abs - Signed, n = 16 fresh seeds): {branch(auc['Signed_r4'], auc['Abs_r4'])}")
    try:                                                       # post hoc, labelled: pooled with CLOCK_SPEED's 16 seeds
        V = json.load(open(f"{REPO}/CLOCK_SPEED_VERDICTS.json"))["auc"]
        print(f"  POST HOC pooled 32 seeds (CLOCK_SPEED 210-225 + these): {branch(V['Signed_r4'] + auc['Signed_r4'], V['Abs_r4'] + auc['Abs_r4'])}")
    except (FileNotFoundError, KeyError) as e:
        print(f"  pooled reading unavailable ({e})")
    json.dump({"auc": auc}, open(f"{REPO}/CLOCK_SPEED_REP_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
