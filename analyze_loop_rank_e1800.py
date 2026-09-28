"""Readouts for LOOP_RANK_E1800_PREREG.md: H1 with every arm retrained from scratch at 1800 epochs."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, perm2_ci, fisher_solved, mde

REPO = "/home/prashr/mapformer"; S = list(range(8)); D = f"{REPO}/runs/loop_rank_e1800/p0"
ARMS = {"A": "Vanilla", "L": "Looped", "L4": "Vanilla_L4", "C": "Vanilla_r4mi"}


def main():
    J = json.load(open(f"{REPO}/LOOP_RANK_E1800.json"))
    accT = lambda v, T: [dict((x[0], x[1]) for x in J[f"0.0|{v}|{T}"])[s] for s in S]
    solved, acc, cls, tails = {}, {}, {}, {}
    print("== run classes (SOLVED = final-5% loss < 0.05, registered) ==")
    for k, v in ARMS.items():
        cls[k] = [classify_run(torch.load(f"{D}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"]) for s in S]
        solved[k] = sum(c["registered"] == "SOLVED" for c in cls[k]); tails[k] = sorted(c["tail"] for c in cls[k])
        acc[k] = accT(v, 1024)
        print(f"  {k:3s} {v:12s} SOLVED {solved[k]}/8  acc {np.mean(acc[k]):.3f}  " +
              " ".join(f"s{s}:{c['cls'][:4]}({c['tail']:.3f})" for s, c in zip(S, cls[k])))
    print("\n== final-loss regimes (sorted final-5% loss) ==")
    for k in ARMS:
        print(f"  {k:3s} " + " ".join(f"{t:.3f}" for t in tails[k]))
    cmax = max(tails["C"])
    enter = [k for k in ("L", "L4") if any(t <= cmax for t in tails[k])]
    print(f"  C's range max {cmax:.3f}; L/L4 runs inside it: {enter or 'none'}")
    print("\n== cutoff sensitivity (solved counts) ==")
    for cut in (0.02, 0.05, 0.08):
        print(f"  < {cut}: " + "  ".join(f"{k} {sum(t < cut for t in tails[k])}/8" for k in ARMS))

    print("\n== contrasts ==")
    res = {}
    for lo, hi in (("A", "L"), ("A", "L4"), ("L", "L4"), ("L", "C"), ("L4", "C")):
        pf = fisher_solved(solved[lo], 8, solved[hi], 8); pp = perm2_p(acc[lo], acc[hi])["p"]
        ci = perm2_ci(acc[lo], acc[hi]); res[(lo, hi)] = (pf, pp)
        print(f"  {hi} - {lo}: SOLVED {solved[hi]}/8 vs {solved[lo]}/8 Fisher p {pf:.4f} | "
              f"acc {np.mean(acc[hi]) - np.mean(acc[lo]):+.3f} perm p {pp:.4f} CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
    ks = (("A", "L"), ("A", "L4"), ("L", "L4")); ps = [min(res[k]) for k in ks]
    order = np.argsort(ps); holm = np.empty(3); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (3 - rank) * ps[i])); holm[i] = run
    print("  Holm-adjusted:", " ".join(f"{n}={h:.4f}" for n, h in zip(("L-A", "L4-A", "L4-L"), holm)))

    fires = lambda lo, hi: min(res[(lo, hi)]) < 0.05
    sd = np.sqrt((np.var(acc["L"], ddof=1) + np.var(acc["A"], ddof=1)) / 2)
    M = mde(sd, 8) * np.sqrt(2); dLA = np.mean(acc["L"]) - np.mean(acc["A"])
    if solved["A"] >= 6:
        v = "RANK 2 WAS BUDGET -- plain r=2 solves at 1800 epochs"
    elif fires("A", "L") and solved["L"] >= 4:
        v = "H1 CONFIRMED"
    elif not fires("A", "L") and abs(dLA) < M and solved["L"] <= 2:
        v = "H1 REFUTED"
    else:
        v = "UNMEASURED"
    print(f"\n== REGISTERED BRANCH: {v}   (L - A acc {dLA:+.3f}, MDE ~{M:.3f})")
    print(f"== regime reading: {'WITHDRAW never-enters-r=4' if enter else 'r=4 stays in a class of its own'}")
    for T in (512, 2048):
        print(f"  T={T}: " + "  ".join(f"{k} {np.mean(accT(v, T)):.3f}" for k, v in ARMS.items()))


if __name__ == "__main__":
    main()
