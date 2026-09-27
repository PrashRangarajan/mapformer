"""Readouts for LOOP_RANK_PREREG.md (H1): does the loop recover rank 2's search failure?"""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, perm2_ci, fisher_solved

REPO = "/home/prashr/mapformer"; S = list(range(8))
ARMS = {"A": ("Vanilla", "rank_mi", "RANK_MI"), "L": ("Looped", "loop_rank", "LOOP_RANK"),
        "L4": ("Vanilla_L4", "loop_rank", "LOOP_RANK"), "C": ("Vanilla_r4mi", "rank_mi", "RANK_MI")}


def main():
    solved, acc, cls = {}, {}, {}
    print("== run classes (SOLVED = final-5% loss < 0.05) ==")
    for k, (v, d, j) in ARMS.items():
        cls[k] = [classify_run(torch.load(f"{REPO}/runs/{d}/p0/{v}_s{s}/{v}.pt", map_location="cpu",
                                          weights_only=False)["losses"]) for s in S]
        solved[k] = sum(c["registered"] == "SOLVED" for c in cls[k])
        J = json.load(open(f"{REPO}/{j}.json")); dd = {x[0]: x for x in J[f"0.0|{v}|1024"]}
        acc[k] = [dd[s][1] for s in S]
        print(f"  {k:3s} {v:12s} SOLVED {solved[k]}/8  acc {np.mean(acc[k]):.3f}  " +
              " ".join(f"s{s}:{c['registered'][:4]}({c['tail']:.3f})" for s, c in zip(S, cls[k])))
    x = np.array(torch.load(f"{REPO}/runs/loop_rank_repro/p0/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
    y = np.array(torch.load(f"{REPO}/runs/rank_mi/p0/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
    print(f"\n== reproduction A s0: max |per-epoch loss diff| {np.abs(x - y).max():.2e} over {len(x)} epochs")

    print("\n== contrasts ==")
    res = {}
    for lo, hi in (("A", "L"), ("A", "L4"), ("L", "L4")):
        pf = fisher_solved(solved[lo], 8, solved[hi], 8); pp = perm2_p(acc[lo], acc[hi])["p"]
        ci = perm2_ci(acc[lo], acc[hi]); res[(lo, hi)] = (pf, pp)
        print(f"  {hi} - {lo}: SOLVED {solved[hi]}/8 vs {solved[lo]}/8 Fisher p {pf:.4f} | "
              f"acc {np.mean(acc[hi]) - np.mean(acc[lo]):+.3f} perm p {pp:.4f} CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
    ps = [min(res[k]) for k in ((("A", "L")), ("A", "L4"), ("L", "L4"))]
    order = np.argsort(ps); holm = np.empty(3); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (3 - rank) * ps[i])); holm[i] = run
    print("  Holm-adjusted:", " ".join(f"{n}={h:.4f}" for n, h in zip(("L-A", "L4-A", "L4-L"), holm)))

    fires = lambda lo, hi: min(res[(lo, hi)]) < 0.05
    if fires("A", "L") and solved["L"] >= 4:
        v = "H1 CONFIRMED -- the loop recovers rank 2 at identical parameters: the deficit is search"
    elif not fires("A", "L") and solved["L"] <= 2:
        v = "H1 REFUTED -- a search aid that works elsewhere does not recover rank 2 here"
    else:
        v = "UNMEASURED"
    print(f"\n== BRANCH: {v}")
    if fires("A", "L"):
        print("   compute reading: " + ("4x compute is sufficient; the loop is not special"
                                        if fires("A", "L4") else
                                        "weight sharing does something 4 real layers do not"))
    for T in (512, 2048):
        print(f"  T={T}: " + "  ".join(
            f"{k} {np.mean([dict((x[0], x) for x in json.load(open(f'{REPO}/{j}.json'))[f'0.0|{v}|{T}'])[s][1] for s in S]):.3f}"
            for k, (v, d, j) in ARMS.items()))


if __name__ == "__main__":
    main()
