"""Readouts for RANK_SEP_PREREG.md: C_bd and D against the stored A, B, C of runs/rank_mi."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; S = list(range(8))
ARMS = {"A": ("Vanilla", "rank_mi", "RANK_MI"), "B": ("Vanilla_r2ph", "rank_mi", "RANK_MI"),
        "C": ("Vanilla_r4mi", "rank_mi", "RANK_MI"), "C_bd": ("Vanilla_r4mibd", "rank_sep", "RANK_SEP"),
        "D": ("Vanilla_r4ph", "rank_sep", "RANK_SEP")}


def main():
    solved, acc = {}, {}
    print("== run classes (SOLVED = final-5% loss < 0.05) ==")
    for k, (v, d, j) in ARMS.items():
        cl = [classify_run(torch.load(f"{REPO}/runs/{d}/p0/{v}_s{s}/{v}.pt", map_location="cpu",
                                      weights_only=False)["losses"]) for s in S]
        solved[k] = sum(c["registered"] == "SOLVED" for c in cl)
        J = json.load(open(f"{REPO}/{j}.json")); dd = {x[0]: x for x in J[f"0.0|{v}|1024"]}
        acc[k] = [dd[s][1] for s in S]
        print(f"  {k:5s} {v:15s} SOLVED {solved[k]}/8  acc {np.mean(acc[k]):.3f}  " +
              " ".join(f"s{s}:{c['registered'][:4]}({c['tail']:.3f})" for s, c in zip(S, cl)))
    x = np.array(torch.load(f"{REPO}/runs/rank_sep_repro/p0/Vanilla_r4mi_s0/Vanilla_r4mi.pt", map_location="cpu", weights_only=False)["losses"])
    y = np.array(torch.load(f"{REPO}/runs/rank_mi/p0/Vanilla_r4mi_s0/Vanilla_r4mi.pt", map_location="cpu", weights_only=False)["losses"])
    print(f"\n== reproduction C s0: max |per-epoch loss diff| {np.abs(x - y).max():.2e} over {len(x)} epochs")
    tests = [("cross-head reading", "C_bd", "C"), ("W_out scale", "B", "C_bd"),
             ("sharing", "C", "D"), ("per-head rank", "C_bd", "D")]
    rows, ps = [], []
    for name, lo, hi in tests:
        pf = fisher_solved(solved[lo], 8, solved[hi], 8); pp = perm2_p(acc[lo], acc[hi])["p"]
        rows.append((name, lo, hi, pf, pp)); ps.append(min(pf, pp))
    order = np.argsort(ps); holm = np.empty(4)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (4 - rank) * ps[i])); holm[i] = run
    print("\n== contrasts (hi - lo) ==")
    fired = {}
    for (name, lo, hi, pf, pp), h in zip(rows, holm):
        fired[name] = pf < 0.05 or pp < 0.05
        print(f"  {name:18s} {hi} - {lo}: SOLVED {solved[hi]}/8 vs {solved[lo]}/8 Fisher p {pf:.4f} | "
              f"acc {np.mean(acc[hi]) - np.mean(acc[lo]):+.3f} perm p {pp:.4f} | Holm {h:.4f} | "
              f"{'FIRES' if fired[name] else 'UNMEASURED'}")
    if not fired["cross-head reading"] and fired["W_out scale"]:
        v = "r=4's advantage is an optimiser-scale effect, not rank"
    elif fired["per-head rank"] and not fired["sharing"]:
        v = "the rank of each head's content-to-angle map"
    elif fired["cross-head reading"]:
        v = "heads reading each other's latent matters"
    else:
        v = "no registered pattern -- reported as it falls, no mechanism sentence"
    print(f"\n== READING: {v}")


if __name__ == "__main__":
    main()
