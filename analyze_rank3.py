"""Readouts for RANK3_PREREG.md: E (per-head r=3) against the stored B (per-head r=2, runs/rank_mi)
and D (per-head r=4, runs/rank_sep)."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; S = list(range(8))
ARMS = {"B": ("Vanilla_r2ph", "rank_mi", "RANK_MI"), "E": ("Vanilla_r3ph", "rank3", "RANK3"),
        "D": ("Vanilla_r4ph", "rank_sep", "RANK_SEP")}


def ld(p):
    return torch.load(p, map_location="cpu", weights_only=False)["losses"]


def main():
    solved, acc = {}, {}
    print("== run classes (SOLVED = final-5% loss < 0.05) ==")
    for k, (v, d, j) in ARMS.items():
        cl = [classify_run(ld(f"{REPO}/runs/{d}/p0/{v}_s{s}/{v}.pt")) for s in S]
        solved[k] = sum(c["registered"] == "SOLVED" for c in cl)
        J = json.load(open(f"{REPO}/{j}.json"))
        for L in (512, 1024, 2048):
            key = f"0.0|{v}|{L}"
            if key in J:
                print(f"  {k} T={L}: acc {np.mean([x[1] for x in J[key]]):.3f}")
        dd = {x[0]: x for x in J[f"0.0|{v}|1024"]}
        acc[k] = [dd[s][1] for s in S]
        print(f"  {k} {v:13s} SOLVED {solved[k]}/8  acc {np.mean(acc[k]):.3f}  " +
              " ".join(f"s{s}:{c['registered'][:4]}({c['tail']:.3f})" for s, c in zip(S, cl)))
    x = np.array(ld(f"{REPO}/runs/rank3_repro/p0/Vanilla_r4ph_s0/Vanilla_r4ph.pt"))
    y = np.array(ld(f"{REPO}/runs/rank_sep/p0/Vanilla_r4ph_s0/Vanilla_r4ph.pt"))
    rep = np.abs(x - y).max()
    print(f"\n== reproduction D s0: max |per-epoch loss diff| {rep:.2e} over {len(x)} epochs"
          f" -> {'OK' if rep == 0 else 'DIFFERS: comparison with stored arms VOID'}")
    rows, ps = [], []
    for name, lo, hi in [("rank 3 vs 2", "B", "E"), ("rank 4 vs 3", "E", "D")]:
        pf = fisher_solved(solved[lo], 8, solved[hi], 8); pp = perm2_p(acc[lo], acc[hi])["p"]
        rows.append((name, lo, hi, pf, pp)); ps.append(min(pf, pp))
    holm = [min(1.0, 2 * min(ps)), min(1.0, max(2 * min(ps), max(ps)))]
    holm = [holm[0] if p == min(ps) else holm[1] for p in ps]
    print("\n== contrasts (hi - lo) ==")
    fired = {}
    for (name, lo, hi, pf, pp), h in zip(rows, holm):
        fired[name] = pf < 0.05 or pp < 0.05
        print(f"  {name:12s} {hi} - {lo}: SOLVED {solved[hi]}/8 vs {solved[lo]}/8 Fisher p {pf:.4f} | "
              f"acc {np.mean(acc[hi]) - np.mean(acc[lo]):+.3f} perm p {pp:.4f} | Holm {h:.4f} | "
              f"{'FIRES' if fired[name] else 'UNMEASURED'}")
    a, b = fired["rank 3 vs 2"], fired["rank 4 vs 3"]
    if a and not b and solved["E"] >= 6:
        v = "RANK 3 SUFFICES"
    elif b and not a and solved["E"] <= 2:
        v = "RANK 3 IS NOT ENOUGH"
    elif a and b:
        v = "GRADED"
    else:
        v = "no registered branch -- reported as it falls, no mechanism sentence"
    print(f"\n== REGISTERED VERDICT: {v}")


if __name__ == "__main__":
    main()
