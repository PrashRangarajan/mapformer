"""H1 part 1 (LOOP_RANK_E1800_PREREG.md, Amendment 1): is rank 2's failure a budget effect?"""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; S = list(range(8))
D = f"{REPO}/runs/loop_rank_e1800/p0"; OLD = f"{REPO}/runs/rank_mi/p0"


def main():
    J = json.load(open(f"{REPO}/LOOP_RANK_E1800_P1.json")); Jo = json.load(open(f"{REPO}/RANK_MI.json"))
    acc = lambda J, v, T=1024: [dict((x[0], x[1]) for x in J[f"0.0|{v}|{T}"])[s] for s in S]
    res = {}
    for k, v in (("A r=2", "Vanilla"), ("C r=4", "Vanilla_r4mi")):
        new = [classify_run(torch.load(f"{D}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"]) for s in S]
        old = [classify_run(torch.load(f"{OLD}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"]) for s in S]
        res[k] = (sum(c["registered"] == "SOLVED" for c in new), acc(J, v))
        print(f"{k}: 1800 ep SOLVED {res[k][0]}/8 acc {np.mean(res[k][1]):.3f}  " +
              " ".join(f"{c['cls'][:4]}({c['tail']:.3f})" for c in new))
        print(f"      900 ep SOLVED {sum(c['registered'] == 'SOLVED' for c in old)}/8 acc {np.mean(acc(Jo, v)):.3f}")
    (a, aa), (c, ca) = res["A r=2"], res["C r=4"]
    print(f"\nC - A at 1800: SOLVED {c}/8 vs {a}/8 Fisher p {fisher_solved(a, 8, c, 8):.4f}; "
          f"acc {np.mean(ca) - np.mean(aa):+.3f} perm p {perm2_p(aa, ca)['p']:.4f}")
    print(f"\nREGISTERED (branch 1 only): {'RANK 2 WAS BUDGET' if a >= 6 else 'not budget at 1800 epochs (A < 6/8)'}")


if __name__ == "__main__":
    main()
