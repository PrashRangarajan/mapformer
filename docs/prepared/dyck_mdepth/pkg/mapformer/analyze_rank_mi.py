"""Readouts for RANK_MI_PREREG.md: A our shared r=2, B the paper's per-head r=2, C shared r=4 at matched init."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, perm2_ci, fisher_solved

REPO = "/home/prashr/mapformer"; S = list(range(8))
ARMS = {"A": "Vanilla", "B": "Vanilla_r2ph", "C": "Vanilla_r4mi"}


def main():
    M = json.load(open(f"{REPO}/RANK_MI.json"))
    acc = lambda v, T: [dict((x[0], x) for x in M[f"0.0|{v}|{T}"])[s][1] for s in S]
    cls, solved = {}, {}
    print("== run classes (registered labels; SOLVED = final-5% loss < 0.05) ==")
    for k, v in ARMS.items():
        cls[k] = []
        for s in S:
            b = torch.load(f"{REPO}/runs/rank_mi/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
            r = classify_run(b["losses"]); cls[k].append(r)
        solved[k] = sum(r["registered"] == "SOLVED" for r in cls[k])
        print(f"  {k} {v:13s} SOLVED {solved[k]}/8  " +
              " ".join(f"s{s}:{r['registered'][:4]}({r['tail']:.3f})" for s, r in zip(S, cls[k])))
    old = [classify_run(torch.load(f"{REPO}/runs/rank_matched_e900/p0/Vanilla_r4_s{s}/Vanilla_r4.pt",
                                   map_location="cpu", weights_only=False)["losses"])["registered"] == "SOLVED" for s in S]
    print(f"  reference: stored r=4 (unmatched init) SOLVED {sum(old)}/8")

    print("\n== reproduction: A seed 3 retrained vs stored ==")
    x = np.array(torch.load(f"{REPO}/runs/rank_mi_repro/p0/Vanilla_s3/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
    y = np.array(torch.load(f"{REPO}/runs/rank_matched_e900/p0/Vanilla_s3/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
    print(f"  max |per-epoch loss diff| over {len(x)} epochs: {np.abs(x - y).max():.2e}")

    print("\n== co-primaries at T=1024 ==")
    a = {k: acc(v, 1024) for k, v in ARMS.items()}
    for k, v in ARMS.items():
        print(f"  {k} {v:13s} acc {np.mean(a[k]):.3f} +/- {np.std(a[k], ddof=1):.3f}  per seed " + " ".join(f"{z:.3f}" for z in a[k]))
    res = {}
    for x_, y_ in (("A", "C"), ("A", "B"), ("B", "C")):
        pf = fisher_solved(solved[x_], 8, solved[y_], 8)
        pp = perm2_p(a[x_], a[y_])["p"]; ci = perm2_ci(a[x_], a[y_])
        d = np.mean(a[y_]) - np.mean(a[x_]); res[(x_, y_)] = (pf, pp, d, solved[y_] - solved[x_])
        print(f"  {y_} - {x_}: SOLVED {solved[y_]}/8 vs {solved[x_]}/8 Fisher p {pf:.4f} | acc {d:+.3f} "
              f"perm p {pp:.4f} CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")

    def sig(x_, y_):   # y significantly better than x on either co-primary
        pf, pp, d, ds = res[(x_, y_)]
        return (pf < 0.05 and ds > 0) or (pp < 0.05 and d > 0)

    def nsig(x_, y_):
        pf, pp, _, _ = res[(x_, y_)]; return pf >= 0.05 and pp >= 0.05
    print("\n== branches ==")
    if sig("A", "C"):
        q1 = "C > A: the r=4 advantage SURVIVES matched initialisation"
    elif nsig("A", "C") and solved["C"] <= 4:
        q1 = "stored r=4's 8/8 depended on its initial draws -- rank recommendation WITHDRAWN"
    else:
        q1 = "UNMEASURED"
    if sig("B", "C"):
        q2 = "C > B: the paper's per-head r=2 has the search problem too; 'use r=4' applies to MapFormer's design"
    elif sig("A", "B") and nsig("B", "C"):
        q2 = "B > A, B ~ C: latent dimension count (4 vs 2) matters, not sharing; the paper's per-head design is fine"
    else:
        q2 = "UNMEASURED"
    print(f"  Q1 (init confound): {q1}\n  Q2 (paper's design): {q2}")

    print("\n== secondary: other lengths ==")
    for T in (512, 2048):
        print(f"  T={T}: " + "  ".join(f"{k} {np.mean(acc(v, T)):.3f}" for k, v in ARMS.items()))
    st = json.load(open(f"{REPO}/RANK_MI_STRATA.json"))
    print("  T=1024 strata: " + "  ".join(
        f"{q}: " + " ".join(f"{k} {np.mean([st[f'{v}|{s}|1024'][q]['acc'] for s in S]):.3f}" for k, v in ARMS.items())
        for q in ("plain_lag<128", "plain_lag>=128", "wrap")))


if __name__ == "__main__":
    main()
