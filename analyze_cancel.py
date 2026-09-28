"""Readouts for CANCEL_PREREG.md (H3)."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p

R = "/home/prashr/mapformer/runs/cancel/p0"; S = range(8); PS = ("0.5", "0.75", "0.9", "1.0")
ARMS = [("Vanilla", 1), ("RoPE", 1), ("RoPE", 2), ("RoPE", 3)]
FLOOR = {"0.5": 0.554, "0.75": 0.544, "0.9": 0.520, "1.0": 0.498}


def main():
    acc, ood, cls, tails = {}, {}, {}, {}
    for p in PS:
        for v, L in ARMS:
            k = (v, L, p); acc[k], ood[k], cls[k] = [], [], []
            for s in S:
                d = f"{R}/{v}_L{L}_p{p}_s{s}"; e = json.load(open(f"{d}/eval.json"))["eval"]
                acc[k].append(e["128"]["acc"]); ood[k].append(e["512"]["acc"])
                c = classify_run(torch.load(f"{d}/{v}.pt", map_location="cpu", weights_only=False)["losses"])
                cls[k].append(c["registered"]); tails.setdefault("x", []).append(c["tail"]); tails.setdefault("y", []).append(acc[k][-1])
    print("== T=128 accuracy (mean +/- sd; classes S/T/D) and T=512 (extrapolation, no verdict) ==")
    print("| p_plus | floor | " + " | ".join(f"{'path' if v == 'Vanilla' else 'index'} {L}L" for v, L in ARMS) + " |")
    for p in PS:
        cells = []
        for v, L in ARMS:
            a = acc[(v, L, p)]; c = cls[(v, L, p)]
            cells.append(f"{np.mean(a):.3f}+/-{np.std(a, ddof=1):.3f} ({c.count('SOLVED')}/{c.count('STALLED')}/{c.count('DESCENDING')}) [{np.mean(ood[(v, L, p)]):.3f}]")
        print(f"| {p} | {FLOOR[p]:.3f} | " + " | ".join(cells) + " |")
    print(f"\n  r(final loss, acc@128) over 128 runs: {np.corrcoef(tails['x'], tails['y'])[0, 1]:+.3f}")
    a1 = {p: acc[("RoPE", 1, p)] for p in PS}
    print("\n== primary: index 1-layer accuracy a1(p), pairwise exact permutation p ==")
    P = {}
    for i, p in enumerate(PS):
        for q in PS[i + 1:]:
            P[(p, q)] = perm2_p(a1[p], a1[q])["p"]
            print(f"  a1({q}) - a1({p}) = {np.mean(a1[q]) - np.mean(a1[p]):+.3f}  perm p {P[(p, q)]:.4f}")
    diff = lambda p, q: P[(p, q)] < 0.05 if (p, q) in P else P[(q, p)] < 0.05
    m = {p: np.mean(a1[p]) for p in PS}
    if (all(diff(x, "0.5") and diff(x, "1.0") for x in ("0.75", "0.9"))
            and m["0.5"] < m["0.75"] < m["0.9"] < m["1.0"]):
        v = "CONTINUUM"
    elif (all((not diff(x, "0.5")) or m[x] < m["0.5"] for x in ("0.75", "0.9"))
          and all(diff("1.0", x) for x in ("0.5", "0.75", "0.9"))):
        v = "DICHOTOMY"
    else:
        v = "no registered branch -- reported as it falls"
    print(f"\n== REGISTERED VERDICT: {v}")
    print("\n== secondary: gap G = path1 - index L, and exchange rate k(p) ==")
    for p in PS:
        pm = np.mean(acc[("Vanilla", 1, p)])
        gs = [pm - np.mean(acc[("RoPE", L, p)]) for L in (1, 2, 3)]
        k = next((L for L, g in zip((1, 2, 3), gs) if g <= 0.01), ">3")
        print(f"  p={p}: path1 {pm:.3f} (floor {FLOOR[p]:.3f})  G1 {gs[0]:+.3f} G2 {gs[1]:+.3f} G3 {gs[2]:+.3f}  k={k}"
              f"  {'VOID: path below floor' if pm < FLOOR[p] else ''}")


if __name__ == "__main__":
    main()
