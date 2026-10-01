"""Headline contrasts recomputed on FRESH seeds only (s2-s7): seeds 0 and 1 of the text-world and H3
batches are byte-identical to their pilots, which were read before registration (audit 2026-09-30 M2)."""
import json
import numpy as np
import torch
from mapformer.stats_core import perm2_p, fisher_solved, classify_run

R = "/home/prashr/mapformer/runs"; F = range(2, 8)
acc = lambda d, T: json.load(open(f"{d}/eval.json"))["eval"][T]["acc"]
print("== text world, T=1024, seeds 2-7 ==")
tw = {k: [acc(f"{R}/textworld/p0/{k}_s{s}", "1024") for s in F] for k in ("Vanilla_r4_L1", "RoPE_L1", "RoPE_L2")}
sol = {k: sum(classify_run(torch.load(f"{R}/textworld/p0/{k}_s{s}/{k.split('_L')[0]}.pt", map_location="cpu",
                                      weights_only=False)["losses"])["registered"] == "SOLVED" for s in F) for k in tw}
for k, v in tw.items():
    print(f"  {k:14s} {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}  SOLVED {sol[k]}/6")
P, Q = "Vanilla_r4_L1", "RoPE_L1"
print(f"  path - RoPE 1L {np.mean(tw[P]) - np.mean(tw[Q]):+.3f}, perm p {perm2_p(tw[Q], tw[P])['p']:.4f}, "
      f"SOLVED {sol[P]}/6 vs {sol[Q]}/6 Fisher p {fisher_solved(sol[Q], 6, sol[P], 6):.4f}")
print("== H3 index 1-layer accuracy a1(p), T=128, seeds 2-7 ==")
a1 = {p: [acc(f"{R}/cancel/p0/RoPE_L1_p{p}_s{s}", "128") for s in F] for p in ("0.5", "0.75", "0.9", "1.0")}
for p, v in a1.items():
    print(f"  p_plus {p}: {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}")
ps = list(a1)
for i, p in enumerate(ps):
    for q in ps[i + 1:]:
        print(f"  a1({q}) - a1({p}) = {np.mean(a1[q]) - np.mean(a1[p]):+.3f}, perm p {perm2_p(a1[p], a1[q])['p']:.4f}")
for p in ps:
    path = np.mean([acc(f"{R}/cancel/p0/Vanilla_L1_p{p}_s{s}", "128") for s in F])
    g = [path - np.mean([acc(f"{R}/cancel/p0/RoPE_L{L}_p{p}_s{s}", "128") for s in F]) for L in (1, 2, 3)]
    print(f"  p_plus {p}: path1 {path:.3f}  G1 {g[0]:+.4f} G2 {g[1]:+.4f} G3 {g[2]:+.4f}")
