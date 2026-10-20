"""Tables for RECURRENCE.md from out/*.json (pass_strata, phase_code). Also checks the k=4 (L) and final
(L4) 'all' accuracies against the stored registered evaluations (LOOP_RANK.json / RANK_MI.json)."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT, REPO  # noqa

sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import perm2_p  # noqa: E402

S = range(8)
STR = ("all", "short", "long", "wrap")
ps = {a: json.load(open(f"{OUT}/pass_strata_{a}.json")) for a in ("A", "L", "L4", "C")}
pc = {a: json.load(open(f"{OUT}/phase_code_{a}.json")) for a in ("A", "L", "L4", "C")}
stored = {"A": ("RANK_MI", "Vanilla"), "C": ("RANK_MI", "Vanilla_r4mi"),
          "L": ("LOOP_RANK", "Looped"), "L4": ("LOOP_RANK", "Vanilla_L4")}
fin = {"A": 0, "C": 0, "L": 3, "L4": 3}   # index of the trained readout pass

print("== check against stored registered evaluation (T=1024, all revisits)")
for a, (j, v) in stored.items():
    st = {x[0]: x[1] for x in json.load(open(f"{REPO}/{j}.json"))[f"0.0|{v}|1024"]}
    d = max(abs(ps[a][str(s)]["all"]["acc"][fin[a]] - st[s]) for s in S)
    print(f"  {a:3s} max |ours - stored| = {d:.2e}")

print("\n== model accuracy (trained readout) vs best one-pass phase-only reader (obs top-1), mean over 8 seeds")
print("| arm | stratum | model acc | reader obs top-1 | reader cell top-1 | model - reader |")
print("|---|---|---|---|---|---|")
gap = {}
for a in ("A", "L", "L4", "C"):
    for k in STR:
        ma = np.array([ps[a][str(s)][k]["acc"][fin[a]] for s in S])
        ro = np.array([pc[a][str(s)]["fitted"][k]["obs_top1"] for s in S])
        rc = np.array([pc[a][str(s)]["fitted"][k]["cell_top1"] for s in S])
        gap[(a, k)] = ma - ro
        print(f"| {a} | {k} | {ma.mean():.3f} | {ro.mean():.3f} | {rc.mean():.3f} | {np.mean(ma - ro):+.3f} |")

print("\n== code quality contrasts (fitted reader cell top-1; permutation p, 8 vs 8)")
for k in ("all", "long", "wrap"):
    v = {a: [pc[a][str(s)]["fitted"][k]["cell_top1"] for s in S] for a in pc}
    for lo, hi in (("A", "L"), ("A", "L4"), ("L", "C"), ("L4", "C")):
        p = perm2_p(v[lo], v[hi])["p"]
        print(f"  {k:5s} {hi} - {lo}: {np.mean(v[hi]) - np.mean(v[lo]):+.3f}  p {p:.4f}")
print("\n== compensation (model - reader, obs top-1) contrasts")
for k in ("long", "wrap"):
    for lo, hi in (("A", "L"), ("A", "L4")):
        p = perm2_p(list(gap[(lo, k)]), list(gap[(hi, k)]))["p"]
        print(f"  {k:5s} {hi} - {lo}: {np.mean(gap[(hi, k)]) - np.mean(gap[(lo, k)]):+.3f}  p {p:.4f}")

print("\n== obs-token step / action step (the leak)")
for a in pc:
    v = [pc[a][str(s)]["leak_obs_over_action"] for s in S]
    print(f"  {a:3s} " + " ".join(f"{x:.3f}" for x in v) + f"  mean {np.mean(v):.3f}")

print("\n== per pass (L: n_loops = j, passes 5-8 beyond training; L4: logit lens after layer j), mean over seeds")
for a in ("A", "L", "L4", "C"):
    P = len(ps[a]["0"]["all"]["acc"])
    print(f"-- {a}")
    for k in STR:
        acc = np.array([ps[a][str(s)][k]["acc"] for s in S])
        print(f"  acc   {k:5s} " + " ".join(f"{x:.3f}" for x in acc.mean(0)))
    for k in STR:
        for nm in ("obs_mass", "act_mass", "argmax_same_cell"):
            x = np.array([ps[a][str(s)][k][nm] for s in S])          # (8, P, H)
            print(f"  {nm:16s} {k:5s} " + " | ".join(
                "h0 %.2f h1 %.2f" % (x[:, p, 0].mean(), x[:, p, 1].mean()) for p in range(P)))

print("\n== L per seed: best pass count, and accuracy at 1 / 2 / 4 / 8 passes")
for s in S:
    acc = ps["L"][str(s)]["all"]["acc"]
    print(f"  s{s}: best k={int(np.argmax(acc)) + 1}  " + " ".join(f"k{k}={acc[k-1]:.3f}" for k in (1, 2, 3, 4, 5, 8)))

print("\n== soft one-pass position-only reader (phase_reader_soft.py) vs model, mean over 8 seeds")
print("| arm | stratum | model acc | soft reader acc | model - reader | perm p (gap vs A's gap) |")
print("|---|---|---|---|---|---|")
sr = {a: json.load(open(f"{OUT}/phase_reader_soft_{a}.json")) for a in ("A", "L", "L4", "C")}
sgap = {}
for a in ("A", "L", "L4", "C"):
    for k in STR:
        ma = np.array([ps[a][str(s)][k]["acc"][fin[a]] for s in S])
        ra = np.array([sr[a][str(s)][k] for s in S])
        sgap[(a, k)] = ma - ra
        p = perm2_p(list(sgap[("A", k)]), list(sgap[(a, k)]))["p"] if a != "A" else float("nan")
        print(f"| {a} | {k} | {ma.mean():.3f} | {ra.mean():.3f} | {np.mean(ma - ra):+.3f} | {p:.4f} |")
print("\n== soft-reader code quality contrasts (perm p)")
for k in ("all", "long", "wrap"):
    v = {a: [sr[a][str(s)][k] for s in S] for a in sr}
    for lo, hi in (("A", "L"), ("A", "L4"), ("L", "C"), ("L4", "C")):
        print(f"  {k:5s} {hi} - {lo}: {np.mean(v[hi]) - np.mean(v[lo]):+.3f}  p {perm2_p(v[lo], v[hi])['p']:.4f}")
