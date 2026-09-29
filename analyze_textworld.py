"""Readouts for TEXTWORLD_PREREG.md."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/textworld/p0"; S = range(8)
ARMS = [("Vanilla_r4", 1), ("RoPE", 1), ("RoPE", 2)]
FLOOR = 0.512


def main():
    acc, ood, cls, tail = {}, {}, {}, {}
    for v, L in ARMS:
        k = f"{v}_L{L}"; acc[k], ood[k], cls[k], tail[k] = [], [], [], []
        for s in S:
            d = f"{R}/{k}_s{s}"; e = json.load(open(f"{d}/eval.json"))["eval"]
            acc[k].append(e["1024"]["acc"]); ood[k].append(e["2048"]["acc"])
            c = classify_run(torch.load(f"{d}/{v}.pt", map_location="cpu", weights_only=False)["losses"])
            cls[k].append(c["registered"]); tail[k].append(c["tail"])
    print(f"== T=1024 held-out-map accuracy (floor {FLOOR}); classes SOLVED/STALLED/DESCENDING; [T=2048, no verdict] ==")
    for k in acc:
        c = cls[k]
        print(f"  {k:14s} {np.mean(acc[k]):.3f} +/- {np.std(acc[k], ddof=1):.3f}  "
              f"({c.count('SOLVED')}/{c.count('STALLED')}/{c.count('DESCENDING')})  [{np.mean(ood[k]):.3f}]  "
              + " ".join(f"{a:.3f}" for a in acc[k]))
    x = sum(tail.values(), []); y = sum(acc.values(), [])
    print(f"  r(final loss, acc) over 24 runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    P, Q = "Vanilla_r4_L1", "RoPE_L1"
    pp = perm2_p(acc[Q], acc[P])["p"]; sp, sq = cls[P].count("SOLVED"), cls[Q].count("SOLVED")
    pf = fisher_solved(sq, 8, sp, 8); d = np.mean(acc[P]) - np.mean(acc[Q])
    fires = pp < 0.05 or pf < 0.05
    print(f"\n== primary A: path 1L - RoPE 1L = {d:+.3f}, perm p {pp:.4f}; SOLVED {sp}/8 vs {sq}/8 Fisher p {pf:.4f}")
    print(f"   REGISTERED A: {'PATH WINS IN WORDS' if fires and d > 0 else 'UNMEASURED'}")
    for a, b in (("RoPE_L2", "RoPE_L1"), ("Vanilla_r4_L1", "RoPE_L2")):
        print(f"   secondary {a} - {b}: {np.mean(acc[a]) - np.mean(acc[b]):+.3f}, perm p {perm2_p(acc[b], acc[a])['p']:.4f}")
    pr = json.load(open(f"{REPO}/TEXTWORLD_PROBE.json"))
    rows = [pr[f"{P}_s{s}"] for s in S]
    full = [r["move_ratio"] < 0.2 and r["opposition"] < 0.3 and r["synonym_cos"] > 0.9 for r in rows]
    syn = [r["synonym_cos"] > 0.9 for r in rows]
    print("\n== primary B: the step table (path arm) ==")
    for s, r, f in zip(S, rows, full):
        print(f"  s{s}: move {r['move_ratio']:.3f}  opp {r['opposition']:.3f}  syn cos {r['synonym_cos']:.3f}  "
              f"syn norm {r['synonym_norm']:.3f}  |cos NE| {r['cos_NE']:.3f}  {'CRITERION' if f else '         '}  "
              f"{cls[P][s]}  top: {' '.join(r['top_words'][:8])}")
    if sum(full) >= 6:
        vb = "FINDS THE ACTION WORDS"
    elif sum(syn) >= 6 and sum(full) <= 2:
        vb = "FINDS SYNONYMS ONLY"
    else:
        vb = "no registered branch -- reported as it falls"
    print(f"   criterion met {sum(full)}/8, synonym cos > 0.9 on {sum(syn)}/8")
    print(f"   REGISTERED B: {vb}")
    solved = [c == "SOLVED" for c in cls[P]]
    a = sum(f and s for f, s in zip(full, solved)); b = sum(f and not s for f, s in zip(full, solved))
    c_ = sum((not f) and s for f, s in zip(full, solved)); d_ = sum((not f) and (not s) for f, s in zip(full, solved))
    from scipy.stats import fisher_exact
    print(f"   secondary: criterion x SOLVED table [[{a},{b}],[{c_},{d_}]], Fisher p {fisher_exact([[a, b], [c_, d_]])[1]:.4f}")


if __name__ == "__main__":
    main()
