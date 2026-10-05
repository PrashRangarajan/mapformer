"""Compare registered vs re-scored (attention x 1/(1-p)) accuracies for the torus batches and recompute the registered
accuracy contrasts (exact permutation, stats_core.perm2_p). Inputs: runs_rescore/<B>_{none,auto}.json from
rescore_torus.sh; the scale-none run must reproduce the registered JSON exactly (checked)."""
import json
import numpy as np
from mapformer.stats_core import perm2_p

R = "/home/prashr/mapformer"; O = f"{R}/runs_rescore"
REG = {"PAPER2X2": "_PAPER2X2_RAW.json", "RANK_MI": "RANK_MI.json", "RANK_SEP": "RANK_SEP.json", "RANK3": "RANK3.json",
       "LOOP_RANK": "LOOP_RANK.json", "LOOP_RANK_E1800_P1": "LOOP_RANK_E1800_P1.json", "SIGN_MATCHED": "SIGN_MATCHED.json"}
LEN = {"PAPER2X2": 128}
acc = {}
for b, f in REG.items():
    reg = json.load(open(f"{R}/{f}")); none = json.load(open(f"{O}/{b}_none.json")); auto = json.load(open(f"{O}/{b}_auto.json"))
    L = LEN.get(b, 1024)
    for k in auto:
        assert k.endswith(f"|{L}")
        r = [x[1] for x in reg[k]]; n = [x[1] for x in none[k]]; a = [x[1] for x in auto[k]]
        assert max(abs(x - y) for x, y in zip(r, n)) < 1e-12, ("scale-none does not reproduce", b, k)
        acc[(b, k.split("|")[1])] = (np.array(r), np.array(a))
print("scale-none reproduces every registered JSON entry exactly (all batches)\n")
print(f"{'batch':20s} {'arm':16s} {'registered':>10s} {'rescored':>9s} {'max per-seed gain':>18s}")
for (b, v), (r, a) in acc.items():
    print(f"{b:20s} {v:16s} {r.mean():10.4f} {a.mean():9.4f} {np.max(a - r):+18.4f}")


def con(label, x, y):
    (rx, ax), (ry, ay) = acc[x], acc[y]
    dr, pr = ry.mean() - rx.mean(), perm2_p(rx, ry)["p"]; da, pa = ay.mean() - ax.mean(), perm2_p(ax, ay)["p"]
    flip = (pr < 0.05) != (pa < 0.05) or np.sign(dr) != np.sign(da)
    print(f"  {label:44s} registered {dr:+.4f} (p {pr:.4f})   rescored {da:+.4f} (p {pa:.4f})  {'<-- CHANGES' if flip else ''}")


print("\n== registered accuracy contrasts, registered vs re-scored ==")
con("paper2x2: MapWM r2 - RoPE (T=128)", ("PAPER2X2", "RoPE"), ("PAPER2X2", "Vanilla"))
con("paper2x2: MapWM r4 - RoPE (T=128)", ("PAPER2X2", "RoPE"), ("PAPER2X2", "Vanilla_r4"))
con("paper2x2: MapPoPE r2 - PoPE (T=128)", ("PAPER2X2", "PoPE-Flat"), ("PAPER2X2", "MapPoPE-Flat"))
con("rank_mi: r4 shared - r2 shared", ("RANK_MI", "Vanilla"), ("RANK_MI", "Vanilla_r4mi"))
con("rank_mi: r4 shared - r2 per head", ("RANK_MI", "Vanilla_r2ph"), ("RANK_MI", "Vanilla_r4mi"))
con("rank_sep: D (r4/head) - C_bd (block-diag)", ("RANK_SEP", "Vanilla_r4mibd"), ("RANK_SEP", "Vanilla_r4ph"))
con("rank3: r3/head - r2/head", ("RANK_MI", "Vanilla_r2ph"), ("RANK3", "Vanilla_r3ph"))
con("rank3: r4/head - r3/head", ("RANK3", "Vanilla_r3ph"), ("RANK_SEP", "Vanilla_r4ph"))
con("loop_rank: loop x4 - plain r2", ("RANK_MI", "Vanilla"), ("LOOP_RANK", "Looped"))
con("loop_rank: 4 layers - plain r2", ("RANK_MI", "Vanilla"), ("LOOP_RANK", "Vanilla_L4"))
con("loop_rank: r4 - loop x4", ("LOOP_RANK", "Looped"), ("RANK_MI", "Vanilla_r4mi"))
con("e1800: r4 - r2 (1800 ep)", ("LOOP_RANK_E1800_P1", "Vanilla"), ("LOOP_RANK_E1800_P1", "Vanilla_r4mi"))
con("sign: Abs - Signed", ("SIGN_MATCHED", "Signed_r4"), ("SIGN_MATCHED", "Abs_r4"))
con("sign: Pos - Signed", ("SIGN_MATCHED", "Signed_r4"), ("SIGN_MATCHED", "Pos_r4"))
con("sign: RoPE - Signed", ("SIGN_MATCHED", "Signed_r4"), ("SIGN_MATCHED", "RoPE"))
