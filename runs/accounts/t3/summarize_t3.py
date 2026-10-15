"""ACCOUNTS T3 summary (exploratory toy, observation-retrieval target): per start, MAP SOLVED (same-cell retrieval acc >= 0.99),
observation-retrieval accuracy, and the share of channels with opposite-sign move angles at the end."""
import json, glob, collections
import numpy as np
g = collections.defaultdict(list)
for f in sorted(glob.glob("/home/prashr/mapformer/runs/accounts/t3/*_obs_s*.json")):
    r = json.load(open(f)); a = r["args"]; A = np.array(r["angles"])
    sel = np.abs(np.angle(np.exp(1j * A[0]))) > 0.05
    opp = float((np.sign(A[0]) != np.sign(A[1]))[sel].mean()) if sel.any() else float("nan")
    g[f"{a['mode']} {a['init']}"].append((a["seed"], r["final_acc_cell"], r["final_acc"], opp))
for k, v in sorted(g.items()):
    print(f"{k:14s} n={len(v)} MAP SOLVED {sum(x[1] >= 0.99 for x in v)}/{len(v)}; acc_cell median {np.median([x[1] for x in v]):.3f}; "
          f"acc_obs median {np.median([x[2] for x in v]):.3f}; opposite-sign channels {np.nanmean([x[3] for x in v]):.2f}")
