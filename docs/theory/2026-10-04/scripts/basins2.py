import sys; sys.argv=['x']
exec(open('/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/basins.py').read().split("tab = {}")[0])
import numpy as np
rows=[]
for d, arm, lab in sets:
    for s in range(8):
        o, fl = heads(R / d / f"{arm}_s{s}" / f"{arm}.pt", True)
        kmin = min(k for k,_ in o); ind = max([i for k,i in o if k<=0.01], default=float('nan'))
        rows.append((lab,s,fl<0.05,fl,kmin,ind,basin(o)))
for r in rows:
    if (r[6]=="CLEAN") != r[2]: print("mismatch:", r)
S=[r for r in rows if r[2]]; U=[r for r in rows if not r[2]]
print("solved: max kmin %.4f ; min indep of clean head %.3f"%(max(r[4] for r in S), min(r[5] for r in S)))
print("unsolved kmin sorted:", np.round(sorted(r[4] for r in U),3))
print("unsolved with drift-free head, indep:", [(r[0],r[1],round(r[5],3)) for r in U if r[4]<=0.01])
