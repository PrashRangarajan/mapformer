"""M8 (post hoc): at which values does sigma = (alpha + beta)/2 lock (|sigma - x| < 0.003)? Theory F4 allows only
x = 1 (period 2) and 1.5 (period 3) for a count that a periodic table can read at every D = 0 return."""
import json, numpy as np
D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
m1 = json.load(open(D + "m1_out.json"))
m4 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m4_out.json"))}
vals = []
for r in m1:
    for hk, inc in r["inc"].items():
        vals.append((inc["sigma"], inc["tau"], r["tabs"][hk]["p2_share"], r["arm"], r["seed"], hk, r["acc"],
                     m4[(r["batch"], r["arm"], r["seed"])]["init"][hk]["sigma"]))
s = np.array([v[0] for v in vals]); s0 = np.array([v[7] for v in vals])
print(f"heads x query types: {len(s)}")
for x in (0.5, 0.75, 1.0, 1.25, 1.5):
    sel = [v for v in vals if abs(v[0] - x) < 0.003]
    print(f"sigma = {x}: final {len(sel):3d} (tau median {np.median([v[1] for v in sel]) if sel else float('nan'):.4f}, "
          f"period-2 share median {np.median([v[2] for v in sel]) if sel else float('nan'):.2f}) | at init {int((np.abs(s0 - x) < 0.003).sum())}")
other = [v for v in vals if min(abs(v[0] - x) for x in (0.5, 0.75, 1.0, 1.25, 1.5)) >= 0.003]
print(f"not locked at any of these: {len(other)}")
# a null for 'locking': fraction of a uniform spread that would fall within +/-0.003 of one of 5 points over [0.3, 1.6]
print(f"chance rate for a value spread uniformly over [0.3, 1.6]: {5 * 0.006 / 1.3:.3f} per head")
