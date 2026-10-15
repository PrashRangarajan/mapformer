"""M7 (post hoc): kappa at init (m4, rebuilt) vs at the end (m1) for every period-P head, by final class (m2).
Question: do OFF heads end near kappa = 0 because they started there, or because training collapsed the move gates?"""
import json, numpy as np
D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
m1 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m1_out.json"))}
m4 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m4_out.json"))}
m2 = json.load(open(D + "m2_out.json"))
res = {}
for o in m2:
    key = (o["batch"], o["arm"], o["seed"])
    for h, (cl, P, k) in enumerate(o["heads"]):
        hk = f"h{h}a{o['a']}"
        i0 = m4[key]["init"][hk]; f = m1[key]["inc"][hk]
        res.setdefault(cl, []).append((i0["kappa"], f["kappa"], i0["g_plus"], i0["g_minus"], f["g_plus"], f["g_minus"]))
print("class | n | |kappa| init median | |kappa| final median | init move gates median (larger, smaller) | final move gates median")
for cl in ("LATTICE", "ALIASED", "OFF", "noTable", "nonP"):
    a = np.array(res.get(cl, []))
    if len(a) == 0:
        continue
    print(f"{cl:8s} | {len(a):3d} | {np.median(np.abs(a[:,0])):.3f} | {np.median(np.abs(a[:,1])):.3f} | "
          f"{np.median(np.maximum(a[:,2], a[:,3])):.3f}, {np.median(np.minimum(a[:,2], a[:,3])):.3f} | "
          f"{np.median(np.maximum(a[:,4], a[:,5])):.3f}, {np.median(np.minimum(a[:,4], a[:,5])):.4f}")
off = np.array(res["OFF"])
print(f"OFF heads: final |kappa| < init |kappa| in {(np.abs(off[:,1]) < np.abs(off[:,0])).sum()}/{len(off)}; "
      f"both move gates < 0.1 at the end in {((off[:,4] < 0.1) & (off[:,5] < 0.1)).sum()}/{len(off)}")
lat = np.array(res["LATTICE"])
print(f"LATTICE heads: both move gates < 0.1 at the end in {((lat[:,4] < 0.1) & (lat[:,5] < 0.1)).sum()}/{len(lat)}")
