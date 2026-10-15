"""M5 (post hoc): the move-key gates g(+1), g(-1) and the mean obs gate of every period-P head (class from m2), with the
sigmoid slope g(1-g) of each move gate (how fast its content logit can move it), and the content logit of the move keys."""
import json, numpy as np
D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
m1 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m1_out.json"))}
m2 = json.load(open(D + "m2_out.json"))
rows = []
for o in m2:
    r = m1[(o["batch"], o["arm"], o["seed"])]
    for h, (cl, P, k) in enumerate(o["heads"]):
        if cl in ("LATTICE", "ALIASED", "OFF"):
            inc = r["inc"][f"h{h}a{o['a']}"]
            gp, gm = inc["g_plus"], inc["g_minus"]
            rows.append((cl, P, k, gp, gm, inc["cbar"], o["arm"], o["seed"], h, o["a"]))
print("class P k | g(+1) g(-1) cbar | min slope g(1-g) of the move gates | run")
for cl, P, k, gp, gm, cb, arm, s, h, a in sorted(rows, key=lambda x: (x[0], x[1])):
    sl = min(gp * (1 - gp), gm * (1 - gm))
    print(f"{cl:7s} P{P} k {k:6.2f} | {gp:.4f} {gm:.4f} {cb:.4f} | {sl:.5f} | {arm} s{s} h{h} a{a}")
for cl in ("LATTICE", "ALIASED", "OFF"):
    sel = [x for x in rows if x[0] == cl and x[1] == 2]
    mx = np.array([max(x[3], x[4]) for x in sel]); mn = np.array([min(x[3], x[4]) for x in sel])
    print(f"{cl:7s} P2 n {len(sel):2d}: larger move gate median {np.median(mx):.4f}, smaller median {np.median(mn):.5f}, "
          f"cbar median {np.median([x[5] for x in sel]):.4f}")
