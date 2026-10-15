"""Summary of M3 (gate-snap, P7) by run regime (m2). Groups: runs with an OFF query type and no NONE query type
(the theory's target), runs with a NONE query type, runs with both query types LATTICE/ALIASED."""
import json, numpy as np
D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
m3 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m3_out.json"))}
m2 = json.load(open(D + "m2_out.json"))
reg = {}
for o in m2:
    reg.setdefault((o["batch"], o["arm"], o["seed"]), []).append(o["regime"])
groups = {"OFF, no NONE": [], "has NONE": [], "LATTICE/ALIASED only": []}
for k, rr in reg.items():
    g = "has NONE" if "NONE" in rr else ("OFF, no NONE" if "OFF" in rr else "LATTICE/ALIASED only")
    groups[g].append(k)
for g, ks in groups.items():
    print(f"== {g}: n {len(ks)} ==")
    for mode in ("S0", "S1", "S2", "S3"):
        acc = np.array([m3[k]["snaps"][mode]["acc"] for k in ks]); wr = np.array([m3[k]["snaps"][mode]["wrap"] for k in ks])
        nll = np.array([m3[k]["snaps"][mode]["nll"] for k in ks]); lo = np.array([m3[k]["snaps"][mode]["local"] for k in ks])
        print(f"  {mode}: acc mean {acc.mean():.4f} (min {acc.min():.3f}) | local {lo.mean():.4f} | wrap-only {wr.mean():.3f} | "
              f"nll {nll.mean():.3f} | acc >= 0.99: {(acc >= 0.99).sum()}/{len(ks)}")
    if g == "OFF, no NONE":
        d = np.array([m3[k]["snaps"]["S3"]["acc"] - m3[k]["snaps"]["S0"]["acc"] for k in ks])
        print(f"  S3 - S0 acc: mean {d.mean():+.4f}, positive in {(d > 0).sum()}/{len(ks)}; per run: "
              + ", ".join(f"{k[1]}:{k[2]} {x:+.3f}" for k, x in zip(ks, d)))
        # path: is there an interior point with NLL above both ends?
        bar = []
        for k in ks:
            p = m3[k].get("path")
            if p:
                n = [x["nll"] for x in p]
                bar.append((k, max(n[1:-1]) - max(n[0], n[-1]), n[0], max(n[1:-1]), n[-1]))
        print(f"  path barrier (max interior NLL - max(end NLLs)) > 0 in {sum(b[1] > 0 for b in bar)}/{len(bar)}; "
              + ", ".join(f"{b[0][1]}:{b[0][2]} {b[2]:.3f}->{b[3]:.3f}->{b[4]:.3f}" for b in bar))
