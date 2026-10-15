"""M2: per (run, query type a) -- is there a LATTICE head, and does wrap-only revisit accuracy follow it?

Lattice head for query type a (theory F4, COPE_DIAGNOSIS.md Sec. 1): period P = alpha + beta = 2 sigma in {2, 3}
(|2 sigma - P| < 0.01), k = 32 alpha / P = 16 + kappa / sigma within 0.05 of an ODD integer (wrap-consistent and
injective on Z_32), and the effective table z_{a,h} has a period-P component with share > 0.15 peaking at p = 0 mod P.
'Aliased' head: same but k even (kernel 16Z or coarser). Regime of the run (per a): LATTICE / ALIASED-only /
OFF-LATTICE (a period-P head exists, k not an integer) / NONE (no period-P head).
Accuracy per query type from score() rows (400 eval trajectories, the registered eval stream)."""
import json, sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
torch.set_num_threads(8)
import lib_cope as lc

D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
m1 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m1_out.json"))}


def periodP(z, P, lo=0, hi=255):
    z = z[lo:hi].astype(np.float64); z = z - z.mean(); p = np.arange(lo, hi)
    c = (z * np.exp(-2j * np.pi * p / P)).mean()
    share = float(len(z) * (2 if P > 2 else 1) * abs(c) ** 2 / max((z ** 2).sum(), 1e-30))
    if P == 2:
        c = c.real
    return min(share, 1.0), float(abs(c)), float(np.angle(c)) if P > 2 else (0.0 if c > 0 else np.pi)


def head_class(inc, z):
    s, kap = inc["sigma"], inc["kappa"]
    P = 2 if abs(2 * s - 2) < 0.01 else (3 if abs(2 * s - 3) < 0.01 else None)
    if P is None:
        return "nonP", None, None
    share, amp, ph = periodP(z, P)
    if share < 0.15 or abs(ph) > 0.5:
        return "noTable", P, None
    k = 16 + kap / s
    kr = round(k)
    if abs(k - kr) < 0.05:
        return ("LATTICE" if kr % 2 == 1 else "ALIASED"), P, k
    return "OFF", P, k


trajs = lc.trajectories(400)
out = []; table_rows = []
for batch, arm, seed, path in lc.run_dirs():
    r = m1[(batch, arm, seed)]
    m, ck = lc.load(path)
    L, G, Z = lc.tables(m)
    acc, nll, rows = lc.score(m, trajs)
    for a in (0, 1):
        heads = []
        for h in range(2):
            cl, P, k = head_class(r["inc"][f"h{h}a{a}"], Z[h, a])
            heads.append((cl, P, None if k is None else round(k, 3)))
        cls = [hc[0] for hc in heads]
        regime = ("LATTICE" if "LATTICE" in cls else "ALIASED" if "ALIASED" in cls else "OFF" if "OFF" in cls else "NONE")
        sel = [x for x in rows if x["a"] == a]
        wr = [x["ok"] for x in sel if x["wrap_only"]]; lo_ = [x["ok"] for x in sel if not x["wrap_only"]]
        rec = dict(batch=batch, arm=arm, seed=seed, a=a, run_acc=r["acc"], run_cls=r["cls"], regime=regime, heads=heads,
                   acc_local=float(np.mean(lo_)), n_local=len(lo_), acc_wrap=float(np.mean(wr)), n_wrap=len(wr),
                   nll_local=float(np.mean([x["nll"] for x in sel if not x["wrap_only"]])),
                   nll_wrap=float(np.mean([x["nll"] for x in sel if x["wrap_only"]])))
        out.append(rec)
        print(f"{batch[:3]} {arm:9s} s{seed} a{a} run {r['acc']:.4f} {r['cls']:4s} | {regime:8s} | "
              + "  ".join(f"h{h}:{c}{'' if P is None else f' P{P}'}{'' if k is None else f' k{k:.2f}'}" for h, (c, P, k) in enumerate(heads))
              + f" | local {rec['acc_local']:.3f} (n {len(lo_)}) wrap-only {rec['acc_wrap']:.3f} (n {len(wr)})", flush=True)

print("\n== wrap-only accuracy by regime (per run x query type) ==")
for reg in ("LATTICE", "ALIASED", "OFF", "NONE"):
    sel = [o for o in out if o["regime"] == reg]
    if sel:
        w = np.array([o["acc_wrap"] for o in sel]); l = np.array([o["acc_local"] for o in sel])
        print(f"{reg:8s} n {len(sel):2d} | wrap-only mean {w.mean():.3f} min {w.min():.3f} max {w.max():.3f} | "
              f"local mean {l.mean():.3f} min {l.min():.3f} max {l.max():.3f}")
json.dump(out, open(D + "m2_out.json", "w"), indent=1)
