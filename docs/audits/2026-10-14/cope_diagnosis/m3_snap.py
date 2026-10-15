"""M3 (P7): eval-only gate-snap. In every head whose (sigma, table) is period-P (P = 2 or 3, class LATTICE / ALIASED / OFF
in m2), replace the gates used for the COUNT by a clean table: obs gates all = cbar (tau -> 0), alpha + beta = P exactly,
and kappa per variant; the attention content logits, the position table and every other weight are unchanged.
  S0 own gates (no-op control)   S1 clean (sigma exact, tau 0, kappa kept)
  S2 registered target (P7 as written: P = 2 heads kappa -> sign(kappa) * 1, i.e. alpha - beta = +/- 1/8; P = 3 heads
     -> nearest odd k)            S3 nearest odd k (kappa = (k_odd - 16) sigma)
Then a PATH (P8 on the learned models): for runs where some head is OFF, kappa moved linearly from its learned value to
the S3 target in 10 steps (all such heads together, sigma/tau clean): accuracy and NLL along the path."""
import json, sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
torch.set_num_threads(8)
import lib_cope as lc

D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
src = open(D + "m2_per_query.py").read().split("trajs = lc.trajectories")[0]
ns = {"__name__": "m2lib"}; exec(src, ns)
head_class = ns["head_class"]
m1 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m1_out.json"))}


def snapped(G, Z, inc_all, mode, t=1.0):
    Gs = G.copy(); touched = []
    for h in range(G.shape[0]):
        for a in (0, 1):
            inc = inc_all[f"h{h}a{a}"]
            cl, P, k = head_class(inc, Z[h, a])
            if cl not in ("LATTICE", "ALIASED", "OFF"):
                continue
            s = P / 2; kap = inc["kappa"]
            if mode == "S1":
                kt = kap
            elif mode == "S2":
                kt = (np.sign(kap) or 1.0) * 1.0 if P == 2 else None
            if mode == "S3" or (mode == "S2" and P == 3):
                ko = 2 * np.floor(k / 2) + 1                                     # nearest odd
                ko = ko if abs(ko - k) <= 1 else ko + 2
                kt = (ko - 16) * s
            if mode == "PATH":
                ko = 2 * np.floor(k / 2) + 1
                kt = kap + t * ((ko - 16) * s - kap)
            al, be = s + kt / 16, s - kt / 16
            cb = inc["cbar"]
            lo, hi = max(al - 1, be - 1, 0.0), min(al, be, 1.0)
            cb = float(np.clip(cb, lo + 1e-6, hi - 1e-6))
            Gs[h, a, 0] = al - cb; Gs[h, a, 1] = be - cb; Gs[h, a, 2:19] = cb
            touched.append((h, a, cl, round(kap, 3), round(kt, 3)))
    return Gs, touched


def strat(rows):
    w = [x for x in rows if x["wrap_only"]]; l = [x for x in rows if not x["wrap_only"]]
    return (np.mean([x["ok"] for x in l]), np.mean([x["ok"] for x in w]), np.mean([x["nll"] for x in rows]))


trajs = lc.trajectories(200)
out = []
for batch, arm, seed, path in lc.run_dirs():
    r = m1[(batch, arm, seed)]
    m, ck = lc.load(path)
    L, G, Z = lc.tables(m)
    rec = dict(batch=batch, arm=arm, seed=seed, cls=r["cls"], snaps={})
    line = f"{batch[:3]} {arm:9s} s{seed} {r['cls']:4s}"
    for mode in ("S0", "S1", "S2", "S3"):
        Gs, touched = (G, []) if mode == "S0" else snapped(G, Z, r["inc"], mode)
        acc, nll, rows = lc.score(m, trajs, Gsnap=Gs)
        lo_, wr, nl = strat(rows)
        rec["snaps"][mode] = dict(acc=acc, nll=nll, local=lo_, wrap=wr, touched=touched)
        line += f" | {mode} {acc:.3f} (loc {lo_:.3f} wrap {wr:.3f} nll {nll:.3f})"
    print(line, "| touched", rec["snaps"]["S3"]["touched"], flush=True)
    off = [x for x in rec["snaps"]["S3"]["touched"] if x[2] == "OFF"]
    if off:
        pth = []
        for t in np.linspace(0, 1, 11):
            Gs, _ = snapped(G, Z, r["inc"], "PATH", t)
            acc, nll, rows = lc.score(m, trajs, Gsnap=Gs)
            lo_, wr, nl = strat(rows)
            pth.append(dict(t=float(t), acc=acc, nll=nll, local=lo_, wrap=wr))
        rec["path"] = pth
        print("   path t: " + " ".join(f"{p['t']:.1f}:{p['acc']:.3f}/{p['nll']:.3f}" for p in pth), flush=True)
    out.append(rec)
json.dump(out, open(D + "m3_out.json", "w"), indent=1, default=float)
