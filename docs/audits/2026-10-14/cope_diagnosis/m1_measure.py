"""M1: per run (4 WRAP_COPE arms x 8 seeds + GENERALITY CoPEF x 8): learned gates by key type for the two
action-token query types, increments alpha/beta/sigma/delta, lattice distance, obs-gate spread, period-2 share of each
effective table, and revisit accuracy stratified (local vs wrap-only; local by gap N). Writes m1_out.json."""
import json, sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
torch.set_num_threads(8)
import lib_cope as lc

CLASS = {}
for line in open(lc.REPO / "WRAP_COPE_ANALYSIS.txt"):
    if line[:3].isdigit() and "|" in line:
        parts = [p.split() for p in line.split("|")]
        s = int(parts[0][0])
        for p in parts[1:]:
            CLASS[(p[0], s)] = p[2]

trajs = lc.trajectories(400)
out = []
for batch, arm, seed, path in lc.run_dirs():
    m, ck = lc.load(path)
    L, G, Z = lc.tables(m)
    inc = lc.increments(G)
    tabs = {}
    for h in range(2):
        for a in (0, 1):
            share, amp = lc.period2(Z[h, a])
            tabs[f"h{h}a{a}"] = dict(p2_share=share, p2_amp=amp, z_sd=float(Z[h, a, :255].std()),
                                     z_first=Z[h, a, :24].round(3).tolist())
    acc, nll, rows = lc.score(m, trajs)
    st = lc.strat_table(rows)
    losses = ck["losses"]; fl = float(np.mean(losses[-max(1, len(losses) // 20):]))
    cls = CLASS.get((arm, seed), "SOLV" if (fl < 0.05 and acc > 0.99) else "?")
    rec = dict(batch=batch, arm=arm, seed=seed, acc=acc, nll=nll, final_loss=fl, cls=cls,
               inc={f"h{h}a{a}": v for (h, a), v in inc.items()}, tabs=tabs,
               L_content={f"h{h}a{a}": L[h, a].round(4).tolist() for h in range(2) for a in (0, 1)},
               strata=st)
    out.append(rec)
    i00 = inc[(0, 0)]; i10 = inc[(1, 0)]
    print(f"{batch[:3]} {arm:9s} s{seed} acc {acc:.4f} cls {cls} | "
          + " | ".join(f"h{h}a{a}: s {inc[(h,a)]['sigma']:.4f} kap {inc[(h,a)]['kappa']:+.3f} tau {inc[(h,a)]['tau']:.3f} "
                       f"p2 {tabs[f'h{h}a{a}']['p2_share']:.2f}/{tabs[f'h{h}a{a}']['p2_amp']:+.2f}"
                       for h in range(2) for a in (0, 1))
          + f" | local {st['local'][0]:.3f} wrap {st['wrap_only'][0]:.3f}", flush=True)
json.dump(out, open("/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/m1_out.json", "w"), indent=1)
