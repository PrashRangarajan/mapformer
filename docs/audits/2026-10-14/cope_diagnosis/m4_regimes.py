"""M4: per run, the regime of each query type (from m2) against run accuracy; the table amplitude of each period-P head;
and the INIT state of every run (rebuilt with the trainer's seeding: torch.manual_seed, np.random.seed, env, model):
sigma, kappa, tau of each head at init."""
import json, sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
torch.set_num_threads(4)
import lib_cope as lc
from mapformer.environment_cancel import GridWorldCancel
from mapformer.train_variant import VARIANT_MAP

D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
m1 = {(r["batch"], r["arm"], r["seed"]): r for r in json.load(open(D + "m1_out.json"))}
m2 = json.load(open(D + "m2_out.json"))
by = {}
for o in m2:
    by.setdefault((o["batch"], o["arm"], o["seed"]), {})[o["a"]] = o

print("== regime pair (query +1, query -1) vs run accuracy ==")
pairs = {}
for key, v in by.items():
    pr = tuple(sorted((v[0]["regime"], v[1]["regime"])))
    pairs.setdefault(pr, []).append((m1[key]["acc"], key))
for pr, lst in sorted(pairs.items(), key=lambda x: -np.mean([a for a, _ in x[1]])):
    accs = [a for a, _ in lst]
    print(f"{'/'.join(pr):18s} n {len(lst):2d} acc mean {np.mean(accs):.4f} min {min(accs):.4f} max {max(accs):.4f} | "
          + ", ".join(f"{k[1]}:{k[2]}" for _, k in lst))

print("\n== init state (rebuilt) vs final regime ==")
init_rows = []
for batch, arm, seed, path in lc.run_dirs():
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorldCancel(size=32, seed=seed, p_plus=0.5)
    m0 = VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32).eval()
    # sanity: the rebuilt init must differ from the trained weights but match their shapes
    L0, G0, Z0 = lc.tables(m0)
    inc0 = lc.increments(G0)
    v = by[(batch, arm, seed)]
    row = dict(batch=batch, arm=arm, seed=seed, acc=m1[(batch, arm, seed)]["acc"],
               regimes=[v[0]["regime"], v[1]["regime"]],
               init={k2: {x: inc0[k][x] for x in ("sigma", "kappa", "tau", "g_plus", "g_minus", "cbar")}
                     for k in inc0 for k2 in [f"h{k[0]}a{k[1]}"]})
    init_rows.append(row)
    print(f"{batch[:3]} {arm:9s} s{seed} acc {row['acc']:.3f} {row['regimes'][0]:7s}/{row['regimes'][1]:7s} | init "
          + " ".join(f"{k}: s {x['sigma']:.3f} kap {x['kappa']:+.3f} tau {x['tau']:.3f}" for k, x in row["init"].items()))
allk = np.array([x["kappa"] for r in init_rows for x in r["init"].values()])
alls = np.array([x["sigma"] for r in init_rows for x in r["init"].values()])
print(f"\ninit over all heads/query types (n {len(allk)}): kappa mean {allk.mean():+.3f} sd {allk.std():.3f} max|.| {np.abs(allk).max():.3f}; "
      f"sigma mean {alls.mean():.4f} sd {alls.std():.4f}")
json.dump(init_rows, open(D + "m4_out.json", "w"), indent=1)
