"""Push the BLIND query phase far past training length, on existing checkpoints.

Inference only. T_query is an eval-time parameter, so this needs no retraining.
The query phase is blind throughout, so doubling it doubles the distance over
which the map must hold with no observations to re-anchor on.
"""
import argparse, json, statistics as st
from pathlib import Path
import torch
from mapformer.environment_match_query import MatchQueryGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_match_query import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="mapformer/runs/match_query")
ap.add_argument("--variants", nargs="+", default=["Vanilla", "PlainFlat"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
ap.add_argument("--size", type=int, default=64); ap.add_argument("--n-obs", type=int, default=16)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default="MATCH_QUERY_LONGQ.md")
a = ap.parse_args()
env = MatchQueryGridWorld(size=a.size, n_obs_types=a.n_obs, seed=10000)
res = {v: {L: [] for L in a.lengths} for v in a.variants}
for v in a.variants:
    for s in a.seeds:
        cp = Path(a.runs_dir) / f"seed{s}" / f"{v}_matchquery.pt"
        if not cp.exists(): print("MISSING", cp); continue
        c = torch.load(cp, map_location=a.device, weights_only=False)
        m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                           n_heads=c["n_heads"], n_layers=c["n_layers"],
                           grid_size=a.size).to(a.device).eval()
        m.load_state_dict(c["model_state"])
        for L in a.lengths:
            acc, _nll, _n = evaluate(m, env, 512, L, 6, 6, a.device, 9000 + s)
            res[v][L].append(acc); print(f"[{v} s{s}] TQ={L}: {acc:.4f}", flush=True)
def cell(xs): return "n/a" if not xs else f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else "")
ln = ["# Match-Query: blind query phase far past training length", "",
      f"Inference only on existing checkpoints. Trained at T_query=256. chance = {1.0/a.n_obs:.4f}.", "",
      "| variant | " + " | ".join(f"TQ={L}" for L in a.lengths) + " |", "|---" * (len(a.lengths) + 1) + "|"]
for v in a.variants: ln.append(f"| {v} | " + " | ".join(cell(res[v][L]) for L in a.lengths) + " |")
Path(a.out).write_text("\n".join(ln) + "\n"); json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=2)
print("\n".join(ln))
