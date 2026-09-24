"""Old per-target evaluate() vs vectorised evaluate(): identical dicts and identical
JSON, on a TRAINED checkpoint, CPU, single thread, a few trials."""
import sys, json, time, importlib
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency")
OS = importlib.import_module("mapformer.eval_rank_strata"); NS = importlib.import_module("mfprop.eval_rank_strata")
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
ok = True
for v, s in (("Vanilla", 2), ("Vanilla_r4", 3)):
    b = torch.load(f"/home/prashr/mapformer/runs/rank_matched_e900/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
    c = b["config"]; m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                                        n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    for T in (256, 1024):
        t0 = time.perf_counter(); ro = OS.evaluate(m, env, T, 3, 1234 + s, "cpu"); t1 = time.perf_counter()
        rn = NS.evaluate(m, env, T, 3, 1234 + s, "cpu"); t2 = time.perf_counter()
        same = ro == rn and json.dumps(ro) == json.dumps(rn)
        ok &= same
        print(f"{v} s{s} T={T}: {'IDENTICAL' if same else 'DIFFERENT'}  old {t1-t0:.2f}s new {t2-t1:.2f}s  "
              f"all n={ro['all']['n']} acc={ro['all']['acc']:.4f}")
# kinds_vec vs kinds on random walks, including long ones that wrap the torus
env = GridWorld(size=8, n_obs_types=4, p_empty=0.5, seed=1)
for sd in range(20):
    np.random.seed(sd); tok, _, _ = env.generate_trajectory(300)
    k = OS.kinds(tok, env.ACTION_DELTAS, env.size); nu, lag = NS.kinds_vec(tok, env.size)
    ok &= [x[0] for x in k] == nu.tolist() and [x[1] for x in k] == lag.tolist()
print("ALL IDENTICAL" if ok else "MISMATCH")
