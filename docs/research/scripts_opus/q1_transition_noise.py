"""POST HOC, CPU, eval-only (Opus 2026-10-29). Stochastic transitions: how fast does the path model's map stop serving
revisits when executed moves differ from the recorded ones? FACTORIAL_N8 Path_L1 (s2800-2803) and RoPE_L4, held-out map,
GridWorld(p_transition_noise=p) (token = commanded move; position follows a random move with prob. p). Accuracy on revisits
(executed position) by gap since the last visit. Models were trained at p = 0 (this is a robustness readout, rule 10)."""
import sys
import numpy as np, torch
sys.path.insert(0, "/home/prashr"); sys.path.insert(0, "/home/prashr/mapformer/docs/research/scripts_opus")
from mapformer.environment import GridWorld
from q1_revisit_taxonomy import load
torch.set_num_threads(8)
BINS = [(0, 16), (16, 64), (64, 256), (256, 1025)]
out = []
for p in (0.0, 0.01, 0.03, 0.1):
    acc = {k: {b: [] for b in BINS} for k in ("Path_L1", "RoPE_L4")}
    for s in (2800, 2801, 2802, 2803):
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234 + s)
        ms = {"Path_L1": load("factorial_n8", "L1", "Vanilla_r4", s), "RoPE_L4": load("factorial_n8", "L4", "RoPE", s)}
        for _ in range(6):
            tok, _om, rev = env.generate_trajectory(1024, p_transition_noise=p)
            locs = env.visited_locations; last = {}; gaps = {}
            for t, c in enumerate(locs):
                if c in last: gaps[t] = t - last[c]
                last[c] = t
            with torch.no_grad():
                pr = {k: m(tok[None, :-1]).argmax(-1)[0].numpy() for k, m in ms.items()}
            for t in range(1024):
                if rev[2 * t + 1] and t in gaps:
                    for b in BINS:
                        if b[0] <= gaps[t] < b[1]:
                            for k in pr: acc[k][b].append(pr[k][2 * t] == int(tok[2 * t + 1]))
    line = f"p {p:.2f}: " + " | ".join(f"{k} " + " ".join(f"gap{b[0]}-{b[1]} {np.mean(acc[k][b]):.3f}" for b in BINS) for k in acc)
    print(line, flush=True); out.append(line)
open("/home/prashr/mapformer/docs/research/scripts_opus/out/q1_transition_noise_out.txt", "w").write("\n".join(out) + "\n")
