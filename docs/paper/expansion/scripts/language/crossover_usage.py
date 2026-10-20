"""Same architecture, two data streams: is the clock in the used channels? (POST HOC, eval only, CPU.)

The unconstrained (signed) rank-4 one-layer MapWM trained on the 2D torus at matched length (runs/sign_matched/p0,
Signed_r4, T = 1024 moves = 2048 tokens, 8 seeds) and on k-back recency with fillers (runs/recency, Signed_r4,
k_max 64, p_filler 0.5, T = 1024 tokens, 8 seeds). Readouts as usage_tick.py: per channel rho_c = |mu_c| L /
range(content_c, L), phi_c = turns of the clock part per window, all-token usage u_c; usage share in clock-dominated
channels (rho > 1 and phi > 0.5) and usage-weighted median rho. Prediction of the drift/content account: the torus
(targets keyed to a latent cell reached at widely varying lags) puts its used channels at rho << 1; recency (targets
keyed to a count of items back) at rho >> 1, in the same architecture.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from usage_tick import per_channel, usage_alltok, summarize  # noqa
from mapformer.environment import GridWorld  # noqa
from mapformer.environment_recency import RecencyWorld  # noqa
from mapformer.train_variant import VARIANT_MAP  # noqa

torch.set_num_threads(8)
REPO = "/home/prashr/mapformer"
D_ = f"{REPO}/docs/paper/expansion/scripts/language"


def main():
    out = {"torus": {}, "recency": {}}
    print("torus, Signed_r4, matched length (L = 2048 tokens):")
    np.random.seed(424244)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    W = np.stack([env.generate_trajectory(1024)[0].numpy() for _ in range(32)])
    for s in range(8):
        b = torch.load(f"{REPO}/runs/sign_matched/p0/Signed_r4_s{s}/Signed_r4.pt", map_location="cpu", weights_only=False)
        c = b["config"]
        m = VARIANT_MAP["Signed_r4"](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=64)
        m.load_state_dict(b["model_state_dict"]); m.eval()
        V = c["vocab_size"]
        f = np.bincount(W.ravel(), minlength=V).astype(float); f /= f.sum()
        st = m.action_to_lie(m.token_emb.weight[None])[0].reshape(V, -1).detach().numpy()
        om = m.path_integrator.omega.detach().numpy().reshape(-1)
        rho, mu = per_channel(st, W, f, W.shape[1])
        u = usage_alltok(m, W[:8])
        out["torus"][f"s{s}"] = summarize(f"torus s{s}", rho, mu, om, u, W.shape[1])
    print("recency (k-back, k_max 64, fillers), Signed_r4 (L = 1024 tokens):")
    env = RecencyWorld(seed=10000, n_symbols=16, k_max=64, p_query=0.25, min_gap=64)
    rng = np.random.RandomState(424245)
    W = []
    while len(W) < 32:
        t = env.generate_episode(1100, rng)[0].numpy()
        if len(t) >= 1024:
            W.append(t[:1024])
    W = np.stack(W)
    for s in range(8):
        b = torch.load(f"{REPO}/runs/recency/Signed_r4_s{s}/Signed_r4_recency.pt", map_location="cpu", weights_only=False)
        V = b["vocab_size"]
        m = VARIANT_MAP["Signed_r4"](vocab_size=V, d_model=128, n_heads=2, n_layers=1, grid_size=b["grid_size"])
        m.load_state_dict(b["model_state"]); m.eval()
        f = np.bincount(W.ravel(), minlength=V).astype(float); f /= f.sum()
        st = m.action_to_lie(m.token_emb.weight[None])[0].reshape(V, -1).detach().numpy()
        om = m.path_integrator.omega.detach().numpy().reshape(-1)
        rho, mu = per_channel(st, W, f, 1024)
        u = usage_alltok(m, W[:8])
        out["recency"][f"s{s}"] = summarize(f"recency s{s}", rho, mu, om, u, 1024)
    json.dump(out, open(f"{D_}/crossover_usage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
