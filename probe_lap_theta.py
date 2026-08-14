"""Did the model solve the lap task WITH path integration, or by breaking it?

The lap circuit has exactly zero net displacement, so a model doing faithful path
integration -- Delta(action) = true displacement, Delta(observation) = 0 -- has
theta return to the same value at the same cell on every lap. Laps then ALIAS in
theta and lap counting must come from content attention.

The escape hatch is to let observation tokens displace. Then theta accumulates
around a zero-displacement circuit and becomes a lap counter -- at the cost of the
property that makes it a cognitive map.

Reports per-lap |theta| drift and the |Delta| ratio between observation and action
tokens. Near-zero drift => counting via content. Large drift => path integration
was abandoned.
"""
import argparse, json
import numpy as np, torch
from mapformer.environment_lap import LapWorld
from mapformer.train_variant import VARIANT_MAP

def probe(ckpt, device="cuda:1", n_ep=60, seed=1):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    m = VARIANT_MAP[c["variant"]](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                  n_heads=c["n_heads"], n_layers=c["n_layers"],
                                  grid_size=64).to(device).eval()
    m.load_state_dict(c["model_state"])
    env = LapWorld(seed=10000); rng = np.random.RandomState(seed)
    drift, obs_d, act_d = [], [], []
    for _ in range(n_ep):
        t, _dp, _dl, info = env.generate_lap_episode(rng)
        L = info["loop_len"]
        with torch.no_grad():
            x = m.token_emb(t.unsqueeze(0).to(device))
            delta = m.action_to_lie(x)
            cos_a, sin_a = m.path_integrator(delta)
        th = torch.atan2(sin_a, cos_a)[0, 0]
        drift.append(np.mean([(th[2*(k*L)] - th[0]).abs().max().item() for k in (1,2,3)]))
        d = delta[0].norm(dim=-1)
        act_d.append(d[0::2].mean().item()); obs_d.append(d[1::2].mean().item())
    return dict(variant=c["variant"], theta_drift_per_lap=float(np.mean(drift)),
                delta_action=float(np.mean(act_d)), delta_obs=float(np.mean(obs_d)),
                obs_act_ratio=float(np.mean(obs_d)/max(np.mean(act_d), 1e-9)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="LAP_THETA_PROBE.json")
    a = ap.parse_args()
    rows = [probe(c, a.device) for c in a.checkpoints]
    for r in rows: print(r)
    json.dump(rows, open(a.out, "w"), indent=2)
