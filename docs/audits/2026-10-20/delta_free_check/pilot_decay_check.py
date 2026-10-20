"""Audit 2026-10-20: how much of the pilot's free-delta movement is AdamW weight decay alone? (pilot seed 1690, outside
the batch; reads stored checkpoints only.) Decay-only trajectory: delta_T = D * init, D = prod_k (1 - lr_k * 0.05) over the
batch's cosine schedule (5% warmup, cosine to 0.1 x lr, lr 1e-3, 300 x 98 steps)."""
import math
import numpy as np
import torch

lr, wd, T = 1e-3, 0.05, 300 * 98
warm = max(1, int(0.05 * T))
f = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, T - warm), 1)))
D = float(np.prod([1 - lr * f(k) * wd for k in range(T)]))
print(f"sum lr {sum(lr * f(k) for k in range(T)):.3f}; decay-only factor D = {D:.4f}")
R = "/home/prashr/mapformer/runs/delta_free_pilot/p0"
for a in ("PoPE-FreeDelta", "MapPoPE-FreeDelta", "MapPoPE-FrozenDelta", "PoPE-Flat", "MapPoPE-Flat"):
    sd = torch.load(f"{R}/{a}_s1690/{a}.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    d = sd["layers.0.pope_delta"]
    if "layers.0.pope_delta_init" not in sd:
        print(f"{a:20s} raw min {d.min():+.3f} max {d.max():+.4f} dead zone {(d >= 0).float().mean():.3f}")
        continue
    i = sd["layers.0.pope_delta_init"]; r = d - D * i
    q = r.abs().mean(0).numpy()
    print(f"{a:20s} mean|d - init| {(d - i).abs().mean():.3f}  mean|d - D init| {r.abs().mean():.3f} (median {r.abs().median():.3f})"
          f"  |resid| by channel quartile fast->slow {[round(float(x), 3) for x in (q[:16].mean(), q[16:32].mean(), q[32:48].mean(), q[48:].mean())]}"
          f"  outside [-2pi, 0] {((d > 0) | (d < -2 * math.pi)).float().mean():.4f}  range [{d.min():+.3f}, {d.max():+.3f}]")
