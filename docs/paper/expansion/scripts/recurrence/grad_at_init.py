"""Gradient reaching the step map at initialisation: 1 pass (A) vs the SAME weights looped x4 (L).

A and L share parameters and initialisation exactly; only the number of block applications differs. Here
both are built from one random init per seed (torch.manual_seed(seed), the trainer's seeding; whether
the bits equal the stored runs' inits is not needed for this comparison) and differentiated on the same
batches (B=16, T=1024 torus walks, loss on revisited observations, as train.py). Readouts per parameter
group: mean gradient norm over batches, and the gradient's batch-to-batch coherence
||mean_b g_b|| / mean_b ||g_b|| (1 = every batch pushes the same way). Also the split of the step-map
gradient into the part that would make opposite actions cancel (the antisymmetric map direction) and
the rest, read on the latent codes z_a = W_in e_a: d/d z of the loss, projected on (z_N - z_S) etc. is
not basis-free, so only norms and coherence are reported.

usage: python3 grad_at_init.py   (all 8 seeds, both arms; ~10 min CPU)
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT  # noqa
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.model_looped import MapFormerWM_Looped  # noqa: E402

torch.set_num_threads(8)
GROUPS = {"step_map": ("action_to_lie.",), "omega": ("path_integrator.",),
          "block": ("layers.",), "embed": ("token_emb.",), "head": ("out_",)}
NB, B, T = 6, 16, 1024
res = {}
for s in range(8):
    torch.manual_seed(s); np.random.seed(s)
    m = MapFormerWM_Looped(vocab_size=21, d_model=128, n_heads=2, grid_size=64, bottleneck_r=2)
    m.train()   # dropout as in training; masks drawn per forward
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10_000 + s)
    batches = []
    for _ in range(NB):
        tr = [env.generate_trajectory(T) for _ in range(B)]
        batches.append((torch.stack([t[0] for t in tr]), torch.stack([t[2] for t in tr])))
    out = {}
    for k in (1, 4):
        m.n_loops = k
        g = {n: [] for n in GROUPS}
        for tok, rev in batches:
            torch.manual_seed(1000 + s)                       # same dropout masks per batch across arms
            m.zero_grad()
            lg = m(tok[:, :-1])
            msk = rev[:, 1:].bool()
            loss = F.cross_entropy(lg[msk], tok[:, 1:][msk])
            loss.backward()
            for n, pre in GROUPS.items():
                v = torch.cat([p.grad.flatten() for nm, p in m.named_parameters()
                               if nm.startswith(pre) and p.grad is not None])
                g[n].append(v.clone())
        out[k] = {"loss": float(loss)}
        for n in GROUPS:
            G = torch.stack(g[n])
            out[k][n] = {"norm": float(G.norm(dim=1).mean()),
                         "coherence": float(G.mean(0).norm() / G.norm(dim=1).mean())}
    res[s] = out
    print(f"s{s} loss 1p {out[1]['loss']:.3f} 4p {out[4]['loss']:.3f} | " + " | ".join(
        f"{n} norm x{out[4][n]['norm'] / out[1][n]['norm']:.2f} coh {out[1][n]['coherence']:.2f}->{out[4][n]['coherence']:.2f}"
        for n in GROUPS), flush=True)
json.dump(res, open(f"{OUT}/grad_at_init.json", "w"), indent=1)
r = lambda n, q: np.array([res[s][4][n][q] / (res[s][1][n][q] if q == "norm" else 1) for s in res])
for n in GROUPS:
    print(f"{n:9s} norm ratio 4p/1p: mean {r(n, 'norm').mean():.2f} (min {r(n, 'norm').min():.2f}, max "
          f"{r(n, 'norm').max():.2f}); coherence 1p {np.mean([res[s][1][n]['coherence'] for s in res]):.2f} "
          f"-> 4p {np.mean([res[s][4][n]['coherence'] for s in res]):.2f}")
