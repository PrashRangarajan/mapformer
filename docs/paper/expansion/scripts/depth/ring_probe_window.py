"""Is the ring index models' 'global count' a count or a recent-window code? (POST HOC, eval only, CPU, ~2 min)

Written for docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md M-DP1. ring_depth.py's probes read action positions
t >= 8 of T = 128 walks, where half of P_t / t's variance sits at t <= 32 and for t < 64 the 64-move window d64 equals
P_t, so a probe cannot tell a prefix count from a window code. Here the same stream (CANCEL eval stream: GridWorldCancel(32,
seed 10000, p_plus), np seed 0, 200 walks; ridge probes trained on walks 0-149, tested on 150-199, residual stream at the
move tokens, as ring_depth.py) is probed only at late positions and on targets a window code cannot supply:
  t >= 64: P_t and P_t / (t + 1);
  t >= 96: P_{t-32} and P_{t-64} (displacement before the last 32 / 64 moves), d32 and d64 (window codes);
  oracle: R^2 of P_t at t >= 96 from the true d64 alone (least squares with intercept, same train / test split).
Index RoPE at 1-3 layers, p_plus 0.5, fresh seeds 2-7. Per seed and pass, out-of-sample R^2; reported: mean over seeds at
the pass whose seed-mean is largest ("best pass"), and per pass.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/expansion/scripts/depth/ring_probe_window.py \
      > mapformer/docs/paper/expansion/scripts/depth/out/ring_probe_window_out.txt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import forward_capture, ridge_r2, REPO  # noqa: E402
from mapformer.environment_cancel import GridWorldCancel  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
SIZE, T, NW, P_PLUS = 32, 128, 200, 0.5
SEEDS = range(2, 8)


def targets(u, tt, specs):
    cols = []
    for kind, w in specs:
        if kind == "P":
            cols.append(u[tt])
        elif kind == "P/t":
            cols.append(u[tt] / (tt + 1))
        elif kind == "Pback":                     # displacement up to move t - w
            cols.append(u[tt - w])
        elif kind == "d":                          # displacement over the last w moves
            cols.append(u[tt] - u[tt - w])
    return np.stack(cols, 1)


BLOCKS = [("t >= 64", 64, [("P", 0), ("P/t", 0)]),
          ("t >= 96", 96, [("P", 0), ("Pback", 32), ("Pback", 64), ("d", 32), ("d", 64)])]
NAMES = {("P", 0): "P_t", ("P/t", 0): "P_t/(t+1)", ("Pback", 32): "P_{t-32}", ("Pback", 64): "P_{t-64}",
         ("d", 32): "d32", ("d", 64): "d64"}

print(__doc__.split("\n")[0])
res = {}
oracle = []
for nl in (1, 2, 3):
    for s in SEEDS:
        d = f"{REPO}/runs/cancel/p0/RoPE_L{nl}_p{P_PLUS}_s{s}"
        b = torch.load(f"{d}/RoPE.pt", map_location="cpu", weights_only=False)
        m = VARIANT_MAP["RoPE"](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=nl, grid_size=SIZE)
        m.load_state_dict(b["model_state_dict"]); m.eval()
        te = GridWorldCancel(size=SIZE, seed=10000, p_plus=P_PLUS); np.random.seed(0)
        toks, U = [], []
        for _ in range(NW):
            tok, _o, _rev = te.generate_trajectory(T)
            mv = (tok[0::2] - te.action_offset).numpy()
            U.append(np.cumsum(np.where(mv == 0, 1, -1))); toks.append(tok)
        X = torch.stack(toks)[:, :-1]
        _logits, rs, _att = forward_capture(m, X)
        for bname, t0, specs in BLOCKS:
            tt = np.arange(t0, T)
            Y = np.stack([targets(u, tt, specs) for u in U])            # (NW, nt, ncol)
            for q, r in enumerate(rs):
                Xr = r[:, 2 * tt].numpy()
                r2 = ridge_r2(Xr[:150].reshape(-1, 128), Y[:150].reshape(-1, Y.shape[-1]),
                              Xr[150:].reshape(-1, 128), Y[150:].reshape(-1, Y.shape[-1]))
                res.setdefault((nl, bname, q), []).append(r2)
        if nl == 1:                                                      # oracle is model-free; once per seed is enough
            tt = np.arange(96, T)
            Y = np.stack([targets(u, tt, [("P", 0), ("d", 64)]) for u in U])
            xtr, ytr = Y[:150, :, 1].reshape(-1), Y[:150, :, 0].reshape(-1)
            xte, yte = Y[150:, :, 1].reshape(-1), Y[150:, :, 0].reshape(-1)
            A = np.stack([xtr, np.ones_like(xtr)], 1); w = np.linalg.lstsq(A, ytr, rcond=None)[0]
            pr = w[0] * xte + w[1]
            oracle.append(1 - ((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
        print(f"  RoPE {nl}L s{s} done", flush=True)

for bname, _t0, specs in BLOCKS:
    print(f"\n== {bname}: out-of-sample R^2, mean over fresh seeds 2-7 (n 6)")
    print("| model | pass | " + " | ".join(NAMES[sp] for sp in specs) + " |")
    print("|---|---|" + "---|" * len(specs))
    for nl in (1, 2, 3):
        qs = sorted(q for (a, b, q) in res if a == nl and b == bname)
        for q in qs:
            m = np.mean(res[(nl, bname, q)], 0)
            print(f"| RoPE {nl}L | {q} | " + " | ".join(f"{x:.3f}" for x in m) + " |")
        best = {sp: max(np.mean(res[(nl, bname, q)], 0)[i] for q in qs) for i, sp in enumerate(specs)}
        print(f"| RoPE {nl}L | best | " + " | ".join(f"{best[sp]:.3f}" for sp in specs) + " |")
print(f"\noracle: R^2 of P_t (t >= 96) from the true d64 alone, mean over the 6 seeds' streams {np.mean(oracle):.3f}")
