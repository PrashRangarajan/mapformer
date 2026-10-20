"""POST HOC (CPU): does index attention on the torus build a GLOBAL count? Ridge probes, as ring_depth.py's, on the
FACTORIAL index checkpoints (L1, L4, LOOP; seeds 500-503), stream of eval_rank_strata (env seed 10000, np seed 1234+s),
walks 0-29 train / 30-39 test, action positions 2t, t >= 16. Targets per axis: unwrapped displacement P_t, the normalised
count P_t / (t+1) (what uniform causal attention over move tokens produces), and 1 / (t+1); d_w for w = 4, 16, 64, 256.
  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/torus_probe.py   -> out/torus_probe_out.txt"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import forward_capture, ridge_r2, OUTDIR  # noqa: E402
from torus_depth import load, DELTA  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
COLS = ["P", "P/t", "1/t", "d4", "d16", "d64", "d256"]


def main():
    L = ["ridge R^2 out of sample (x and y averaged where per-axis), mean over seeds 500-503; rows: residual after pass q"]
    for arm in ("L1", "L4", "LOOP"):
        acc = []
        for s in (500, 501, 502, 503):
            m = load(arm, s); env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234 + s)
            X, Y = [], []
            for w in range(40):
                tok, _o, _r = env.generate_trajectory(1024)
                _lg, res, _a = forward_capture(m, tok[None, :-1], keep_attn=False)
                u = np.cumsum(DELTA[tok[0::2].numpy()], 0); tt = np.arange(16, 1024)
                cols = [u[tt, 0], u[tt, 1], u[tt, 0] / (tt + 1), u[tt, 1] / (tt + 1), 1.0 / (tt + 1)]
                for wd in (4, 16, 64, 256):
                    pu = np.where((tt - wd)[:, None] >= 0, u[np.clip(tt - wd, 0, None)], 0)
                    cols += [u[tt, 0] - pu[:, 0], u[tt, 1] - pu[:, 1]]
                Y.append(np.stack(cols, 1)); X.append([r[0, 2 * tt].numpy() for r in res])
            Ytr, Yte = np.concatenate(Y[:30]), np.concatenate(Y[30:])
            r2 = []
            for q in range(len(X[0])):
                r = ridge_r2(np.concatenate([x[q] for x in X[:30]]), Ytr, np.concatenate([x[q] for x in X[30:]]), Yte)
                r2.append([r[0:2].mean(), r[2:4].mean(), r[4], r[5:7].mean(), r[7:9].mean(), r[9:11].mean(), r[11:13].mean()])
            acc.append(r2)
            print(arm, s, flush=True)
        a = np.array(acc).mean(0)
        L.append(f"  {arm}: " + " ".join(f"{c:>6s}" for c in COLS))
        for q in range(a.shape[0]):
            L.append(f"    q{q}: " + " ".join(f"{x:6.3f}" for x in a[q]))
    txt = "\n".join(L); print(txt)
    open(f"{OUTDIR}/torus_probe_out.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
