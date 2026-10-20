"""Walk statistics behind the depth theory (CPU, no model). On each task's evaluation stream:
  - sd of the displacement over the last w moves (per axis), to turn probe R^2 into an error in cells:
    rmse = sd * sqrt(1 - R^2)
  - sd of the unwrapped displacement P_t at t = T (the walk's excursion), in grid widths
  - share of revisits whose gap exceeds w, and the mean number of runs spanned per gap bin
  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/walk_stats.py   -> out/walk_stats_out.txt"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import OUTDIR  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.environment_cancel import GridWorldCancel  # noqa: E402

DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)


def main():
    L = []
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234 + 500)
    U = []
    for _ in range(100):
        tok, _o, _r = env.generate_trajectory(1024)
        U.append(np.cumsum(DELTA[tok[0::2].numpy()], 0))
    U = np.stack(U)                                     # (100, 1024, 2)
    L.append("torus 64x64, T = 1024 (FACTORIAL stream, seed 500):")
    L.append(f"  sd of P_T per axis {U[:, -1].std():.1f} cells (= {U[:, -1].std() / 64:.2f} grid widths); "
             f"sd of P_t at t = 128: {U[:, 127].std():.1f}")
    for w in (4, 16, 64, 256):
        d = U[:, w:] - U[:, :-w]
        L.append(f"  window w = {w:4d}: sd of d_w per axis {d.std():.2f} cells")
    for p in (0.5, 0.75, 0.9, 1.0):
        te = GridWorldCancel(size=32, seed=10000, p_plus=p); np.random.seed(0)
        U = []
        for _ in range(200):
            tok, _o, _r = te.generate_trajectory(128)
            U.append(np.cumsum(np.where((tok[0::2] - te.action_offset).numpy() == 0, 1, -1)))
        U = np.stack(U); tt = np.arange(8, 128)
        L.append(f"ring 32, T = 128, p_plus {p}: sd of P_T {U[:, -1].std():.1f} (= {U[:, -1].std() / 32:.2f} ring widths), "
                 f"mean |P_T| {np.abs(U[:, -1]).mean():.1f}; sd of P_t/(t+1) over t >= 8 {(U[:, tt] / (tt + 1)).std():.3f}")
        for w in (4, 16, 64):
            d = U[:, w:] - U[:, :-w]
            L.append(f"    w = {w:3d}: sd of d_w {d.std():.2f}")
        L.append(f"    sd of P_t over t >= 8: {U[:, tt].std():.2f}")
    txt = "\n".join(L); print(txt)
    open(f"{OUTDIR}/walk_stats_out.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
