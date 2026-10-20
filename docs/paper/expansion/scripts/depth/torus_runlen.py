"""POST HOC (CPU, eval-only): is index attention's reach on the torus limited by the number of SEGMENTS (direction runs)
a revisit spans, or by the GAP in moves? FACTORIAL index checkpoints (L1, L4, LOOP) and the path model, evaluated on
walks whose run lengths are redrawn from other distributions inside the training support (1..10 moves):
  U1-10   the training walk (direction uniform over 4, run length uniform 1..10)
  F2      every run 2 moves          F5   every run 5 moves          F10   every run 10 moves
Same held-out map (GridWorld env seed 10000), N walks of T = 1024 per seed and condition (np seed 7000 + s; N = 60 for L1, 30 for L4 / LOOP, 10 for PATH). Accuracy on
plain revisits by (segments spanned, gap) cell, pooled over seeds 500-503 (cells with >= 50 revisits).
Prediction of a segment-limited mechanism: within a segment bin, accuracy does not depend on the condition (gap shifts);
of a gap-limited one: within a gap bin, accuracy does not depend on segments.
  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/torus_runlen.py ARM[,ARM] N_WALKS
  -> out/torus_runlen_<ARMS>_out.txt"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import OUTDIR  # noqa: E402
from torus_depth import load, labels, DELTA  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
CONDS = {"U1-10": None, "F2": 2, "F5": 5, "F10": 10}
SB = [(2, 3), (3, 5), (5, 9), (9, 17), (17, 10 ** 6)]
GB = [(2, 16), (16, 64), (64, 128), (128, 1025)]


def walk(env, T, k):
    x, y = np.random.randint(0, 64, 2)
    acts = []
    while len(acts) < T:
        a = np.random.randint(0, 4); kk = np.random.randint(1, 11) if k is None else k
        acts += [a] * kk
    a = np.array(acts[:T]); d = DELTA[a]
    xs = (x + np.cumsum(d[:, 0])) % 64; ys = (y + np.cumsum(d[:, 1])) % 64
    tok = np.empty(2 * T, np.int64); tok[0::2] = a + env.action_offset
    tok[1::2] = env.obs_map.numpy()[xs, ys] + env.obs_offset
    cell = xs * 64 + ys; seen = set(); rev = np.zeros(T, bool)
    for t in range(T):
        rev[t] = cell[t] in seen; seen.add(cell[t])
    return torch.as_tensor(tok), rev


def main():
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["L1", "L4", "LOOP", "PATH"]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out = [f"plain revisits (not wrap-only); cells: accuracy [n], pooled over seeds 500-503, {nw} walks each"]
    for arm in arms:
        res = {c: [] for c in CONDS}
        for s in (500, 501, 502, 503):
            m = load(arm, s)
            for c, k in CONDS.items():
                np.random.seed(7000 + s)
                for _ in range(nw):
                    tok, rev = walk(env, 1024, k)
                    with torch.no_grad():
                        pred = m(tok[None, :-1])[0].argmax(-1)
                    lb = labels(tok)
                    ts = np.nonzero(rev & ~lb["wrap"])[0]
                    corr = (pred[2 * ts] == tok[2 * ts + 1]).numpy()
                    res[c].append(np.stack([corr, lb["nruns"][ts], lb["lag"][ts], lb["retr"][ts]], 1))
            print(arm, s, flush=True)
        out.append(f"\n{arm}: rows segments spanned, columns gap (moves)")
        for c in CONDS:
            R = np.concatenate(res[c])
            out.append(f"  {c:6s} all plain {R[:, 0].mean():.3f}; retrace {R[R[:, 3] == 1, 0].mean():.3f}")
            for s0, s1 in SB:
                cells = []
                for g0, g1 in GB:
                    sel = (R[:, 1] >= s0) & (R[:, 1] < s1) & (R[:, 2] >= g0) & (R[:, 2] < g1)
                    cells.append(f"{R[sel, 0].mean():.3f} [{sel.sum():5d}]" if sel.sum() >= 50 else "   --        ")
                out.append(f"    seg {s0:2d}-{(s1 - 1) if s1 < 10 ** 6 else '+':>2}: " + "  ".join(cells))
        print("\n".join(out[-(1 + len(CONDS) * (1 + len(SB))):]), flush=True)
    out.insert(1, "gap bins: " + ", ".join(f"{a}-{b - 1}" for a, b in GB))
    open(f"{OUTDIR}/torus_runlen_{'_'.join(arms)}_out.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
