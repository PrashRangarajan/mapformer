"""MODEL-FREE, CPU (Opus scope expansion 2026-10-29). Cover-space revisit shares across regimes, to predict where a map
(path integration) is needed and where depth / local rules suffice.

For each regime: walks of runs of 1-10 repeated moves (the GridWorld policy; the torus regimes call GridWorld itself), and
per revisit its cover class: TREE (free-group / stack revisit: reversal or nested reversal), PLANE-only (needs
commutation), TORUS-only (needs wrap). Also: the plane+torus share at gap >= 16 moves (beyond the very-short loops that
four index layers partly solve, q1_cover_strata), the retrace-applies share, and EDGE: among plane/torus revisits, the
share whose entering edge (previous cell -> current cell, either direction) was traversed before (the most a
transition-graph / induction mechanism could serve with unique views).
Bounded rooms ('room L') are a SURROGATE for MiniWorld OneRoom (4 moves, a run that hits the wall stops and a new run
starts; no stay tokens): not MiniWorld's dynamics.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld  # noqa: E402

INV = {0: 1, 1: 0, 2: 3, 3: 2}
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def room_walk(L, T, rng):
    p = rng.integers(0, L, 2); acts = []; cells = []
    while len(acts) < T:
        a = int(rng.integers(0, 4)); k = int(rng.integers(1, 11))
        for _ in range(k):
            q = p + DELTA[a]
            if (q < 0).any() or (q >= L).any():
                break
            p = q; acts.append(a); cells.append((int(p[0]), int(p[1])))
            if len(acts) == T:
                break
    return np.array(acts), cells


def classify(acts, cells):
    stack = []; st = {}; pl = {}; ce = {}; edges = set(); P = np.zeros(2, int)
    out = []; prev_cell = None; prev_run = cur = 0
    for t, a in enumerate(acts):
        if t > 0 and a == acts[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a == INV[acts[t - 1]]) else 0
            cur = 1
        if stack and stack[-1] == INV[a]:
            stack.pop()
        else:
            stack.append(a)
        P = P + DELTA[a]; node = tuple(stack); pp = (int(P[0]), int(P[1])); c = cells[t]
        if c in ce:
            cls = "tree" if node in st else ("plane" if pp in pl else "torus")
            e = (prev_cell, c) in edges or (c, prev_cell) in edges
            out.append((cls, t - ce[c], prev_run > 0 and cur <= prev_run, e))
        st[node] = t; pl[pp] = t; ce[c] = t
        if prev_cell is not None:
            edges.add((prev_cell, c))
        prev_cell = c
    return out


def summarise(name, rows):
    R = np.array([(r[0] == "tree", r[0] == "plane", r[0] == "torus", r[1], r[2], r[3]) for r in rows], dtype=float)
    tree, plane, torus, gap, rt, edge = R.T
    need = (plane + torus) > 0
    print(f"{name:28s} revisits/walk {len(rows)/N:7.1f} | tree {tree.mean():.3f} (retrace {rt.mean():.3f}) plane-only {plane.mean():.3f} "
          f"torus-only {torus.mean():.3f} | need-map {need.mean():.3f}, at gap>=16 {(need & (gap >= 16)).mean():.3f}, gap>=128 "
          f"{(need & (gap >= 128)).mean():.3f} | edge-recallable among need-map {edge[need > 0].mean() if need.any() else float('nan'):.3f}")


N = 40
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for size, T in ((64, 128), (64, 132), (64, 512), (64, 1024), (64, 4096), (32, 1024), (16, 1024), (128, 1024)):
        env = GridWorld(size=size, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1)
        rows = []
        for _ in range(N):
            tok, _o, _r = env.generate_trajectory(T)
            a = tok.numpy()[0::2]; U = np.cumsum(DELTA[a], 0) % size
            rows += classify(a, [tuple(u) for u in U])
        summarise(f"torus {size} T {T}", rows)
    for L, T in ((6, 512), (8, 512), (11, 512), (16, 512), (23, 512), (32, 512), (16, 1024), (32, 128)):
        rows = []
        for _ in range(N):
            a, c = room_walk(L, T, rng); rows += classify(a, c)
        summarise(f"room {L}x{L} ({L*L//2} occ.) T {T}", rows)
