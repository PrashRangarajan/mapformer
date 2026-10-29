"""POST HOC, CPU only (Opus scope expansion 2026-10-29). Revisit strata by COVERING SPACE: is the revisit a revisit on the
Cayley tree of the free group (moves cancel only against their immediate inverse, as a stack: reversal / nested
reversal, Dyck-like), on the plane Z^2 but not on the tree (needs commutation: a metric map), or on the torus only (wrap)?

tree node at step t = the freely reduced move word of the prefix (stack: push a, or pop when the top is a's inverse);
plane node = unwrapped displacement; torus node = cell. A revisit is classed by the most 'free' cover on which the current
node was visited before: TREE ⊂ PLANE ⊂ TORUS. Also: tree-revisit age (moves since the last visit of the tree node) and
stack depth at the revisit.

Joins q1_revisit_taxonomy.py's rows (same walks: env seed 10000, np.random.seed(1234 + s), same N_WALKS) and prints model
accuracy per cover stratum and, within the plane-only stratum, by gap.
  CUDA_VISIBLE_DEVICES= python3 docs/research/scripts_opus/q1_cover_strata.py [n_walks]
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld  # noqa: E402

OUT = "/home/prashr/mapformer/docs/research/scripts_opus/out"
N_WALKS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
T = 1024
INV = {0: 1, 1: 0, 2: 3, 3: 2}
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
SEEDS = (500, 501, 502, 503, 2800, 2801, 2802, 2803)


def cover_labels(a, size=64):
    stack = []; seen_tree = {(): -1}; seen_plane = {(0, 0): -1}; seen_torus = {(0, 0): -1}
    # NOTE the env never counts the start cell as visited; we mirror that by tracking but only classing env revisits
    p = np.zeros(2, int); out = {}
    first_cell = None
    for t, x in enumerate(a):
        if stack and stack[-1] == INV[x]:
            stack.pop()
        else:
            stack.append(x)
        p = p + DELTA[x]
        node, pl, to = tuple(stack), (int(p[0]), int(p[1])), (int(p[0] % size), int(p[1] % size))
        out[t] = dict(tree=node in seen_tree, plane=pl in seen_plane, torus=to in seen_torus,
                      tree_age=t - seen_tree.get(node, t), depth=len(stack))
        seen_tree[node] = t; seen_plane[pl] = t; seen_torus[to] = t
    return out


def main():
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    lab = {}
    for s in SEEDS:
        np.random.seed(1234 + s)
        for w in range(N_WALKS):
            tok, _om, rev = env.generate_trajectory(T)
            a = tok.numpy()[0::2]
            # the env's start position is random; displacement-based covers are start-invariant
            L = cover_labels(a)
            for t, d in L.items():
                lab[(s, w, t)] = d
    rows = json.load(open(f"{OUT}/q1_revisit_rows.json"))
    lines = []
    P = lambda *x: (print(*x), lines.append(" ".join(str(y) for y in x)))
    arms = ["RoPE_L1", "RoPE_L4", "RoPE_Loop", "Path_L1"]
    n_bad = 0
    for r in rows:
        d = lab[(r["seed"], r["walk"], r["t"])]
        if not d["torus"]:
            n_bad += 1          # env revisit that is the start cell's first counted visit etc.
        r.update(d)
        r["cover"] = "tree" if d["tree"] else ("plane" if d["plane"] else "torus")
        r["stratum"] = "wrap" if r["wrap"] else ("short" if r["lag"] < 128 else "long")
        r["rt_ok"] = int(r["rt_app"] and r["rt_pred_o"] == r["tgt"])
        r["blank"] = int(r["tgt"] == 20)
    P(f"rows {len(rows)}; env revisits not torus-revisits under the start-cell convention: {n_bad}")
    P("== cover stratum shares and model accuracy (mean over revisits; 8 seeds) ==")
    for c in ("tree", "plane", "torus"):
        S = [r for r in rows if r["cover"] == c]
        P(f"{c:6s} n {len(S):6d} share {len(S)/len(rows):.3f} blank {np.mean([r['blank'] for r in S]):.3f} retrace-applies "
          f"{np.mean([r['rt_app'] for r in S]):.3f} | " + " ".join(f"{k} {np.mean([r[k] for r in S]):.3f}" for k in arms))
    P("== tree revisits: retrace vs nested (non-retrace tree), by tree age (moves) ==")
    for rt in (True, False):
        for lo, hi in ((0, 16), (16, 64), (64, 256), (256, 2000)):
            S = [r for r in rows if r["cover"] == "tree" and r["rt_app"] == rt and lo <= r["tree_age"] < hi]
            if len(S) < 30:
                continue
            P(f"tree {'retrace' if rt else 'nested '} age {lo:4d}-{hi:4d}: n {len(S):5d} blank {np.mean([r['blank'] for r in S]):.3f} | "
              + " ".join(f"{k} {np.mean([r[k] for r in S]):.3f}" for k in arms))
    P("== plane-only (needs commutation, no wrap) revisits by gap (moves since last visit of the cell) ==")
    for lo, hi in ((0, 16), (16, 64), (64, 128), (128, 512), (512, 2000)):
        S = [r for r in rows if r["cover"] == "plane" and lo <= r["lag"] < hi]
        if len(S) < 30:
            continue
        P(f"plane gap {lo:4d}-{hi:4d}: n {len(S):5d} blank {np.mean([r['blank'] for r in S]):.3f} | "
          + " ".join(f"{k} {np.mean([r[k] for r in S]):.3f}" for k in arms))
    P("== per seed: tree-nested (non-retrace) vs plane-only short-gap (< 128) accuracy, index arms ==")
    for k in arms:
        dd = []
        for s in SEEDS:
            A = [r[k] for r in rows if r["seed"] == s and r["cover"] == "tree" and not r["rt_app"]]
            B = [r[k] for r in rows if r["seed"] == s and r["cover"] == "plane" and r["lag"] < 128]
            Ab = [r["blank"] for r in rows if r["seed"] == s and r["cover"] == "tree" and not r["rt_app"]]
            Bb = [r["blank"] for r in rows if r["seed"] == s and r["cover"] == "plane" and r["lag"] < 128]
            dd.append((np.mean(A) - np.mean(Ab)) - (np.mean(B) - np.mean(Bb)))
        dd = np.array(dd)
        P(f"{k}: (tree-nested above its blank) - (plane short above its blank) = {dd.mean():+.3f}; seeds + {int((dd > 0).sum())}/8")
    open(f"{OUT}/q1_cover_strata_out.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
