"""Floors on SCORE_RANK's evaluation stream (no model). The stream is eval_noise_refine's exactly: held-out map
GridWorld(64, 16 obs types, p_empty 0.5, seed 10000), np.random.seed(1234 + s) for model seed s, 100 trajectories,
scored on revisit targets. Per seed s in 10..25 and T in {512, 1024, 2048}:
  always-blank   predict blank at every scored target (= best constant here);
  n-gram         best backoff token n-gram, orders 0..5 (the n tokens before the target), fitted on revisit targets of
                 50 T=1024 walks on each of the 16 TRAINING maps (GridWorld seed k, np seed 70000 + k, k = 10..25);
  retrace        while a run of actions reverses the previous run, step j copies the observation 2j tokens back
                 (ported from docs/audits/2026-09-27/nd_floor_wrap.py; walks are runs of 1-10 repeated actions);
  wrap-only      share of revisits whose unwrapped position is new (reachable only round the torus).
Output: score_rank_floors_out.txt next to this file."""
import collections, os, sys
import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_rank_floors_out.txt")
SEEDS = list(range(10, 26)); N_ACT = 4; BLANK = 20
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def walks(env, np_seed, n, T):
    np.random.seed(np_seed)
    return [tuple(x.numpy() for x in env.generate_trajectory(T)[::2]) for _ in range(n)]   # (tokens, revisit)


counts = [collections.defaultdict(collections.Counter) for _ in range(6)]
for k in SEEDS:
    for tok, rev in walks(GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=k), 70000 + k, 50, 1024):
        for t in np.nonzero(rev)[0]:
            for n in range(6):
                counts[n][tuple(tok[t - n:t])][tok[t]] += 1
best = [{c: cnt.most_common(1)[0][0] for c, cnt in counts[n].items()} for n in range(6)]


def floors_one(tok, rev):
    a = tok[0::2]; o = tok[1::2]; r = rev[1::2]; T = len(a)
    U = np.cumsum(DELTA[a], 0); seen_u = set()
    out = collections.Counter(); prev_run = cur = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        if r[t]:
            out["n"] += 1; out["blank"] += o[t] == BLANK
            j = cur; pr = BLANK
            if prev_run > 0 and j <= prev_run and t - 2 * j >= 0:
                pr = o[t - 2 * j]
            out["retrace"] += pr == o[t]
            out["wrap"] += tuple(U[t]) not in seen_u
            i = 2 * t + 1                                       # token index of the target
            for n in range(6):
                pred = None
                for m in range(n, -1, -1):
                    pred = best[m].get(tuple(tok[i - m:i]))
                    if pred is not None:
                        break
                out[f"ng{n}"] += pred == o[t]
        seen_u.add(tuple(U[t]))
    return out


env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
for T in (512, 1024, 2048):
    rows = []
    for s in SEEDS:
        c = collections.Counter()
        for tok, rev in walks(env, 1234 + s, 100, T):
            c.update(floors_one(tok, rev))
        ng = [c[f"ng{n}"] / c["n"] for n in range(6)]
        rows.append((s, c["blank"] / c["n"], max(ng), c["retrace"] / c["n"], c["wrap"] / c["n"], ng, c["n"]))
        log(f"T={T} s{s}: n {c['n']}  blank {rows[-1][1]:.4f}  n-gram best {rows[-1][2]:.4f} (orders 0-5 "
            + " ".join(f"{v:.3f}" for v in ng) + f")  retrace {rows[-1][3]:.4f}  wrap-only share {rows[-1][4]:.3f}")
    R = np.array([r[1:5] for r in rows])
    log(f"T={T} MEAN over seeds 10-25: blank {R[:, 0].mean():.4f}  n-gram {R[:, 1].mean():.4f}  retrace {R[:, 2].mean():.4f}  "
        f"best-of {R[:, :3].max(1).mean():.4f}  wrap-only share {R[:, 3].mean():.3f}\n")
open(OUT, "w").write("\n".join(lines) + "\n")
