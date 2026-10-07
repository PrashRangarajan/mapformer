"""CPU gate for RANK_NOWRAP_PREREG.md (rule 11): the RANK_ND 2D walk (environment_nd.GridWorldND, called, not
reimplemented) at T=1024 on tori of growing size. Per grid N, over n trajectories (env seed = map seed, walk seed
`seed`):
  revisit   fraction of observation slots that are scored revisits; scored targets per sequence
  wrap      share of revisits whose cell was seen before only at a different UNWRAPPED position (exactly the
            definition of analyze_rank_nd_secondary.strat_acc and docs/audits/2026-09-27/nd_floor_wrap.py)
  lag       share of plain revisits with gap < 128 steps (eval_rank_strata's short / long split)
  floors    best constant (always-blank and the majority class), action-stream n-gram orders 1-5 (validate_nd.g2,
            imported), reversal-copy and retrace (nd_floor_wrap's predictors, copied verbatim below as functions),
            and an OBSERVATION n-gram (previous k observations -> next observation, orders 1-3; a non-path predictor)
  extent    max unwrapped displacement per axis over the trajectory (how far the walk spreads relative to N)
Run: python3 -m mapformer.docs.audits.2026-10-06... is not importable (dash); run as a file from /home/prashr:
  cd /home/prashr && python3 mapformer/docs/audits/2026-10-06/rank_nowrap_gate.py --grids 32 64 96 128 192 256
"""
import argparse
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer.environment_nd import GridWorldND          # noqa: E402
from mapformer.validate_nd import g2_action_ngram         # noqa: E402


def opp(a):
    return a ^ 1


def rollout(N, T, n, map_seed, walk_seed):
    env = GridWorldND(2, N, seed=map_seed); np.random.seed(walk_seed)
    out = []
    for _ in range(n):
        tok, _m, rev = env.generate_trajectory(T)
        out.append((tok.numpy(), rev.numpy(), [tuple(x) for x in env.visited_locations]))
    return env, out


def floors_and_strata(env, trajs, T):
    """nd_floor_wrap.run's loop, verbatim logic, plus lag split and per-trajectory counts."""
    blank = env.unified_blank
    tot = c_const = c_rev = c_ret = 0
    wrap_only = 0; short_plain = 0; per_seq = []; ext = []; labels = Counter()
    for tok, rev, P in trajs:
        a = tok[0::2]; o = tok[1::2]; r = rev[1::2]
        U = np.cumsum(env.action_deltas[a], axis=0)
        ext.append(np.abs(U).max(0).max())
        last_cell = {}; last_unw = {}
        prev_run_len = 0; cur_len = 0; n_here = 0
        for t in range(T):
            if t > 0 and a[t] == a[t - 1]:
                cur_len += 1
            else:
                if t > 0 and a[t] == opp(a[t - 1]):
                    prev_run_len = cur_len
                else:
                    prev_run_len = 0
                cur_len = 1
            j = cur_len
            if r[t]:
                tot += 1; n_here += 1; labels[int(o[t])] += 1
                c_const += o[t] == blank
                pred = o[t - 2] if (t >= 2 and a[t] == opp(a[t - 1])) else blank
                c_rev += pred == o[t]
                pr = blank
                if prev_run_len > 0 and j <= prev_run_len and t - 2 * j >= 0 and (a[t - j] == opp(a[t]) if t - j >= 0 else False):
                    pr = o[t - 2 * j]
                c_ret += pr == o[t]
                uk = tuple(U[t])
                if uk not in last_unw:
                    wrap_only += 1
                elif t - last_cell[P[t]] < 128:
                    short_plain += 1
            last_cell[P[t]] = t; last_unw[tuple(U[t])] = t
        per_seq.append(n_here)
    plain = tot - wrap_only
    return {"scored": tot, "per_seq": float(np.mean(per_seq)), "per_seq_min": int(np.min(per_seq)),
            "const_blank": c_const / tot, "majority": max(labels.values()) / tot,
            "reversal_copy": c_rev / tot, "retrace": c_ret / tot,
            "wrap_share": wrap_only / tot, "wrap_per_seq": wrap_only / len(trajs),
            "plain_short_share": short_plain / max(plain, 1), "extent_median": float(np.median(ext))}


def obs_ngram(trajs_tr, trajs_te, max_order=3):
    """Previous k OBSERVATION tokens (with the current action) -> next observation, fit on train walks."""
    out = {}
    for k in range(1, max_order + 1):
        tab = defaultdict(Counter)
        for tok, rev, _ in trajs_tr:
            a = tok[0::2]; o = tok[1::2]; r = rev[1::2]
            for t in range(k, len(o)):
                if r[t]:
                    tab[(int(a[t]),) + tuple(o[t - k:t].tolist())][int(o[t])] += 1
        hit = tot = 0
        for tok, rev, _ in trajs_te:
            a = tok[0::2]; o = tok[1::2]; r = rev[1::2]
            for t in range(k, len(o)):
                if r[t]:
                    tot += 1; c = tab.get((int(a[t]),) + tuple(o[t - k:t].tolist()))
                    hit += bool(c) and c.most_common(1)[0][0] == int(o[t])
        out[k] = hit / max(tot, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", type=int, default=[32, 64, 96, 128, 192, 256])
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    print(f"T={a.T}, {a.n} trajectories per grid, held-out map seed 10000 (walk seed 10000); n-gram floors fit on "
          f"{a.n} walks of a training map (seed 200, walk seed 200) and tested on the held-out walks\n")
    hdr = ("grid cells | revisit frac | targets/seq (min) | wrap share | wrap/seq | plain short<128 | median extent | "
           "blank | majority | rev-copy | retrace | act n-gram best (order) | obs n-gram 1/2/3")
    print(hdr); print("-" * len(hdr))
    for N in a.grids:
        env, te = rollout(N, a.T, a.n, 10000, 10000)
        _, tr = rollout(N, a.T, a.n, 200, 200)
        f = floors_and_strata(env, te, a.T)
        rev_frac = np.mean([r[1::2].mean() for _, r, _ in te])
        ng = g2_action_ngram([t for t, _, _ in tr + te], [r for _, r, _ in tr + te])
        bo = max(ng, key=ng.get)
        on = obs_ngram(tr, te)
        print(f"{N:4d} {N * N:6d} | {rev_frac:.3f} | {f['per_seq']:.0f} ({f['per_seq_min']}) | {f['wrap_share']:.4f} | "
              f"{f['wrap_per_seq']:.1f} | {f['plain_short_share']:.3f} | {f['extent_median']:.0f} | {f['const_blank']:.3f} | "
              f"{f['majority']:.3f} | {f['reversal_copy']:.3f} | {f['retrace']:.3f} | {ng[bo]:.3f} ({bo}) | "
              + "/".join(f"{on[k]:.3f}" for k in (1, 2, 3)), flush=True)


if __name__ == "__main__":
    main()
