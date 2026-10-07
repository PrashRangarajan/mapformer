"""CPU gate and code checks for the RANK_NOWRAP memorisation control (Amendment 1): environment_nd_redraw.GridWorldNDRedraw.
Checks (each prints PASS / FAIL):
  C1 draw_map reproduces GridWorldND.__init__'s map for fixed seeds
  C2 at the same global RNG state the redraw walk is the parent's walk: identical actions, visited cells, revisit mask;
     the observations differ; the global RNG ends in the same state (the map draw consumes nothing from it)
  C3 the map differs between trajectories, within a generate_batch call too; the env pickles (data workers) and still redraws
Gate at T=1024 on the 32-torus (rule 11), floors as rank_nowrap_gate.py (imported): revisit fraction, wrap share,
blank / majority, reversal-copy, retrace-or-blank, action n-gram (validate_nd.g2), observation n-gram -- n-grams fit on
100 redrawn walks (seed 200) and tested on 100 redrawn walks (seed 10000).
Run from /home/prashr: python3 mapformer/docs/audits/2026-10-06/rank_nowrap_redraw_gate.py
"""
import importlib.util
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer.environment_nd import GridWorldND                       # noqa: E402
from mapformer.environment_nd_redraw import GridWorldNDRedraw, draw_map  # noqa: E402
from mapformer.validate_nd import g2_action_ngram                       # noqa: E402

spec = importlib.util.spec_from_file_location("gate", "/home/prashr/mapformer/docs/audits/2026-10-06/rank_nowrap_gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
ok_all = True


def check(name, ok):
    global ok_all
    ok_all &= bool(ok); print(f"{'PASS' if ok else 'FAIL'}  {name}")


# C1
check("C1 draw_map == GridWorldND map for seeds 0..9 (2D, 32) and 3 (3D, 10)",
      all(bool((draw_map(np.random.RandomState(s), 2, 32, 16, 0.5, 16) == GridWorldND(2, 32, seed=s).obs_map).all()) for s in range(10))
      and bool((draw_map(np.random.RandomState(3), 3, 10, 16, 0.5, 16) == GridWorldND(3, 10, seed=3).obs_map).all()))

# C2
par, red = GridWorldND(2, 32, seed=7), GridWorldNDRedraw(2, 32, seed=7)
same_a = same_loc = same_rev = same_state = True; obs_diff = []
for k in range(20):
    np.random.seed(1000 + k); tp, _, rp = par.generate_trajectory(1024); lp = list(par.visited_locations); sp = np.random.get_state()[1].copy()
    np.random.seed(1000 + k); tr, _, rr = red.generate_trajectory(1024); lr = list(red.visited_locations); sr = np.random.get_state()[1].copy()
    same_a &= bool((tp[0::2] == tr[0::2]).all()); same_loc &= lp == lr; same_rev &= bool((rp == rr).all())
    same_state &= bool((sp == sr).all()); obs_diff.append(float((tp[1::2] != tr[1::2]).float().mean()))
check("C2 identical actions over 20 walks", same_a)
check("C2 identical visited cells", same_loc)
check("C2 identical revisit masks", same_rev)
check("C2 global RNG state identical after the walk (map draw consumes nothing)", same_state)
check(f"C2 observations differ (mean fraction of differing obs tokens {np.mean(obs_diff):.3f}; 0.734 = 1 - (0.25 + 0.25/16) expected for independent maps)",
      np.mean(obs_diff) > 0.4)

# C3
np.random.seed(5); maps = []
for _ in range(30):
    red.generate_trajectory(64); maps.append(red.obs_map.numpy().tobytes())
check(f"C3 30 trajectories -> {len(set(maps))} distinct maps", len(set(maps)) == 30)
np.random.seed(6); toks, _, _, _ = red.generate_batch(8, 64)
check("C3 generate_batch uses the redraw (8 sequences, distinct last maps differ from before)", red.obs_map.numpy().tobytes() not in maps)
red2 = pickle.loads(pickle.dumps(red)); np.random.seed(9); a = red2.generate_trajectory(64)[0]; m1 = red2.obs_map.clone()
np.random.seed(9); b = red.generate_trajectory(64)[0]
check("C3 pickled env redraws and is deterministic given the global RNG state (as in a data worker)",
      type(red2).__name__ == "GridWorldNDRedraw" and bool((a == b).all()) and bool((m1 == red.obs_map).all()))


# gate
def rollout_redraw(N, T, n, seed):
    env = GridWorldNDRedraw(2, N, seed=seed); np.random.seed(seed); out = []
    for _ in range(n):
        tok, _m, rev = env.generate_trajectory(T)
        out.append((tok.numpy(), rev.numpy(), [tuple(x) for x in env.visited_locations]))
    return env, out


print("\nGate, 32-torus, T=1024, map redrawn per trajectory, 100 walks (seed 10000); n-grams fit on 100 redrawn walks (seed 200)")
env, te = rollout_redraw(32, 1024, 100, 10000); _, tr = rollout_redraw(32, 1024, 100, 200)
f = g.floors_and_strata(env, te, 1024)        # floors_and_strata reads only env.unified_blank / action_deltas
rev_frac = np.mean([r[1::2].mean() for _, r, _ in te])
ng = g2_action_ngram([t for t, _, _ in tr + te], [r for _, r, _ in tr + te]); bo = max(ng, key=ng.get)
on = g.obs_ngram(tr, te)
print(f"revisit frac {rev_frac:.3f} | targets/seq {f['per_seq']:.0f} (min {f['per_seq_min']}) | wrap share {f['wrap_share']:.4f} | "
      f"blank {f['const_blank']:.3f} | majority {f['majority']:.3f} | rev-copy {f['reversal_copy']:.3f} | retrace-or-blank {f['retrace']:.3f} | "
      f"act n-gram {ng[bo]:.3f} (order {bo}) | obs n-gram 1/2/3 " + "/".join(f"{on[k]:.3f}" for k in (1, 2, 3)))
print(f"hard targets (missed by retrace-or-blank) per sequence: {(1 - f['retrace']) * f['per_seq']:.0f}")
print(f"\nALL CHECKS {'PASS' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
