"""Independent checks of eval_rank_strata.kinds() against the env's own state."""
import numpy as np, torch, time
from mapformer.environment import GridWorld
from mapformer.eval_rank_strata import kinds
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
for T, seed in ((1024, 1234), (1024, 1241), (2048, 1234), (128, 1234)):
    np.random.seed(seed)
    n_bad_rev = n_bad_cell = 0; cnt = {"wrap": 0, "lt": 0, "ge": 0}; ambig = 0; n_rev = 0
    lagmax_lt = 0
    for _ in range(100):
        tok, _o, rev = env.generate_trajectory(T)
        vis = list(env.visited_locations)          # env's TRUE wrapped cells per step
        c = kinds(tok, env.ACTION_DELTAS, env.size)
        # independent: true wrapped cells from env; unwrapped from actions
        a = tok[0::2].numpy(); u = np.cumsum([env.ACTION_DELTAS[int(x)] for x in a], 0)
        x0 = (vis[0][0] - u[0][0]) % 64, (vis[0][1] - u[0][1]) % 64
        seen_w, last_u, last_w = set(), {}, {}
        for t in range(T):
            w = vis[t]
            if ((u[t][0] + x0[0]) % 64, (u[t][1] + x0[1]) % 64) != tuple(w):
                n_bad_cell += 1
            isrev = bool(rev[2 * t + 1])
            if isrev != (w in seen_w):
                n_bad_rev += 1
            if isrev:
                n_rev += 1
                wrapped, lag = c[t]
                if wrapped: cnt["wrap"] += 1
                elif lag < 128: cnt["lt"] += 1; lagmax_lt = max(lagmax_lt, lag)
                else: cnt["ge"] += 1
                # ambiguity: plain revisit whose most recent wrapped visit is a DIFFERENT unwrapped copy
                ku = tuple(u[t])
                if not wrapped and last_w[w] != last_u.get(ku, -1):
                    ambig += 1
                # consistency of lag with env cells
                assert lag == t - last_w[w], (t, lag, last_w[w])
                # wrap flag == unwrapped position never visited
                assert wrapped == (ku not in last_u)
            seen_w.add(w); last_u[tuple(u[t])] = t; last_w[w] = t
    tot = sum(cnt.values())
    print(f"T={T} seed={seed}: revisit-mask mismatches {n_bad_rev}, cell mismatches {n_bad_cell}, "
          f"revisits {n_rev} = strata sum {tot}; shares lt {cnt['lt']/tot:.3f} ge {cnt['ge']/tot:.3f} wrap {cnt['wrap']/tot:.3f}; "
          f"plain revisits whose last wrapped visit was a different unwrapped copy: {ambig} ({ambig/max(1,cnt['lt']+cnt['ge']):.4f}); max lag in lt {lagmax_lt}")
