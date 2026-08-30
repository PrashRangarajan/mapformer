"""Measure the PRIOR-VISIT COUNT at each scored position, per condition.

The visits-per-cell hypothesis says the position effect switches on when a
revisit has ~one prior exemplar in context: with many prior visits attention has
plenty to match against, with one it must localise precisely. That was inferred
from arithmetic (T / n_occupied = 16 / 4 / 1 for grid 8 / 16 / 32 at T=512).
Arithmetic is not measurement -- the walk is a directed random walk, not uniform
coverage, so the realised distribution can differ a lot from T/n_occupied.

CPU only, no training. Calls the env's own generate_trajectory and revisit mask
rather than reimplementing the walk (rule 7: a gate that reimplements the task
certifies a different task from the one the trainer runs).
"""
import argparse
from collections import Counter

import numpy as np
import pyglet
pyglet.options["headless"] = True
from mapformer.miniworld_env import MiniWorldWorld


def run(grid, T, n_obs, episodes, seed, oracle=True):
    w = MiniWorldWorld(env_name="MiniWorld-OneRoom-v0", grid_size=grid,
                       n_obs_types=n_obs, seed=seed, oracle=oracle)
    rng = np.random.RandomState(seed)
    priors, cells_seen, scored = [], [], 0
    for _ in range(episodes):
        tok, _om, rev = w.generate_trajectory(T, rng=rng)
        rev = rev.numpy()
        cells = w.visited_locations
        seen = Counter()
        for i in range(T):
            c = tuple(cells[i])
            if rev[2 * i + 1]:                 # scored cross-cell revisit
                priors.append(seen[c])         # prior visits BEFORE this one
                scored += 1
            seen[c] += 1
        cells_seen.append(len(seen))
    p = np.array(priors)
    return dict(grid=grid, T=T, n_obs=n_obs, scored=scored,
                distinct_cells=float(np.mean(cells_seen)),
                mean_prior=float(p.mean()), median_prior=float(np.median(p)),
                frac_exactly_1=float((p == 1).mean()),
                frac_le_2=float((p <= 2).mean()),
                frac_ge_5=float((p >= 5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    # (grid, T, n_obs) -- the three measured conditions plus the two proposed
    conds = [(8, 512, 16), (16, 512, 64), (32, 512, 256)]          # measured
    for g in (8, 16, 24, 32):
        for T in (128, 256, 512, 1024, 2048):
            if (g, T) in ((8,512),(16,512),(32,512)): continue
            conds.append((g, T, 64))
    rows = []
    print(f"{'grid':>5} {'T':>5} {'distinct':>9} {'T/cells':>8} {'mean prior':>11} "
          f"{'median':>7} {'=1':>6} {'<=2':>6} {'>=5':>6}")
    for g, T, n in conds:
        r = run(g, T, n, a.episodes, a.seed)
        rows.append(r)
        print(f"{g:>5} {T:>5} {r['distinct_cells']:>9.0f} "
              f"{T/max(r['distinct_cells'],1):>8.2f} {r['mean_prior']:>11.2f} "
              f"{r['median_prior']:>7.0f} {r['frac_exactly_1']:>6.2f} "
              f"{r['frac_le_2']:>6.2f} {r['frac_ge_5']:>6.2f}")
    import json
    json.dump(rows, open("VISITS_PER_CELL.json", "w"), indent=2)
    print("\nwrote VISITS_PER_CELL.json")


if __name__ == "__main__":
    main()
