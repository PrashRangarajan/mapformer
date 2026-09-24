"""Byte-exactness gate for fastgen.generate_batch_fast against the UNMODIFIED
GridWorld.generate_batch / generate_trajectory. Single thread, small.

Checks, per config: tokens, obs_mask, revisit_mask, locations, last_x/last_y, the
global numpy RNG state after the call, and a SEQUENCE of calls (so state carry-over
between batches -- the serial training stream -- is covered). Also the
data_parallel worker protocol (np.random.seed(_seed_for(base, idx)) then a batch).
"""
import sys, time, itertools
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency")
from mapformer.environment import GridWorld
from mapformer.data_parallel import _seed_for
import fastgen


def state_eq(s1, s2):
    return s1[0] == s2[0] and np.array_equal(s1[1], s2[1]) and s1[2:] == s2[2:]


def check(env, B, n, seed, calls=3, traj=False):
    np.random.seed(seed); ref = []
    for _ in range(calls):
        if traj:
            ref.append(env.generate_trajectory(n) + (list(env.visited_locations),))
        else:
            ref.append(env.generate_batch(B, n))
    ref_state = np.random.get_state(); ref_last = (env.last_x, env.last_y)
    np.random.seed(seed); got = []
    for _ in range(calls):
        if traj:
            got.append(fastgen.generate_trajectory_fast(env, n) + (list(env.visited_locations),))
        else:
            got.append(fastgen.generate_batch_fast(env, B, n))
    ok = state_eq(ref_state, np.random.get_state()) and ref_last == (env.last_x, env.last_y)
    for r, g in zip(ref, got):
        for a, b in zip(r[:3], g[:3]):
            ok &= a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        ok &= r[3] == g[3]
    return ok


fails = 0; n_checks = 0
configs = [dict(size=64, n_obs_types=16), dict(size=32, n_obs_types=256),
           dict(size=48, n_obs_types=16),             # non-power-of-2: rejection on x0/y0
           dict(size=64, n_obs_types=16, n_landmarks=200), dict(size=5, n_obs_types=4)]
for cfg, (B, n), seed in itertools.product(configs, [(1, 1), (1, 7), (3, 128), (16, 1024), (2, 2048)], [0, 1234]):
    env = GridWorld(p_empty=0.5, seed=10000, **cfg)
    for traj in ([False, True] if B == 1 else [False]):
        n_checks += 1
        if not check(env, B, n, seed, traj=traj):
            fails += 1; print("FAIL", cfg, B, n, seed, "traj" if traj else "batch")
# data_parallel worker protocol at the rank config
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=3)
for idx in range(5):
    np.random.seed(_seed_for(3 + 1, idx)); r = env.generate_batch(16, 1024)
    np.random.seed(_seed_for(3 + 1, idx)); g = fastgen.generate_batch_fast(env, 16, 1024)
    n_checks += 1
    if not (all(torch.equal(a, b) for a, b in zip(r[:3], g[:3])) and r[3] == g[3]):
        fails += 1; print("FAIL worker idx", idx)
# fallback configs must route to the original code (no change by construction)
for kw, ptn in [(dict(action_mode="rotate"), 0.0), (dict(obs_mode="ego"), 0.0),
                (dict(boundary="wall"), 0.0), ({}, 0.1)]:
    env = GridWorld(size=16, n_obs_types=16, p_empty=0.5, seed=1, **kw)
    assert not fastgen.fast_ok(env, ptn), kw
print(f"{n_checks - fails}/{n_checks} byte-exact checks passed")

# speed, rank config, single thread (machine is loaded: compare RELATIVE numbers)
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=0)
for name, fn in [("original generate_batch", lambda: env.generate_batch(16, 1024)),
                 ("fast (with locations)", lambda: fastgen.generate_batch_fast(env, 16, 1024)),
                 ("fast (no locations)", lambda: fastgen.generate_batch_fast(env, 16, 1024, want_locations=False))]:
    ts = []
    for rep in range(3 if name.startswith("orig") else 10):
        np.random.seed(rep); t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    print(f"{name:28s} B=16 n=1024: min {min(ts)*1e3:7.2f} ms  median {sorted(ts)[len(ts)//2]*1e3:7.2f} ms")
for T in (512, 1024, 2048):
    ts0, ts1 = [], []
    for rep in range(3):
        np.random.seed(rep); t = time.perf_counter(); env.generate_trajectory(T); ts0.append(time.perf_counter() - t)
        np.random.seed(rep); t = time.perf_counter(); fastgen.generate_trajectory_fast(env, T); ts1.append(time.perf_counter() - t)
    print(f"generate_trajectory T={T}: original {min(ts0)*1e3:6.2f} ms  fast {min(ts1)*1e3:6.3f} ms  ({min(ts0)/min(ts1):.0f}x)")
