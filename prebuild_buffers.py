"""Pre-build MiniWorld trajectory buffers ONE JOB AT A TIME, before training.

WHY THIS EXISTS. MiniWorld creates an EGL graphics context per worker process --
about 150 MiB of GPU memory each, even with rendering disabled. Launching 6
training jobs at once means 6 x 24 = 144 worker processes, which pinned a 24 GiB
4090 at 23,955 MiB (97.5%) before a single model had been placed on the device.
It also oversubscribed 32 cores by 4.5x, so every build ran ~5x slower than
necessary. Both problems vanish if the buffers are built serially up front and
the training jobs then just load from cache.

It calls the trainer's OWN env construction and build function rather than
recomputing the cache key, because the key includes n_workers, env fields and a
CODE_VERSION, and a prebuilder that drifts from the trainer silently builds
buffers nothing will ever load (rule 7: call the task code, do not reimplement
it).

    python3 -m mapformer.prebuild_buffers --grid-size 32 --n-obs 256 64 \
        --seeds 0 1 2 3 4 --n-workers 24 --oracle
"""
import argparse
import time

from mapformer.miniworld_env import MiniWorldWorld
from mapformer.train_miniworld import build_or_load_buffer, build_or_load_eval_buffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="MiniWorld-OneRoom-v0")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-obs", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--buffer-size", type=int, default=24000)
    ap.add_argument("--eval-trials", type=int, default=128)
    ap.add_argument("--eval-lengths", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--n-dir", type=int, default=24)
    ap.add_argument("--n-workers", type=int, default=24)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--allocentric", action="store_true")
    ap.add_argument("--fixed-map", action="store_true")
    a = ap.parse_args()

    kw = dict(env_name=a.env_name, grid_size=a.grid_size, n_dir=a.n_dir,
              allocentric=a.allocentric, fixed_map=a.fixed_map, oracle=a.oracle)
    t0 = time.time()
    n = 0
    for nobs in a.n_obs:
        for sd in a.seeds:
            env = MiniWorldWorld(seed=sd, n_obs_types=nobs, **kw)
            t = time.time()
            build_or_load_buffer(env, a.n_steps, a.buffer_size, sd,
                                 n_workers=a.n_workers)
            n += 1
            print(f"[{n:3d}] train buffer n_obs={nobs} seed={sd}  "
                  f"{time.time()-t:6.1f}s  (total {time.time()-t0:.0f}s)", flush=True)
        # eval buffers: seed 10000 for fresh-map, shared by every seed and arm at
        # this n_obs, one per eval length
        env_test = MiniWorldWorld(seed=a.seeds[0] if a.fixed_map else 10000,
                                  n_obs_types=nobs, **kw)
        for L in a.eval_lengths:
            t = time.time()
            build_or_load_eval_buffer(env_test, L, a.eval_trials,
                                      n_workers=a.n_workers)
            n += 1
            print(f"[{n:3d}] eval  buffer n_obs={nobs} T={L}      "
                  f"{time.time()-t:6.1f}s  (total {time.time()-t0:.0f}s)", flush=True)
    print(f"\n{n} buffers ready in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
