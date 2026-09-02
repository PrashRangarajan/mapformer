"""Gate for ParallelBatchGenerator. Run this before trusting any parallel run.

Four checks, in the order that matters:

  1. WORKER FIDELITY. A worker's pickled environment must produce a
     BYTE-IDENTICAL batch to the parent's env given the same numpy seed. This is
     the check that would catch a rebuilt-environment bug (different obs_map,
     missing landmark state) -- the class of error that silently certifies a
     different task from the one the trainer runs.
  2. WORKER-COUNT INVARIANCE. Batch i is seeded by its INDEX, so the stream must
     be byte-identical across n_workers=1,3,6 and across repeat runs. If this
     fails, results are not reproducible and the path is unusable.
  3. DISTRIBUTIONAL EQUIVALENCE vs the serial path. The parallel stream is a
     DIFFERENT SAMPLE from the same generator, so it cannot match byte for byte.
     What must match is the distribution: revisit rate, token marginals, and the
     run-length structure of the directed walk.
  4. SPEEDUP, measured, at the standard config.
"""
import argparse, time
import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.data_parallel import ParallelBatchGenerator, _seed_for


def h(a):
    return hash(np.asarray(a).tobytes())


def stats(tok, rev, env):
    """Summary statistics that a broken walk would move."""
    t = np.asarray(tok)
    acts = t[:, 0::2].ravel()
    obs = t[:, 1::2].ravel()
    # run length of repeated identical actions, the directed-walk signature
    runs, cur = [], 1
    row = t[0, 0::2]
    for i in range(1, len(row)):
        if row[i] == row[i - 1]:
            cur += 1
        else:
            runs.append(cur); cur = 1
    return dict(revisit_rate=float(np.asarray(rev).mean()),
                action_entropy=float(-sum(p * np.log(p) for p in
                                          np.bincount(acts - env.action_offset,
                                                      minlength=env.n_actions)
                                          / len(acts) if p > 0)),
                blank_frac=float((obs == env.blank_token).mean()),
                obs_nunique=int(len(np.unique(obs))),
                mean_run=float(np.mean(runs)) if runs else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--env-seed", type=int, default=42)
    a = ap.parse_args()
    B, T = a.batch_size, a.n_steps
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=a.env_seed)
    ok = True

    # -- 1. worker fidelity ---------------------------------------------------
    np.random.seed(_seed_for(0, 0))
    ref = env.generate_batch(B, T)[0].numpy()
    with ParallelBatchGenerator(env, B, T, n_workers=2, base_seed=0) as g:
        got = g.next_batch()[0].numpy()
    same = np.array_equal(ref, got)
    ok &= same
    print(f"1. worker fidelity (pickled env == parent env, same seed): "
          f"{'PASS' if same else 'FAIL'}")

    # -- 2. worker-count invariance ------------------------------------------
    sigs = {}
    for nw in a.workers:
        with ParallelBatchGenerator(env, B, T, n_workers=nw, base_seed=7) as g:
            sigs[nw] = [h(g.next_batch()[0]) for _ in range(a.n_batches)]
    with ParallelBatchGenerator(env, B, T, n_workers=a.workers[-1], base_seed=7) as g:
        sigs["repeat"] = [h(g.next_batch()[0]) for _ in range(a.n_batches)]
    inv = all(v == sigs[a.workers[0]] for v in sigs.values())
    ok &= inv
    print(f"2. worker-count invariance over {a.workers} + a repeat run, "
          f"{a.n_batches} batches: {'PASS' if inv else 'FAIL'}")

    # -- 3. distributional equivalence vs serial ------------------------------
    np.random.seed(1234)
    S = [env.generate_batch(B, T) for _ in range(a.n_batches)]
    ser = {k: np.mean([stats(t, r, env)[k] for t, _o, r, _l in S])
           for k in ("revisit_rate", "action_entropy", "blank_frac",
                     "obs_nunique", "mean_run")}
    with ParallelBatchGenerator(env, B, T, n_workers=a.workers[-1], base_seed=99) as g:
        P = [g.next_batch() for _ in range(a.n_batches)]
    par = {k: np.mean([stats(t, r, env)[k] for t, _o, r, _l in P]) for k in ser}
    # tolerance from the serial path's own batch-to-batch spread
    print(f"3. distributional equivalence vs serial ({a.n_batches} batches each):")
    for k in ser:
        sd = np.std([stats(t, r, env)[k] for t, _o, r, _l in S], ddof=1)
        tol = max(3 * sd / np.sqrt(a.n_batches), 1e-9)
        d = abs(par[k] - ser[k]); good = d <= tol or sd == 0
        ok &= bool(good)
        print(f"     {k:16s} serial {ser[k]:9.4f}  parallel {par[k]:9.4f}  "
              f"|d| {d:.4f} vs tol {tol:.4f}  {'ok' if good else 'FAIL'}")

    # -- 4. speedup -----------------------------------------------------------
    print("4. throughput at the standard config:")
    t0 = time.time()
    for _ in range(a.n_batches):
        env.generate_batch(B, T)
    t_ser = time.time() - t0
    print(f"     serial{'':12s} {t_ser / a.n_batches * 1000:7.1f} ms/batch"
          f"   -> {t_ser / a.n_batches * 98:6.1f} s/epoch")
    for nw in a.workers:
        with ParallelBatchGenerator(env, B, T, n_workers=nw, base_seed=5) as g:
            g.next_batch()                                    # warm the pool
            t0 = time.time()
            for _ in range(a.n_batches):
                g.next_batch()
            t = time.time() - t0
        print(f"     parallel x{nw:<2d}{'':8s} {t / a.n_batches * 1000:7.1f} ms/batch"
              f"   -> {t / a.n_batches * 98:6.1f} s/epoch   ({t_ser / t:.2f}x)")

    print(f"\n{'ALL CHECKS PASS' if ok else 'FAILURES ABOVE -- do not use'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
