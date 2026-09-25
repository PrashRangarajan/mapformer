"""Parallel trajectory generation for the torus task.

WHY. Measured split of one training epoch at the standard config (d_model=128,
n_heads=2, n_layers=1, n_steps=128, batch_size=128, n_batches=98):

    trajectory generation   9.1 s      79-95% of the epoch
    model fwd+bwd           0.5 s      Vanilla
                            2.5 s      Level15Looped

Generation is single-threaded Python -- 83% of its cost is the interpreted loop
body in GridWorld.generate_trajectory, so there is no micro-optimisation worth
having (converting obs_map off torch saves 5%). Each training process therefore
pegs exactly one core; on a 32-core box running 8 jobs, 24 cores sat idle.

WHAT THIS IS NOT. It is not a cached buffer. Trajectories are still generated
fresh for every batch of every epoch, so the data distribution is unchanged.
prebuild_buffers.py does the cached thing for MiniWorld and that IS a
methodological change (cached and live differed by ~2pp there); this is not.

DETERMINISM, which is the whole design constraint. GridWorld.generate_trajectory
draws from the GLOBAL numpy RNG one step at a time, so a parallel scheme cannot
reproduce the serial draw order. Instead of pretending otherwise, batch i is
seeded by its own INDEX:

    seed(i) = (base_seed * 1_000_003 + i) % 2**31

and generated in its entirety by whichever worker picks it up. The content of
batch i therefore depends on i alone -- not on the worker count, not on
scheduling, not on arrival order. Consequences worth stating plainly:

  * same base_seed and any n_workers -> BYTE-IDENTICAL token stream. Verified.
  * the stream DIFFERS from the serial path's. It is the same distribution drawn
    from the same generator, not the same sample. So a parallel run will not
    reproduce a stored serial checkpoint, and must not be compared to one
    (rule 3: retrain every arm in the same batch). This is why it is opt-in.

Workers call env.generate_batch rather than reimplementing the walk (rule 7: a
gate must CALL the task code -- a duplicated walk in validate_family_tree.py once
certified a different task from the one the trainer ran).

The context is 'spawn', not 'fork': the trainer has CUDA initialised by the time
this starts, and forking a CUDA process is unsafe.
"""
import multiprocessing as mp
import os
import numpy as np
import torch


def _seed_for(base_seed: int, batch_idx: int) -> int:
    return int((base_seed * 1_000_003 + batch_idx) % (2 ** 31))


def _worker(env, batch_size, n_steps, p_transition_noise, want_locations,
            base_seed, task_q, result_q):
    """Serve (batch_idx -> batch) requests forever from one pickled environment.

    The ENVIRONMENT OBJECT ITSELF is shipped to the worker rather than rebuilt
    from constructor arguments. GridWorld does not store its `seed`, so a rebuild
    could not reproduce obs_map / is_landmark_cell / landmark_cells -- and an
    enumerated attribute list would silently rot the next time the constructor
    grows a knob. Pickling the object is identity by construction: obs_map is a
    64x64 int64 tensor, 32 KB, sent once per worker at startup.
    """
    torch.set_num_threads(1)                     # one core per worker, by design
    env.visited_locations = []
    while True:
        idx = task_q.get()
        if idx is None:
            break
        np.random.seed(_seed_for(base_seed, idx))
        tok, obs, rev, locs = env.generate_batch(
            batch_size, n_steps, p_transition_noise=p_transition_noise)
        result_q.put((idx, tok.numpy(), obs.numpy(), rev.numpy(),
                      np.asarray(locs, dtype=np.int16) if want_locations else None))


class ParallelBatchGenerator:
    """Yields batches identical in form to env.generate_batch, in index order.

    Usage mirrors the serial path exactly:

        gen = ParallelBatchGenerator(env, batch_size, n_steps, n_workers=6,
                                     base_seed=seed, want_locations=False)
        for _ in range(n_batches):
            tokens, obs_mask, revisit_mask, locations = gen.next_batch()
        ...
        gen.close()

    `locations` is None unless want_locations=True. Pickling the location lists
    costs more than generating them, and only the DoG-style aux variants read
    them, so they are off unless asked for.
    """

    def __init__(self, env, batch_size, n_steps, n_workers=6, base_seed=0,
                 p_transition_noise=0.0, want_locations=False, prefetch=None):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1; use the serial path for 0")
        self.batch_size, self.n_steps = batch_size, n_steps
        self.want_locations = want_locations
        ctx = mp.get_context("spawn")
        self.task_q, self.result_q = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=_worker,
                                  args=(env, batch_size, n_steps, p_transition_noise,
                                        want_locations, int(base_seed),
                                        self.task_q, self.result_q),
                                  daemon=True)
                      for _ in range(n_workers)]
        for p in self.procs:
            p.start()
        self._next_to_submit = 0
        self._next_to_yield = 0
        self._pending = {}
        # keep every worker fed plus one spare batch each
        for _ in range(n_workers * (prefetch or 2)):
            self._submit()

    def _submit(self):
        self.task_q.put(self._next_to_submit)
        self._next_to_submit += 1

    def next_batch(self):
        while self._next_to_yield not in self._pending:
            idx, tok, obs, rev, locs = self.result_q.get()
            self._pending[idx] = (tok, obs, rev, locs)
        tok, obs, rev, locs = self._pending.pop(self._next_to_yield)
        self._next_to_yield += 1
        self._submit()
        return (torch.from_numpy(tok), torch.from_numpy(obs), torch.from_numpy(rev),
                [[tuple(int(v) for v in xy) for xy in traj] for traj in locs]
                if locs is not None else None)

    def close(self):
        for _ in self.procs:
            try:
                self.task_q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
