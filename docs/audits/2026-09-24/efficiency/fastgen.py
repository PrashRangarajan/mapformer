"""Bit-exact vectorised trajectory generation for GridWorld's DEFAULT configuration.

Proposed for mapformer/environment.py (see environment_fastpath.patch). Stand-alone
here so it can be verified against the unmodified environment without touching it.

WHAT IT REPRODUCES, BYTE FOR BYTE (verified by verify_fastgen.py):
  tokens, obs_mask, revisit_mask, visited_locations, last_x/last_y, AND the global
  numpy RNG state after the call -- so every later draw (the next trajectory, the
  next batch, the serial training stream, the --data-workers stream) is unchanged.

HOW. GridWorld.generate_trajectory draws, from the global legacy RandomState, in
this order: x0 = randint(0,size), y0 = randint(0,size), then per segment
a = randint(0,n_actions), k = randint(1,11), until sum(k) >= n_steps. Legacy
RandomState.randint (frozen by NumPy's stream-compatibility policy, NEP 19) draws
each bounded integer by masked rejection on 32-bit MT19937 words:
    val = next_uint32() & mask  repeated while val > (high-1-low)
So the whole trajectory is a deterministic parse of the raw MT19937 word stream.
We read a block of raw words, parse it (a short Python loop per SEGMENT, ~n_steps/5.5
iterations, instead of the original per-STEP loop with torch .item() calls), then
restore the RNG state and advance it by exactly the number of words consumed.
Everything per-step (walk, obs lookup, revisit flags) is numpy-vectorised.

SCOPE. Only the default configuration: action_mode='translate', obs_mode='allo',
boundary='torus', action_record='commanded', no continuous headings, and
p_transition_noise == 0. Anything else falls back to the original loop, so
behaviour elsewhere is unchanged by construction.
"""
import numpy as np
import torch

_DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # == ACTION_DELTAS


def _mask_for(rng: int) -> int:
    """numpy's gen_mask: smallest all-ones bit mask >= rng."""
    m = rng
    for s in (1, 2, 4, 8, 16):
        m |= m >> s
    return m


def fast_ok(env, p_transition_noise=0.0) -> bool:
    from mapformer.environment import GridWorld
    # a subclass that overrides the walk (environment_topology.py does) keeps its own
    if type(env).generate_trajectory is not GridWorld.generate_trajectory:
        return False
    return (env.action_mode == "translate" and env.obs_mode == "allo"
            and env.boundary == "torus" and env.action_record == "commanded"
            and not env.continuous and p_transition_noise == 0.0
            and env.size >= 2 and env.n_actions == 4)


def _next_accept(w: np.ndarray, rng: int) -> list:
    """nxt[i] = smallest j >= i whose masked word is accepted (<= rng); len(w) if none."""
    n = len(w)
    acc = (w & _mask_for(rng)) <= rng
    idx = np.where(acc, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1].tolist()


def _parse(env, batch_size, n_steps, starts, words_hint):
    """Parse the raw word stream into (x0, y0, acts, ks) per trajectory.

    Returns (per_traj, consumed) or None if the block was too short."""
    bg = np.random.mtrand._rand._bit_generator
    w = bg.random_raw(words_hint).astype(np.int64)
    n = len(w)
    wl = w.tolist()
    rs, ra, rk = env.size - 1, env.n_actions - 1, 9          # randint(0,size), (0,na), (1,11)
    ms, ma, mk = _mask_for(rs), _mask_for(ra), _mask_for(rk)
    ns = _next_accept(w, rs) if ms != rs else None            # None -> never rejects
    na_ = _next_accept(w, ra) if ma != ra else None
    nk = _next_accept(w, rk)
    out = []
    p = 0
    for b in range(batch_size):
        if starts is None:
            q = ns[p] if ns is not None else p
            if q >= n:
                return None
            x0 = wl[q] & ms; p = q + 1
            q = ns[p] if ns is not None else p
            if q >= n:
                return None
            y0 = wl[q] & ms; p = q + 1
        else:
            x0, y0 = starts
        acts, ks, t = [], [], 0
        while t < n_steps:
            q = na_[p] if na_ is not None else p
            if q >= n:
                return None
            a = wl[q] & ma; p = q + 1
            if p >= n:
                return None
            q = nk[p]
            if q >= n:
                return None
            k = 1 + (wl[q] & mk); p = q + 1
            acts.append(a); ks.append(k); t += k
        ks[-1] -= t - n_steps
        out.append((x0, y0, acts, ks))
    return out, p


def _parse_with_retry(env, batch_size, n_steps, starts):
    state = np.random.get_state()
    hint = batch_size * (n_steps + 64) + 16          # ~2x the expected consumption
    while True:
        r = _parse(env, batch_size, n_steps, starts, hint)
        np.random.set_state(state)                   # rewind ...
        if r is not None:
            per_traj, consumed = r
            if consumed:
                np.random.mtrand._rand._bit_generator.random_raw(consumed)  # ... and advance exactly
            return per_traj
        hint *= 2


def generate_batch_fast(env, batch_size, n_steps=128, p_transition_noise=0.0,
                        start=None, want_locations=True):
    """Drop-in for env.generate_batch (and, with batch_size=1, generate_trajectory)."""
    if not fast_ok(env, p_transition_noise):
        return env.generate_batch(batch_size, n_steps, p_transition_noise=p_transition_noise)
    size = env.size
    per_traj = _parse_with_retry(env, batch_size, n_steps, start)
    x0 = np.array([t[0] for t in per_traj], dtype=np.int64)
    y0 = np.array([t[1] for t in per_traj], dtype=np.int64)
    acts = np.fromiter((a for t in per_traj for a in t[2]), dtype=np.int64)
    ks = np.fromiter((k for t in per_traj for k in t[3]), dtype=np.int64)
    a_step = np.repeat(acts, ks).reshape(batch_size, n_steps)          # commanded == executed
    d = _DELTAS[a_step]                                                # (B, n, 2)
    xs = (x0[:, None] + np.cumsum(d[..., 0], axis=1)) % size          # Python and numpy % agree
    ys = (y0[:, None] + np.cumsum(d[..., 1], axis=1)) % size          # (floor-mod) on ints
    obs_np = env.obs_map.numpy()                                       # zero-copy view
    obs = obs_np[xs, ys]
    tok = np.empty((batch_size, 2 * n_steps), dtype=np.int64)
    tok[:, 0::2] = a_step + env.action_offset
    tok[:, 1::2] = obs + env.obs_offset
    # revisit = the post-step cell appeared at an EARLIER step of the same trajectory
    # (the start cell is never added to `seen`; moved is always True on a torus walk)
    cell = (xs * size + ys) + (np.arange(batch_size, dtype=np.int64) * size * size)[:, None]
    _, first = np.unique(cell.ravel(), return_index=True)
    is_first = np.zeros(batch_size * n_steps, dtype=bool)
    is_first[first] = True
    rev = np.zeros((batch_size, 2 * n_steps), dtype=bool)
    rev[:, 1::2] = ~is_first.reshape(batch_size, n_steps)
    obs_mask = np.zeros((batch_size, 2 * n_steps), dtype=bool)
    obs_mask[:, 1::2] = True
    locs = None
    if want_locations:
        xl, yl = xs.tolist(), ys.tolist()
        locs = [list(zip(xl[b], yl[b])) for b in range(batch_size)]
        env.visited_locations = list(locs[-1])
    env.last_x, env.last_y = int(xs[-1, -1]), int(ys[-1, -1])
    return (torch.from_numpy(tok), torch.from_numpy(obs_mask), torch.from_numpy(rev), locs)


def generate_trajectory_fast(env, n_steps=128, start=None, p_transition_noise=0.0):
    if not fast_ok(env, p_transition_noise):
        return env.generate_trajectory(n_steps, start=start, p_transition_noise=p_transition_noise)
    tok, om, rev, _ = generate_batch_fast(env, 1, n_steps, start=start)
    return tok[0], om[0], rev[0]
