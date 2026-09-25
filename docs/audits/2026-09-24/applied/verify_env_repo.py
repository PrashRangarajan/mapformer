"""Patched repo environment.py vs the pristine HEAD copy (base worktree), byte for byte,
through the public API. Adapted from docs/audits/2026-09-24/efficiency/verify_env_patch.py
and verify_fastgen.py (which compared against the audit's scratch copy)."""
import sys, itertools, importlib.util, time
import numpy as np, torch
torch.set_num_threads(1)
SP = __import__("os").environ["MF_SCRATCH"]  # holds base/mapformer: a git worktree of the pre-change commit
spec = importlib.util.spec_from_file_location("oldenv", f"{SP}/base/mapformer/environment.py")
OE = importlib.util.module_from_spec(spec); spec.loader.exec_module(OE)
sys.path.insert(0, "/home/prashr")
import mapformer.environment as NE
from mapformer.data_parallel import _seed_for
Old, New = OE.GridWorld, NE.GridWorld
assert not hasattr(Old, "_fast_walk_ok") and hasattr(New, "_fast_walk_ok")

def run(env, seq, seed):
    np.random.seed(seed); out = []
    for kind, a, b in seq:
        if kind == "b":
            r = env.generate_batch(a, b); out.append((r[0], r[1], r[2], r[3]))
        else:
            r = env.generate_trajectory(a); out.append((r[0], r[1], r[2], list(env.visited_locations)))
    st = np.random.get_state()
    return out, (st[0], st[1].tobytes(), st[2:]), (env.last_x, env.last_y)

def same_run(o, m):
    ok = o[1] == m[1] and o[2] == m[2] and len(o[0]) == len(m[0])
    for x, y in zip(o[0], m[0]):
        ok &= all(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) for a, b in zip(x[:3], y[:3]))
        ok &= x[3] == y[3]
    return ok

ok = n = 0
seqs = [[("b", 16, 1024), ("t", 2048, None), ("b", 3, 7), ("t", 1, None), ("b", 1, 128), ("t", 512, None)],
        [("b", 128, 128), ("b", 128, 128)], [("t", 1, None)] * 3, [("b", 2, 2048)]]
cfgs = [dict(size=64, n_obs_types=16), dict(size=48, n_obs_types=16), dict(size=32, n_obs_types=256),
        dict(size=64, n_obs_types=16, n_landmarks=200), dict(size=3, n_obs_types=4), dict(size=2, n_obs_types=4),
        dict(size=5, n_obs_types=4), dict(size=16, n_obs_types=64), dict(size=128, n_obs_types=16)]
for cfg, seed, seq in itertools.product(cfgs, [0, 7, 1234], seqs):
    e_new = New(p_empty=0.5, seed=10000, **cfg)
    assert e_new._fast_walk_ok(0.0), cfg
    o = run(Old(p_empty=0.5, seed=10000, **cfg), seq, seed); m = run(e_new, seq, seed)
    s = same_run(o, m); n += 1; ok += s
    if not s: print("FAIL", cfg, seed, seq)
# data_parallel worker protocol at the rank config and the torus default
for (B, T) in [(16, 1024), (128, 128)]:
    eo, en = Old(size=64, n_obs_types=16, p_empty=0.5, seed=3), New(size=64, n_obs_types=16, p_empty=0.5, seed=3)
    for idx in range(4):
        np.random.seed(_seed_for(3 + 1, idx)); r = eo.generate_batch(B, T)
        np.random.seed(_seed_for(3 + 1, idx)); g = en.generate_batch(B, T)
        s = all(torch.equal(a, b) for a, b in zip(r[:3], g[:3])) and r[3] == g[3]; n += 1; ok += s
        if not s: print("FAIL worker", B, T, idx)
# start= argument
for st in [(0, 0), (5, 63), (31, 17)]:
    eo, en = Old(size=64, n_obs_types=16, p_empty=0.5, seed=2), New(size=64, n_obs_types=16, p_empty=0.5, seed=2)
    np.random.seed(9); a = eo.generate_trajectory(300, start=st); la = list(eo.visited_locations); sa = np.random.get_state()[1].tobytes()
    np.random.seed(9); b = en.generate_trajectory(300, start=st); lb = list(en.visited_locations); sb = np.random.get_state()[1].tobytes()
    s = all(torch.equal(x, y) for x, y in zip(a, b)) and la == lb and sa == sb and (eo.last_x, eo.last_y) == (en.last_x, en.last_y)
    n += 1; ok += s
    if not s: print("FAIL start", st)
# non-default configs must take the loop and match
for kw, ptn in [(dict(action_mode="rotate", score_moves_only=True), 0.0), (dict(obs_mode="ego"), 0.0),
                (dict(boundary="wall"), 0.0), ({}, 0.1), (dict(action_mode="rotate"), 0.0),
                (dict(action_record="allocentric"), 0.0), (dict(score_moves_only=True), 0.1),
                (dict(action_mode="rotate", action_record="allocentric", n_headings=12), 0.0),
                (dict(action_mode="rotate", heading_noise=0.1), 0.0)]:
    eo, en = Old(size=16, n_obs_types=16, p_empty=0.5, seed=1, **kw), New(size=16, n_obs_types=16, p_empty=0.5, seed=1, **kw)
    assert not en._fast_walk_ok(ptn), kw
    np.random.seed(3); a = eo.generate_batch(4, 200, p_transition_noise=ptn)
    np.random.seed(3); b = en.generate_batch(4, 200, p_transition_noise=ptn)
    s = all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])) and a[3] == b[3]; n += 1; ok += s
    if not s: print("FAIL fallback", kw, ptn)
# score_moves_only on the default torus is a no-op -> fast path is valid and must match
eo, en = Old(size=64, n_obs_types=16, p_empty=0.5, seed=4, score_moves_only=True), New(size=64, n_obs_types=16, p_empty=0.5, seed=4, score_moves_only=True)
assert en._fast_walk_ok(0.0)
s = same_run(run(eo, seqs[0], 5), run(en, seqs[0], 5)); n += 1; ok += s
# guards: modified obs_map / deltas / subclass override fall back
e = New(size=16, n_obs_types=16, p_empty=0.5, seed=1); e.obs_map = e.obs_map.int(); assert not e._fast_walk_ok(0.0)
e = New(size=16, n_obs_types=16, p_empty=0.5, seed=1); e.obs_map = e.obs_map.numpy(); assert not e._fast_walk_ok(0.0)
class SubD(New): ACTION_DELTAS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
assert not SubD(size=16, n_obs_types=16, p_empty=0.5, seed=1)._fast_walk_ok(0.0)
class Sub(New):
    def generate_trajectory(self, n_steps=128, start=None, p_transition_noise=0.0):
        self.visited_locations = [(0, 0)] * n_steps
        return (torch.zeros(2 * n_steps, dtype=torch.long), torch.zeros(2 * n_steps, dtype=torch.bool), torch.zeros(2 * n_steps, dtype=torch.bool))
e = Sub(size=16, n_obs_types=16, p_empty=0.5, seed=1)
assert not e._fast_walk_ok(0.0); t = e.generate_batch(2, 5); n += 1; ok += int(t[0].abs().sum()) == 0
from mapformer.environment_topology import OpenGridWorld, WallsGridWorld
for C in (OpenGridWorld, WallsGridWorld):
    assert not C(16, 16, 0.5, 0, 1)._fast_walk_ok(0.0), C
# FAST_WALK=False restores the loop exactly
NE.FAST_WALK = False
en = New(size=64, n_obs_types=16, p_empty=0.5, seed=5); assert not en._fast_walk_ok(0.0)
s = same_run(run(Old(size=64, n_obs_types=16, p_empty=0.5, seed=5), seqs[0][:2], 1), run(en, seqs[0][:2], 1)); n += 1; ok += s
NE.FAST_WALK = True
print(f"{ok}/{n} patched-repo-vs-HEAD checks passed")
# speed (machine idle; single thread)
env_o = Old(size=64, n_obs_types=16, p_empty=0.5, seed=0); env_n = New(size=64, n_obs_types=16, p_empty=0.5, seed=0)
for (B, T) in [(16, 1024), (128, 128)]:
    to, tn = [], []
    for rep in range(5):
        np.random.seed(rep); t = time.perf_counter(); env_o.generate_batch(B, T); to.append(time.perf_counter() - t)
        np.random.seed(rep); t = time.perf_counter(); env_n.generate_batch(B, T); tn.append(time.perf_counter() - t)
    print(f"generate_batch B={B} T={T}: HEAD {min(to)*1e3:.1f} ms  patched {min(tn)*1e3:.2f} ms  ({min(to)/min(tn):.0f}x)")
