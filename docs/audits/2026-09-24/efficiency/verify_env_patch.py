"""Gate for environment_fastwalk.patch: the PATCHED GridWorld vs the ORIGINAL, byte for
byte, through the public API (generate_batch, generate_trajectory), including the
global RNG state afterwards and a mixed sequence of calls. Single thread."""
import sys, itertools
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency")
from mapformer.environment import GridWorld as Old
from mfprop.environment import GridWorld as New
import mfprop.environment as NE

def run(env, seq, seed):
    np.random.seed(seed); out = []
    for kind, a, b in seq:
        if kind == "b":
            r = env.generate_batch(a, b); out.append((r[0], r[1], r[2], r[3]))
        else:
            r = env.generate_trajectory(a); out.append((r[0], r[1], r[2], list(env.visited_locations)))
    st = np.random.get_state()
    return out, (st[0], st[1].tobytes(), st[2:]), (env.last_x, env.last_y)

seq = [("b", 16, 1024), ("t", 2048, None), ("b", 3, 7), ("t", 1, None), ("b", 1, 128), ("t", 512, None)]
ok = n = 0
for cfg, seed in itertools.product(
        [dict(size=64, n_obs_types=16), dict(size=48, n_obs_types=16), dict(size=32, n_obs_types=256),
         dict(size=64, n_obs_types=16, n_landmarks=200), dict(size=3, n_obs_types=4)], [0, 7]):
    o = run(Old(p_empty=0.5, seed=10000, **cfg), seq, seed)
    m = run(New(p_empty=0.5, seed=10000, **cfg), seq, seed)
    same = o[1] == m[1] and o[2] == m[2] and len(o[0]) == len(m[0])
    for x, y in zip(o[0], m[0]):
        same &= all(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) for a, b in zip(x[:3], y[:3]))
        same &= x[3] == y[3]
    n += 1; ok += same
    if not same: print("FAIL", cfg, seed)
# non-default configs must take the loop (FAST path not used) and match
for kw, ptn in [(dict(action_mode="rotate", score_moves_only=True), 0.0), (dict(obs_mode="ego"), 0.0),
                (dict(boundary="wall"), 0.0), ({}, 0.1),
                (dict(action_mode="rotate", action_record="allocentric", n_headings=12), 0.0)]:
    eo, en = Old(size=16, n_obs_types=16, p_empty=0.5, seed=1, **kw), New(size=16, n_obs_types=16, p_empty=0.5, seed=1, **kw)
    assert not en._fast_walk_ok(ptn), kw
    np.random.seed(3); a = eo.generate_batch(4, 200, p_transition_noise=ptn)
    np.random.seed(3); b = en.generate_batch(4, 200, p_transition_noise=ptn)
    n += 1; s = all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])) and a[3] == b[3]; ok += s
    if not s: print("FAIL fallback", kw, ptn)
# a subclass overriding generate_trajectory must keep its own walk inside generate_batch
class Sub(New):
    def generate_trajectory(self, n_steps=128, start=None, p_transition_noise=0.0):
        self.visited_locations = [(0, 0)] * n_steps
        return (torch.zeros(2 * n_steps, dtype=torch.long),) * 1 + (torch.zeros(2 * n_steps, dtype=torch.bool),) * 2
e = Sub(size=16, n_obs_types=16, p_empty=0.5, seed=1)
assert not e._fast_walk_ok(0.0); t = e.generate_batch(2, 5); n += 1; ok += int(t[0].abs().sum()) == 0
# FAST_WALK=False restores the loop exactly
NE.FAST_WALK = False
o = run(Old(size=64, n_obs_types=16, p_empty=0.5, seed=5), seq[:2], 1); m = run(New(size=64, n_obs_types=16, p_empty=0.5, seed=5), seq[:2], 1)
n += 1; ok += o[1] == m[1]
print(f"{ok}/{n} patched-vs-original checks passed")
