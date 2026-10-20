"""Independent CPU checks for DELTA_FREE (audit 2026-10-20). Seeds 1670-1679 only.
T1 twin pairing in TRAINING: pinned and free arm at a seed, with the pinned arm's delta overwritten by the free arm's
   draw squeezed into the clamp interior (and the free arm's likewise), train identically (losses and every weight
   bitwise) as long as no entry reaches the clamp: data stream, dropout masks, init and optimiser all paired.
T2 frozen arm: delta never moves, no AdamW state for it, the buffer equals the draw.
T3 the unmodified pinned twin vs free arm diverge (sanity: the check above is not vacuous).
T4 absorbing dead zone: a pinned entry pushed above 0 never comes back below 0 under AdamW (zero grad, multiplicative decay).
"""
import math, sys
import numpy as np
import torch

from mapformer.train_variant import VARIANT_MAP
from mapformer.environment import GridWorld
from mapformer.train import train
from mapformer.model_pope import DELTA_MIN, DELTA_MAX

KW = dict(n_epochs=2, lr=1e-3, batch_size=8, n_steps=32, n_batches=4, device="cpu", verbose=False,
          schedule="cosine")


def build(arm, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
    m = VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    return m, env


def run(arm, seed, set_delta=None, ret_state=False):
    m, env = build(arm, seed)
    if set_delta is not None:
        with torch.no_grad():
            m.layers[0].pope_delta.copy_(set_delta)
    out = train(m, env, return_state=ret_state, **KW)
    return m, out


ok = True
for pin, free in (("PoPE-Flat", "PoPE-FreeDelta"), ("MapPoPE-Flat", "MapPoPE-FreeDelta")):
    for seed in (1670, 1671):
        m0, _ = build(free, seed)
        d = m0.layers[0].pope_delta.detach().clone()
        d = d.clamp(DELTA_MIN + 0.3, DELTA_MAX - 0.3)          # interior: no entry can reach the clamp in 8 steps
        a, la = run(pin, seed, d)
        b, lb = run(free, seed, d)
        sa, sb = a.state_dict(), b.state_dict()
        diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
        inside = bool(((a.layers[0].pope_delta > DELTA_MIN) & (a.layers[0].pope_delta < DELTA_MAX)).all())
        good = la == lb and diff == [] and inside
        ok &= good
        print(f"{'ok ' if good else 'BAD'} T1 {pin} vs {free} s{seed}: losses equal {la == lb} ({la}), "
              f"tensors differing {diff}, delta stayed interior {inside}")
        # T3: the real arms (pinned zero init vs free draw) do differ
        c, lc = run(pin, seed)
        e, le = run(free, seed)
        good = lc != le
        ok &= good
        print(f"{'ok ' if good else 'BAD'} T3 real twins differ: {lc} vs {le}")

for seed in (1672, 1673):
    m, (losses, opt, sch) = run("MapPoPE-FrozenDelta", seed, ret_state=True)
    L = m.layers[0]
    st = opt.state.get(L.pope_delta, {})
    good = torch.equal(L.pope_delta, L.pope_delta_init) and len(st) == 0 and L.pope_delta.grad is None
    m2, _ = build("MapPoPE-FreeDelta", seed)
    good &= torch.equal(L.pope_delta_init, m2.layers[0].pope_delta)
    ok &= good
    print(f"{'ok ' if good else 'BAD'} T2 frozen s{seed}: unmoved {torch.equal(L.pope_delta, L.pope_delta_init)}, "
          f"AdamW state entries {len(st)}, grad {L.pope_delta.grad}, init = free draw")

# T4: absorbing dead zone. One parameter p, clamp(p) used in a loss whose gradient pushes p UP until p > 0.
p = torch.nn.Parameter(torch.tensor([-0.002]))
opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.05)
hist = []
for step in range(3000):
    opt.zero_grad()
    loss = -p.clamp(DELTA_MIN, DELTA_MAX).sum() if step < 10 else (p.clamp(DELTA_MIN, DELTA_MAX) * 0).sum()
    loss.backward(); opt.step(); hist.append(float(p))
crossed_back = any(h <= 0 for h in hist[20:])
print(f"{'ok ' if not crossed_back else 'BAD'} T4 dead zone absorbing: p after push {hist[15]:+.5f}, "
      f"after 3000 zero-grad steps {hist[-1]:+.6f}, ever back <= 0: {crossed_back}")
print("ALL OK" if ok else "SOME CHECK FAILED")
sys.exit(0 if ok else 1)
