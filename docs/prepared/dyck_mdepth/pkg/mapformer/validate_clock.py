"""Well-posedness checks for the modular-clock navigation task."""
import numpy as np
import torch

from mapformer.environment_clock import ModularClockWorld

env = ModularClockWorld(radix_fast=16, radix_slow=8, n_obs_types=16, seed=0)
print(f"P={env.P}  vocab={env.unified_vocab_size}  radices=(slow {env.rs}, fast {env.rf})")

rng = np.random.RandomState(1)
toks, om, am, infos = env.generate_clock_batch(300, T_explore=64, T_navigate=64, rng=rng)
tgt = toks[:, 1:][am[:, :-1]]
print(f"targets: {int(am.sum())} total, {am.sum(1).float().mean():.1f}/episode; "
      f"all in [0,3] (actions)? {(tgt < env.N_ACTIONS).all().item()}")
cnt = torch.bincount(tgt, minlength=4).float()
print(f"action distribution: {[round(x,3) for x in (cnt/cnt.sum()).tolist()]}  "
      f"majority-class acc = {cnt.max().item()/cnt.sum().item():.3f}")

# path integration required: fixed goal, first navigate action varies with the walk
g = env.P // 2 + 3; firsts = []
for s in range(40):
    r = np.random.RandomState(1000 + s)
    full, _, _, _ = env.generate_clock_episode(64, 64, rng=r, goal=g)
    firsts.append(int(full[2 + 2 * 64].item()))   # first navigate action token
print(f"path-integration: for a FIXED goal, the first navigate action takes "
      f"{len(set(firsts))} distinct values over 40 walks "
      f"-> current time depends on the walk. {'OK' if len(set(firsts)) > 1 else 'WEAK'}")

# both scales matter
print(f"both-scales: hands(0)={env.hands(0)}  hands(rf)={env.hands(env.rf)} (slow changes)  "
      f"hands(1)={env.hands(1)} (fast changes)  "
      f"{'OK' if env.hands(env.rf) != env.hands(0) and env.hands(1) != env.hands(0) else 'FAIL'}")
