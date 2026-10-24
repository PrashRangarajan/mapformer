import numpy as np, torch
from mapformer.environment import GridWorld
for s in [2490, 2491] + list(range(2400, 2408)):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=s); np.random.seed(s)
    cnt = np.zeros(21)
    for _ in range(6):
        tok, om, rm, _l = env.generate_batch(16, 1024)
        t = tok[:, 1:][rm[:, 1:]].numpy(); cnt += np.bincount(t, minlength=21)
    p = cnt / cnt.sum(); H = -(p[p > 0] * np.log(p[p > 0])).sum()
    print(s, "revisit-target entropy %.4f (n=%d)" % (H, cnt.sum()))
