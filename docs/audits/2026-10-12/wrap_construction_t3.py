import numpy as np
N, T = 32, 128
for p in (0.5, 0.9, 1.0):
  for TT in (128, 512):
    np.random.seed(123); nrev = nwrap = 0; ntraj_wrap = 0; R = 400
    for _ in range(R):
        x = 0; seenU = set(); seenC = set(); tw = False; t = 0
        while t < TT:
            s = 1 if np.random.random() < p else -1; k = np.random.randint(1, 11)
            for _ in range(k):
                if t >= TT: break
                x += s; c = x % N
                if c in seenC:
                    nrev += 1
                    if x not in seenU: nwrap += 1; tw = True
                seenC.add(c); seenU.add(x); t += 1
        ntraj_wrap += tw
    print(f"p {p} T {TT}: revisits/traj {nrev/R:.1f}; wrap-only revisit fraction {nwrap/nrev:.3f}; trajectories with any wrap-only revisit {ntraj_wrap/R:.3f}")
