"""THEORY REVIEW (CPU): sensitivity of a held code's revisit phase to a common rescaling of a channel (omega -> (1+eps) omega,
or AdamW's decoupled weight decay). Wrapped (non-negative) code: d phase / d eps at a revisit = the raw interval phase =
2 pi * (# backward moves) per wrapping axis (exact: a(+) + a(-) = 2 pi). Signed code: = delta . d, which is 0 at every
revisit with zero net displacement. Curvature of the loss along eps ~ E[(d phase / d eps)^2] over revisit pairs. Walks as
the environments: runs of U{1..10} moves; ring N 32 T 128 (2 actions), torus N 64 T 1024 (4 actions)."""
import numpy as np
def walk(rng, T, n_act):
    a_seq = []
    while len(a_seq) < T:
        a = rng.randint(n_act); a_seq += [a] * rng.randint(1, 11)
    return np.array(a_seq[:T])
def stats(N, T, n_act, n=300, seed=0):
    rng = np.random.RandomState(seed); D = 1 if n_act == 2 else 2
    steps = {2: np.array([[1], [-1]]), 4: np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])}[n_act]
    back = {2: [1], 4: [1, 3]}[n_act]              # the moves whose angle is 2 pi - a in the wrapped code
    raw_last, raw_all, disp_last, gaps = [], [], [], []
    for _ in range(n):
        a = walk(rng, T, n_act); p = np.cumsum(steps[a], 0); cell = p % N
        nb = np.cumsum(np.isin(a, back))
        key = [tuple(c) for c in cell]; seen = {}
        for t in range(T):
            if key[t] in seen:
                prev = seen[key[t]]
                s = prev[-1]
                raw_last.append(2 * np.pi * (nb[t] - nb[s])); gaps.append(t - s)
                disp_last.append(np.abs(p[t] - p[s]).sum())
                raw_all.append(np.mean([(2 * np.pi * (nb[t] - nb[u])) ** 2 for u in prev]))
                prev.append(t)
            else:
                seen[key[t]] = [t]
    raw_last = np.array(raw_last); disp_last = np.array(disp_last)
    return dict(revisits_per_seq=len(gaps) / n, gap_mean=np.mean(gaps), gap_rms=np.sqrt(np.mean(np.square(gaps))),
                wrapped_ms_last=np.mean(raw_last ** 2), wrapped_ms_all=np.mean(raw_all),
                share_zero_disp=np.mean(disp_last == 0), disp_rms_last=np.sqrt(np.mean(disp_last.astype(float) ** 2)))
r = stats(32, 128, 2); t = stats(64, 1024, 4, n=60)
for k in r:
    print(f"{k:18s} ring {r[k]:12.2f}   torus {t[k]:12.2f}   ratio {t[k] / max(r[k], 1e-12):8.2f}")
print("lr-weighted (continuation lr ring 1e-3, torus 1e-4; torus_hi 1e-3): torus/ring = "
      f"{1e-4 * t['wrapped_ms_last'] / (1e-3 * r['wrapped_ms_last']):.2f}; torus_hi/ring = {t['wrapped_ms_last'] / r['wrapped_ms_last']:.2f}")
