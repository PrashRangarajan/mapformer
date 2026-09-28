"""Rule-11 gate (map redrawn per trajectory, so train/held-out seeds only change the walk RNG) for H3 (CANCEL_PREREG.md), calling the task code (environment_cancel.GridWorldCancel).
Per p_plus: revisit rate; floors on the HELD-OUT map (env seed 10000) scored at revisit positions:
best constant, and an n-gram over the preceding tokens (orders 1-5) fit on the training map
(seed 0); plus a 'fixed-lag clock' that copies the observation exactly `size` steps back, the best
an index-only lookup at the ring period can do."""
from collections import Counter, defaultdict
import numpy as np
from mapformer.environment_cancel import GridWorldCancel

N, T, NTR, NTE = 32, 128, 3000, 1000
for p in (0.5, 0.75, 0.9, 1.0):
    np.random.seed(1)
    tr = GridWorldCancel(size=N, seed=0, p_plus=p); te = GridWorldCancel(size=N, seed=10000, p_plus=p)
    TR = [tr.generate_trajectory(T) for _ in range(NTR)]; TE = [te.generate_trajectory(T) for _ in range(NTE)]
    rev = np.mean([r[1::2].float().mean().item() for _, _, r in TE])
    tgt = [(tok.tolist(), i) for tok, _, r in TE for i in np.nonzero(r.numpy())[0]]
    ys = [tok[i] for tok, i in tgt]
    const = Counter(ys).most_common(1)[0][1] / len(ys)
    ng = {}
    for n in range(1, 6):
        tab = defaultdict(Counter)
        for tok, _, r in TR:
            t = tok.tolist()
            for i in np.nonzero(r.numpy())[0]:
                tab[tuple(t[max(0, i - n):i])][t[i]] += 1
        fb = Counter(y for tok, _, r in TR for y in [tok.tolist()[i] for i in np.nonzero(r.numpy())[0]]).most_common(1)[0][0]
        ng[n] = np.mean([(tab[tuple(tok[max(0, i - n):i])].most_common(1)[0][0] if tab[tuple(tok[max(0, i - n):i])] else fb) == tok[i]
                         for tok, i in tgt])
    lag = np.mean([tok[i - 2 * N] == tok[i] if i >= 2 * N else False for tok, i in tgt])
    print(f"p={p:.2f} revisit rate {rev:.3f} | n targets {len(ys)} | const {const:.3f} | n-gram " +
          " ".join(f"{n}:{v:.3f}" for n, v in ng.items()) + f" | fixed-lag clock {lag:.3f}")
