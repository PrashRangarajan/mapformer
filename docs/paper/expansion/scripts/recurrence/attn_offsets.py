"""Where does each pass look? Attention mass of scored revisit queries by key offset (in tokens back
from the query): 0-1 (self / own previous observation), 2-15, 16-255, >=256, per pass, mean over heads,
seeds 0-7, 10 evaluation trajectories each (np seed 1234 + s). Also the mean attention entropy (nats)."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT, load, env_for, forward_passes, trajectories  # noqa
torch.set_num_threads(8)
BINS = [(0, 2), (2, 16), (16, 256), (256, 10**9)]
res = {}
for arm in ("A", "L", "L4", "C"):
    acc = None
    for s in range(8):
        m, c, _ = load(arm, s)
        for tok, rev in trajectories(env_for(c), 1024, 10, 1234 + s):
            inp = tok[:-1]
            _lg, at = forward_passes(m, arm, inp)
            idx = torch.nonzero(rev[1:]).flatten()
            if acc is None:
                acc = np.zeros((len(at), len(BINS) + 1)); cnt = 0
            for p, A in enumerate(at):
                Aq = A[:, idx, :].mean(0)                                   # (Q, L) head-mean
                cs = torch.cat([torch.zeros(len(idx), 1), Aq.cumsum(-1)], 1)  # cs[:, j] = mass on keys < j
                for b, (lo, hi) in enumerate(BINS):
                    up = (idx - lo + 1).clamp(min=0)                        # keys <= i - lo
                    dn = (idx - hi + 1).clamp(min=0)                        # keys <= i - hi
                    acc[p, b] += float((cs.gather(1, up[:, None]) - cs.gather(1, dn[:, None])).sum())
                ent = -(A[:, idx, :].clamp_min(1e-12).log() * A[:, idx, :]).sum(-1).mean()
                acc[p, -1] += float(ent) * len(idx)
            cnt += len(idx)
    res[arm] = (acc / cnt).tolist()
    print(arm)
    for p, r in enumerate(acc / cnt):
        print(f"  pass {p+1}: " + " ".join(f"[{lo},{hi if hi < 10**8 else 'inf'}) {r[b]:.3f}" for b, (lo, hi) in enumerate(BINS))
              + f"  entropy {r[-1]:.2f}")
json.dump(res, open(f"{OUT}/attn_offsets.json", "w"), indent=1)
