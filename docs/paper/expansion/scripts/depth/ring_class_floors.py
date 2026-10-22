"""Best-constant floor of each ring revisit class (POST HOC, CPU, seconds); docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md
M-DP3. Same stream and class labels as ring_depth.py (GridWorldCancel(32, seed 10000, p_plus), np seed 0, 200 walks of
T = 128; classes retrace / lap-not-retrace / neither from ring_depth.labels). The floor of a class is the share of its
most common target token (pooled over walks; the stream is the same for every model and seed).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 .../depth/ring_class_floors.py > .../depth/out/ring_class_floors_out.txt"""
import collections
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from ring_depth import labels, SIZE, T, NW, PS  # noqa: E402
from mapformer.environment_cancel import GridWorldCancel  # noqa: E402

for p in PS:
    te = GridWorldCancel(size=SIZE, seed=10000, p_plus=p); np.random.seed(0)
    C = collections.defaultdict(collections.Counter)
    for _ in range(NW):
        tok, _o, rev = te.generate_trajectory(T)
        lb = labels((tok[0::2] - te.action_offset).numpy())
        obs = tok[1::2].numpy()
        for t in np.nonzero(lb["rev"])[0]:
            k = "retrace" if lb["retr"][t] else ("lap, not retrace" if lb["lap"][t] else "neither")
            C[k][int(obs[t])] += 1
    parts = []
    for k in ("retrace", "lap, not retrace", "neither"):
        n = sum(C[k].values())
        parts.append(f"{k}: n {n}, best constant {max(C[k].values()) / n:.3f}" if n else f"{k}: n 0")
    print(f"p_plus {p}: " + "; ".join(parts))
