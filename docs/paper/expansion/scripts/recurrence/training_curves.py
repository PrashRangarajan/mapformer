"""Training-loss trajectories of A / L / L4 / C (stored per-epoch losses in the checkpoints).

Search-side readouts: first epoch at which the per-epoch training loss falls below fixed levels, the
loss at fixed epochs, and the final-5% mean (the registered SOLVED quantity). If the loop/depth act on
SEARCH they should change WHEN and WHETHER runs leave the plateau; capacity limits would show as a
floor reached at the same time.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT, load  # noqa

LEVELS = (0.5, 0.3, 0.15, 0.08, 0.05)
EPOCHS = (50, 100, 200, 400, 600, 900)
res = {}
for arm in ("A", "L", "L4", "C"):
    res[arm] = {}
    for s in range(8):
        _m, _c, b = load(arm, s)
        x = np.array(b["losses"], dtype=float)
        first = {str(l): (int(np.argmax(x < l)) + 1 if (x < l).any() else None) for l in LEVELS}
        at = {str(e): float(x[:e][-10:].mean()) for e in EPOCHS}
        res[arm][s] = {"first_below": first, "loss_at": at, "tail5": float(x[-45:].mean()),
                       "min": float(x.min())}
    print(arm)
    for l in LEVELS:
        v = [res[arm][s]["first_below"][str(l)] for s in range(8)]
        print(f"  first epoch < {l:<5}: " + " ".join("--" if a is None else str(a) for a in v))
    for e in EPOCHS:
        print(f"  loss@{e:<4}: " + " ".join(f"{res[arm][s]['loss_at'][str(e)]:.3f}" for s in range(8)))
json.dump(res, open(f"{OUT}/training_curves.json", "w"), indent=1)
