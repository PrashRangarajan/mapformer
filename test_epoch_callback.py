"""Rule 19: the optional `train(epoch_callback=...)` hook must not change training.

Three short runs of the G-1 recipe on the torus (shared r2, T = 64, 3 epochs, 4 batches, cosine,
data_workers 0 by default, --data-workers 3 for the batch's own setting), same seed:
  A  no callback (the default path every earlier batch used)
  A' no callback again (the determinism floor: if A != A', a difference below is not the hook)
  B  a callback that snapshots the geometry state to disk every epoch
Pass = A, A' and B have bitwise-equal per-epoch losses and final state dicts, and B's snapshots
reproduce B's weights at the epochs they were taken. (That the hook is train.py's only change is shown by
`git diff train.py`, not by this script.)

  cd /home/prashr && python3 -m mapformer.test_epoch_callback [--device cuda:1]
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train import train
from mapformer.train_variant import VARIANT_MAP


def run(device, callback=None, seed=4321, workers=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
    m = VARIANT_MAP["Vanilla"](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1,
                               grid_size=64)
    losses = train(m, env, n_epochs=3, lr=1e-3, batch_size=8, n_steps=64, n_batches=4, device=device,
                   verbose=False, schedule="cosine", data_workers=workers, epoch_callback=callback)
    return losses, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-workers", type=int, default=0, help="the G-1 batch uses 3")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    la, sa = run(a.device, workers=a.data_workers)
    la2, sa2 = run(a.device, workers=a.data_workers)
    floor_ok = la == la2 and all(torch.equal(sa[k], sa2[k]) for k in sa)
    print(f"determinism floor (A vs A'): {'bitwise equal' if floor_ok else 'DIFFERENT -- hook test inconclusive'}")

    tmp = tempfile.mkdtemp(prefix="cbtest_")
    snaps = {}

    def cb(epoch, model):
        with torch.no_grad():
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(sd, os.path.join(tmp, f"e{epoch}.pt"))
        snaps[epoch] = sd

    lb, sb = run(a.device, cb, workers=a.data_workers)
    same_loss = la == lb
    same_w = all(torch.equal(sa[k], sb[k]) for k in sa)
    last = torch.load(os.path.join(tmp, "e2.pt"))
    snap_ok = all(torch.equal(last[k], sb[k]) for k in sb) and len(snaps) == 3
    print(f"losses A vs B: {'bitwise equal' if same_loss else 'DIFFERENT'} {la} {lb}")
    print(f"final weights A vs B: {'bitwise equal' if same_w else 'DIFFERENT'}")
    print(f"snapshots: {len(snaps)} taken, last equals final weights: {snap_ok}")
    ok = floor_ok and same_loss and same_w and snap_ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
