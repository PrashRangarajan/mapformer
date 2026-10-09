"""Trainer-path reproduction of a stored text-world run (TW_AMBIG_PREREG.md). The first --epochs epochs of the stored
TW_NORMSTEP MapWM s10 run (runs/tw_normstep/p0/MapWM_s10, trained on GPU with the 900-epoch x 98-batch cosine schedule)
are re-run through train_tw_ambig's path (TextWorldAmbig at p_nm = 0 == TextWorld, model_tw_ambig.build('MapWM'),
train.train with --data-workers 3) under the STORED schedule (5% warmup of 900 x 98 steps), and the per-epoch losses are
compared with the stored ones.
  CPU (tw_ambig_checks.py C10): close, not bitwise (dropout RNG and kernels differ between devices).
  GPU (the GPU pilot): bitwise if the stored run's kernels were deterministic; the difference is printed either way.
    python3 docs/audits/2026-10-08/tw_ambig_repro.py --device cuda:0 --epochs 2
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
import importlib
TR = importlib.import_module("mapformer.train")   # the module (mapformer/__init__ exports a function named train)
from mapformer.environment_tw_ambig import TextWorldAmbig
from mapformer.model_tw_ambig import build

REPO = "/home/prashr/mapformer"


def repro(device, epochs=1, seed=10):
    b = torch.load(f"{REPO}/runs/tw_normstep/p0/MapWM_s{seed}/MapWM.pt", map_location="cpu", weights_only=False)
    total = 900 * 98; warm = max(1, int(0.05 * total))
    real = TR.optim.lr_scheduler.LambdaLR
    TR.optim.lr_scheduler.LambdaLR = lambda opt, f: real(opt, lambda s: (s + 1) / warm if s < warm else
                                                       0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min((s - warm) / (total - warm), 1.0))))
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        env = TextWorldAmbig(seed=seed, p_nm=0.0); m = build("MapWM", env)
        los = TR.train(m, env, n_epochs=epochs, lr=1e-3, batch_size=16, n_steps=1024, n_batches=98, device=device,
                       schedule="cosine", data_workers=3, verbose=False)
    finally:
        TR.optim.lr_scheduler.LambdaLR = real
    return los, b["losses"][:epochs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    new, old = repro(a.device, a.epochs)
    for k, (x, y) in enumerate(zip(new, old)):
        print(f"epoch {k + 1}: re-run {x!r} stored {y!r} diff {x - y:+.3e} {'BITWISE' if x == y else ''}")
