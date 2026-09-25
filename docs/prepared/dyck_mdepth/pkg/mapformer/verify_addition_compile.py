"""Eager vs torch.compile on IDENTICAL batches and init, per SAMEBLOCK arm: loss agreement and speed.

    python3 -m mapformer.verify_addition_compile --steps 300 --device cuda:0
"""
import argparse, copy, time
import numpy as np
import torch

import mapformer.train_addition as ta
from mapformer.environment_addition import AdditionWorld, batch_fast
from mapformer.train_variant import VARIANT_MAP


def run(model, batches, dev, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []; t0 = None
    for i, (T, M, P) in enumerate(batches):
        if i == 20:
            torch.cuda.synchronize(); t0 = time.time()       # exclude compile warm-up
        loss = ta.loss_fn(model, T, M, P, dev)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    torch.cuda.synchronize()
    return np.array(losses), (time.time() - t0) / (len(batches) - 20)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device; ta.AMP = True
    env = AdditionWorld("role", max_pos=202); rng = np.random.RandomState(0)
    batches = [batch_fast(env, 1000, rng, dmax=30, random_start=True, pad_to=95)[:3] for _ in range(a.steps)]
    for v in ("ChoPos_signed", "ChoPos_abs", "ChoPos_rope", "ChoPos_coupled"):
        torch.manual_seed(0); base = VARIANT_MAP[v](34, max_pos=202).to(dev)
        eager = copy.deepcopy(base); comp = torch.compile(copy.deepcopy(base))
        le, te = run(eager, batches, dev); lc, tc = run(comp, batches, dev)
        rel = np.abs(le - lc) / np.maximum(np.abs(le), 1e-3)
        print(f"{v:16s} final loss eager {le[-1]:.4f} compiled {lc[-1]:.4f} | max |diff| {np.abs(le-lc).max():.4f} "
              f"median rel diff {np.median(rel):.4f} | s/step eager {te:.3f} compiled {tc:.3f} (x{te/tc:.2f})", flush=True)


if __name__ == "__main__":
    main()
