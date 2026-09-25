"""Capture / re-check a reference fingerprint of the DEFAULT (serial) data path.

Used to prove that wiring parallel generation into train.py leaves the serial
path byte-identical. Trains a short run and fingerprints the final weights: a
weight hash catches any change to the data stream, the RNG draw order or the
optimisation, where comparing final loss would not.
"""
import argparse, hashlib, json, os
import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train import train


def fingerprint(seed, epochs, n_batches, device):
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=42)
    m = VARIANT_MAP["Vanilla"](vocab_size=env.unified_vocab_size, d_model=128,
                               n_heads=2, n_layers=1, grid_size=64).to(device)
    train(m, env, n_epochs=epochs, batch_size=128, n_steps=128,
          n_batches=n_batches, device=device, verbose=False, schedule="cosine")
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fp = fingerprint(a.seed, a.epochs, a.n_batches, a.device)
    if a.check:
        ref = json.load(open(a.out))["fingerprint"]
        same = fp == ref
        print(f"reference {ref[:16]}...\ncurrent   {fp[:16]}...\n"
              f"serial path unchanged: {'PASS' if same else 'FAIL'}")
        raise SystemExit(0 if same else 1)
    json.dump({"fingerprint": fp, "seed": a.seed, "epochs": a.epochs,
               "n_batches": a.n_batches}, open(a.out, "w"), indent=2)
    print(f"reference fingerprint {fp[:16]}... -> {a.out}")
