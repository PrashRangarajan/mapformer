"""Trainer for landmarks vs path integration in words (TW_LANDMARK_PREREG.md). train_textworld.py's recipe on
environment_tw_landmark.TextWorldLandmark at a training name rate; the model, the optimiser, the schedule and the
data path are train_textworld's / train.py's unchanged. Writes <arm>.pt and train.json in --output-dir; every readout
is computed afterwards from the checkpoint by tw_landmark_eval.py (the driver re-checks the md5 guard first).

The vocabulary (58 text-world words + 'reached' + 512 names) is the same at every rate, so for one seed the model
initialisation is identical across rates and the walk stream is the same walks (names never consume global draws)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_tw_landmark import TextWorldLandmark
from mapformer.train import train
from mapformer.train_variant import VARIANT_MAP

VARIANT = {"MapWM": "Vanilla_r4", "RoPE": "RoPE"}


def build(arm, env, n_layers, size=64):
    return VARIANT_MAP[VARIANT[arm]](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=n_layers,
                                     grid_size=size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(VARIANT))
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--name-rate", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    assert a.name_rate in (0.0, 0.5, 1.0), a.name_rate

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = TextWorldLandmark(size=a.size, seed=a.seed, name_rate=a.name_rate)
    model = build(a.arm, env, a.n_layers, a.size)
    assert len(model.layers) == a.n_layers, (len(model.layers), a.n_layers)   # rule 17
    print(f"{a.arm} L{a.n_layers} rate={a.name_rate} seed={a.seed} vocab={env.unified_vocab_size} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "arm": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size}}, out / f"{a.arm}.pt")
    json.dump({"final_loss": float(np.mean(losses[-max(1, len(losses) // 20):])), "epochs": a.epochs},
              open(out / "train.json", "w"), indent=2)


if __name__ == "__main__":
    main()
