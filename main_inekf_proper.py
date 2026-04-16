#!/usr/bin/env python3
"""Train MapFormer-WM with a proper Invariant EKF (constant Q, actual state correction)."""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_inekf_proper import MapFormerWM_ProperInEKF
from mapformer.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-batches", type=int, default=156)
    parser.add_argument("--p-action-noise", type=float, default=0.10)
    parser.add_argument("--output-dir", type=str, default="figures_inekf_proper")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(__file__).parent / args.output_dir
    out.mkdir(exist_ok=True)

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=args.seed)
    model = MapFormerWM_ProperInEKF(
        vocab_size=env.unified_vocab_size,
        d_model=128, n_heads=2, n_layers=1, grid_size=64,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"Proper InEKF model: {n:,} params")
    print(f"Training with action noise p={args.p_action_noise}")
    print()

    losses = train(
        model, env,
        n_epochs=args.epochs, lr=3e-4,
        batch_size=128, n_steps=128,
        n_batches=args.n_batches, device=args.device,
        p_action_noise=args.p_action_noise,
    )

    ckpt = out / "MapFormer_WM_ProperInEKF.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "config": {
            "vocab_size": env.unified_vocab_size,
            "d_model": 128, "n_heads": 2, "n_layers": 1,
            "grid_size": 64, "n_obs_types": 16, "p_empty": 0.5,
        },
    }, ckpt)
    print(f"\nSaved: {ckpt}")


if __name__ == "__main__":
    main()
