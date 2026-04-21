#!/usr/bin/env python3
"""Train Level 1.5 InEKF (constant Pi, per-token R_t) MapFormer."""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-batches", type=int, default=156)
    parser.add_argument("--p-action-noise", type=float, default=0.10)
    parser.add_argument("--n-landmarks", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="figures_inekf_level15")
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    out = Path(__file__).parent / args.output_dir
    out.mkdir(exist_ok=True)

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                    n_landmarks=args.n_landmarks, seed=42)
    model = MapFormerWM_Level15InEKF(
        vocab_size=env.unified_vocab_size,
        d_model=128, n_heads=2, n_layers=1, grid_size=64,
    )
    print(f"Level 1.5 InEKF: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training p_noise={args.p_action_noise}, n_landmarks={args.n_landmarks}\n")

    losses = train(
        model, env,
        n_epochs=args.epochs, lr=3e-4,
        batch_size=128, n_steps=128,
        n_batches=args.n_batches, device=args.device,
        p_action_noise=args.p_action_noise,
    )

    ckpt = out / "MapFormer_WM_Level15InEKF.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "config": {
            "vocab_size": env.unified_vocab_size,
            "d_model": 128, "n_heads": 2, "n_layers": 1,
            "grid_size": 64, "n_obs_types": 16, "p_empty": 0.5,
            "n_landmarks": args.n_landmarks,
        },
    }, ckpt)
    print(f"\nSaved: {ckpt}")


if __name__ == "__main__":
    main()
