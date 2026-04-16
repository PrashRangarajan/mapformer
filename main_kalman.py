#!/usr/bin/env python3
"""Train MapFormer-WM + InEKF and save checkpoint for comparison."""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_kalman import MapFormerWM_InEKF
from mapformer.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-batches", type=int, default=156)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-obs-types", type=int, default=16)
    parser.add_argument("--p-empty", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="figures_kalman")
    parser.add_argument("--p-action-noise", type=float, default=0.10,
                        help="Fraction of training actions to corrupt (noise injection)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(__file__).parent / args.output_dir
    out.mkdir(exist_ok=True)

    env = GridWorld(size=args.grid_size, n_obs_types=args.n_obs_types,
                    p_empty=args.p_empty, seed=args.seed)

    model = MapFormerWM_InEKF(
        vocab_size=env.unified_vocab_size,
        d_model=128, n_heads=2, n_layers=1, grid_size=args.grid_size,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MapFormer-WM+InEKF: {n_params:,} params")
    print(f"Vocab={env.unified_vocab_size}, grid={args.grid_size}, p_empty={args.p_empty}")
    print(f"Training on {args.epochs * args.n_batches * args.batch_size:,} sequences")
    print()

    print(f"Training with action noise p={args.p_action_noise}\n")
    losses = train(
        model, env,
        n_epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, n_steps=args.n_steps,
        n_batches=args.n_batches, device=args.device,
        p_action_noise=args.p_action_noise,
    )

    ckpt_path = out / "MapFormer_WM_InEKF.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "config": {
            "vocab_size": env.unified_vocab_size,
            "d_model": 128, "n_heads": 2, "n_layers": 1,
            "grid_size": args.grid_size,
            "n_obs_types": args.n_obs_types,
            "p_empty": args.p_empty,
        },
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path}")


if __name__ == "__main__":
    main()
