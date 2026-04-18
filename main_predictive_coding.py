#!/usr/bin/env python3
"""Train MapFormer-WM with predictive-coding corrections.

Includes optional auxiliary loss on prediction error (forces forward model
to actually model observations rather than collapse).
"""

import argparse
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding


def train_pc(
    model, env,
    n_epochs=50, lr=3e-4,
    batch_size=128, n_steps=128, n_batches=156,
    device="cuda",
    weight_decay=0.05,
    p_action_noise=0.10,
    aux_coef=0.1,
    verbose=True,
):
    """Training loop that adds auxiliary prediction-error loss."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = n_epochs * n_batches
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    losses = []
    aux_losses = []
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_aux = 0.0

        for _ in range(n_batches):
            tokens, obs_mask, revisit_mask, _ = env.generate_batch(batch_size, n_steps)
            tokens = tokens.to(device)
            revisit_mask = revisit_mask.to(device)

            # Optional action noise (data augmentation)
            if p_action_noise > 0:
                even = torch.zeros_like(tokens, dtype=torch.bool)
                even[:, 0::2] = True
                noise_mask = (torch.rand_like(tokens, dtype=torch.float) < p_action_noise) & even
                rand_a = torch.randint(0, env.N_ACTIONS, tokens.shape, device=device)
                tokens = torch.where(noise_mask, rand_a, tokens)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            target_mask = revisit_mask[:, 1:]

            logits = model(input_tokens)

            if target_mask.sum() == 0:
                continue

            logits_m = logits[target_mask]
            targets_m = target_tokens[target_mask]
            next_token_loss = criterion(logits_m, targets_m)

            # Auxiliary predictive-coding loss
            aux_loss = model.prediction_error_loss()
            total_loss = next_token_loss + aux_coef * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += next_token_loss.item()
            epoch_aux += aux_loss.item()

        avg_loss = epoch_loss / n_batches
        avg_aux = epoch_aux / n_batches
        losses.append(avg_loss)
        aux_losses.append(avg_aux)

        if verbose and (epoch + 1) % 5 == 0:
            dt = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} "
                  f"| Aux: {avg_aux:.4f} | LR: {lr_now:.2e} | {dt:.1f}s")

    return losses, aux_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-batches", type=int, default=156)
    parser.add_argument("--p-action-noise", type=float, default=0.10)
    parser.add_argument("--aux-coef", type=float, default=0.1,
                        help="Weight on prediction-error auxiliary loss")
    parser.add_argument("--n-landmarks", type=int, default=0,
                        help="Number of unique-ID landmark cells (0 = disabled)")
    parser.add_argument("--output-dir", type=str, default="figures_predictive_coding")
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    out = Path(__file__).parent / args.output_dir
    out.mkdir(exist_ok=True)

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                    n_landmarks=args.n_landmarks, seed=42)
    model = MapFormerWM_PredictiveCoding(
        vocab_size=env.unified_vocab_size,
        d_model=128, n_heads=2, n_layers=1, grid_size=64,
    )
    print(f"Predictive-Coding MapFormer: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training p_noise={args.p_action_noise}, aux={args.aux_coef}, n_landmarks={args.n_landmarks}")
    print()

    losses, aux_losses = train_pc(
        model, env,
        n_epochs=args.epochs, n_batches=args.n_batches,
        device=args.device,
        p_action_noise=args.p_action_noise,
        aux_coef=args.aux_coef,
    )

    ckpt = out / "MapFormer_WM_PredictiveCoding.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "aux_losses": aux_losses,
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
