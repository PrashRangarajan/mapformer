"""
Training loop for MapFormer and baseline models.

Self-supervised objective: predict next observation given actions and
previous observations. No labels required.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Optional

from .environment import GridWorld


def train_epoch(
    model: nn.Module,
    env: GridWorld,
    optimizer: optim.Optimizer,
    batch_size: int = 32,
    seq_len: int = 64,
    n_batches: int = 100,
    device: str = "cpu",
) -> float:
    """Train for one epoch.

    Args:
        model: MapFormer or baseline model
        env: GridWorld environment
        optimizer: optimizer
        batch_size: batch size
        seq_len: trajectory length
        n_batches: number of batches per epoch
        device: device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for _ in range(n_batches):
        actions, observations, _ = env.generate_batch(batch_size, seq_len)
        actions = actions.to(device)
        observations = observations.to(device)

        # Input: (a_1..a_{T-1}, o_1..o_{T-1}), Target: o_2..o_T
        logits = model(actions[:, :-1], observations[:, :-1])
        targets = observations[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


def train(
    model: nn.Module,
    env: GridWorld,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seq_len: int = 64,
    n_batches: int = 100,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """Full training loop.

    Returns:
        List of per-epoch average losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        t0 = time.time()
        loss = train_epoch(
            model, env, optimizer, batch_size, seq_len, n_batches, device
        )
        scheduler.step()
        losses.append(loss)

        if verbose and (epoch + 1) % 5 == 0:
            dt = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

    return losses
