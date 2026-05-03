"""Training entry point for continuous 2D nav (Cueva/Wei/Sorscher style).

Runs Vanilla / Level15 (and others) on ContinuousNav2D with MSE loss
on predicted DoG place-cell vectors at action positions.

Loss design:
    Input:    interleaved (a_0, o_0, a_1, o_1, ..., a_{T-1}, o_{T-1})
    At sequence position 2t (action a_t), the model has causally seen
    (a_0..a_t, o_0..o_{t-1}). Its output at 2t is regressed to o_t —
    the place-cell vector at the position arrived at after taking a_t.

Usage:
    python3 -m mapformer.train_continuous --variant Continuous --seed 0 \
        --epochs 30 --buffer-size 10000 \
        --output-dir runs/cnav/Vanilla/seed0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.continuous_nav import ContinuousNav2D_Cached
from mapformer.model_continuous import (
    MapFormerWM_Continuous,
    MapFormerWM_Continuous_Level15,
    MapFormerEM_Continuous,
    MapFormerEM_Continuous_Level15,
)

VARIANT_CLS = {
    "Vanilla":   MapFormerWM_Continuous,
    "Level15":   MapFormerWM_Continuous_Level15,
    "VanillaEM": MapFormerEM_Continuous,
    "Level15EM": MapFormerEM_Continuous_Level15,
}


def _loss_fn(preds_at_actions: torch.Tensor, obs: torch.Tensor,
             loss_type: str, temperature: float = 0.1) -> torch.Tensor:
    """Continuous-nav loss.

    'mse' (legacy, broken): mean((preds - DoG)^2). Has a degenerate
        near-zero minimum because DoG targets are sparse (~0.6% nonzero
        entries). First CNAV run hit this — model collapsed to
        predicting zero, position decoding at chance.

    'soft_ce' (smoothed classification, temperature-sensitive):
            target_p = softmax(DoG / temperature)
            loss     = - sum(target_p * log_softmax(preds), dim=-1).mean()
        With temperature=1.0 the target is nearly uniform (DoG.max() ≈
        0.33 → peak prob 0.54% vs uniform 0.39%; gradient too weak).
        Default temperature=0.1 makes the target peaked at the closest
        place cell while keeping graded info from neighbours.

    'hard_ce' (recommended default): treat the closest place cell as
        the target, plain cross-entropy:
            target_idx = obs.argmax(dim=-1)
            loss       = F.cross_entropy(preds, target_idx)
        Robust, no temperature to tune, always informative gradient.
        Loses graded firing-rate info but the spatial structure that
        Sorscher-style theory cares about lives in the place_proj
        mapping, not the loss.
    """
    import torch.nn.functional as F
    if loss_type == "mse":
        return (preds_at_actions - obs).pow(2).mean()
    if loss_type == "soft_ce":
        log_p = F.log_softmax(preds_at_actions, dim=-1)         # (B, T, n_pc)
        target_p = F.softmax(obs / temperature, dim=-1)         # (B, T, n_pc)
        return -(target_p * log_p).sum(dim=-1).mean()
    if loss_type == "hard_ce":
        target_idx = obs.argmax(dim=-1)                         # (B, T)
        return F.cross_entropy(
            preds_at_actions.reshape(-1, preds_at_actions.shape[-1]),
            target_idx.reshape(-1),
        )
    raise ValueError(f"unknown loss_type: {loss_type}")


def train(model, env, n_epochs, n_batches, batch_size, n_steps, lr,
          weight_decay, device, loss_type: str = "hard_ce",
          loss_temperature: float = 0.1, log_every=5):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=n_epochs * n_batches,
    )
    losses = []
    for ep in range(n_epochs):
        t0 = time.time()
        ep_loss = 0.0
        for _ in range(n_batches):
            actions, _actions_actual, _positions, _headings, obs = env.generate_batch(
                batch_size, n_steps,
            )
            actions = actions.to(device)
            obs     = obs.to(device)

            preds = model(actions, obs)               # (B, 2T, obs_dim) logits
            preds_at_actions = preds[:, 0::2]         # (B, T, obs_dim)
            loss = _loss_fn(preds_at_actions, obs,
                             loss_type=loss_type, temperature=loss_temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()

        avg = ep_loss / n_batches
        losses.append(avg)
        if (ep + 1) % log_every == 0:
            dt = time.time() - t0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {ep+1:3d}/{n_epochs} | MSE: {avg:.5f} | "
                  f"LR: {cur_lr:.2e} | {dt:.1f}s")
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(VARIANT_CLS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-batches", type=int, default=156)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=25_000)
    p.add_argument("--size", type=float, default=64.0)
    p.add_argument("--n-place-cells", type=int, default=256)
    p.add_argument("--sigma-E", type=float, default=1.5)
    p.add_argument("--sigma-I", type=float, default=3.0)
    p.add_argument("--v-mean", type=float, default=0.7)
    p.add_argument("--v-std", type=float, default=0.3)
    p.add_argument("--omega-std", type=float, default=0.5)
    p.add_argument("--v-noise-std", type=float, default=0.0)
    p.add_argument("--omega-noise-std", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--loss", type=str, default="hard_ce",
                   choices=["mse", "soft_ce", "hard_ce"],
                   help="Loss for DoG predictions. 'mse' is degenerate "
                        "(broken default in earlier runs); 'soft_ce' "
                        "needs careful temperature tuning; 'hard_ce' "
                        "(default) treats the closest place cell as a "
                        "classification target — robust, no temperature.")
    p.add_argument("--loss-temperature", type=float, default=0.1,
                   help="Softmax temperature for soft_ce target. Default 0.1 "
                        "is reasonable for DoG values that peak near 0.33; "
                        "temperature=1.0 gives near-uniform targets.")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-grid-units", type=int, default=0,
                   help="If > 0, insert a Linear -> ReLU -> Linear bottleneck "
                        "with this hidden size before the DoG output head. "
                        "The ReLU activations are the candidate hex-emergence "
                        "layer (Sorscher's non-negativity condition).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env = ContinuousNav2D_Cached(
        size=args.size, n_place_cells=args.n_place_cells,
        sigma_E=args.sigma_E, sigma_I=args.sigma_I,
        v_mean=args.v_mean, v_std=args.v_std, omega_std=args.omega_std,
        v_noise_std=args.v_noise_std, omega_noise_std=args.omega_noise_std,
        seed=args.seed, buffer_size=args.buffer_size,
    )
    cls = VARIANT_CLS[args.variant]
    extra = {}
    if args.variant == "Level15":
        extra["log_R_init_bias"] = 0.0  # WM has content fallback
    elif args.variant == "Level15EM":
        extra["log_R_init_bias"] = 3.0  # EM Hadamard has no fallback at init

    model = cls(
        action_dim=2, obs_dim=args.n_place_cells,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        grid_size=int(args.size), n_grid_units=args.n_grid_units, **extra,
    )
    print(f"{args.variant} seed={args.seed} v_noise={args.v_noise_std} "
          f"omega_noise={args.omega_noise_std}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    losses = train(
        model, env,
        n_epochs=args.epochs, n_batches=args.n_batches,
        batch_size=args.batch_size, n_steps=args.n_steps,
        lr=args.lr, weight_decay=args.weight_decay, device=args.device,
        loss_type=args.loss, loss_temperature=args.loss_temperature,
    )

    ckpt = out / f"{args.variant}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "variant": args.variant,
        "config": vars(args),
    }, ckpt)
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
