"""
Train a variant on the compositional-motif task.

Loss target options (--target):
  motif : CE at motif-revisit positions (motif-cell seen before, any copy) —
          the compositional objective; rewards learning motif structure.
  cross : CE at cross-instance positions only (motif seen, exact cell NOT).
  exact : CE at exact-revisit positions (paper-standard fine target).

Model is built from train_variant.VARIANT_MAP so any registered variant works
(Vanilla, VanillaEM, Hourglass_k2, HourglassFlat3, ...).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mapformer.environment_compositional import CompositionalGridWorld
from mapformer.train_variant import VARIANT_MAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--target", default="motif", choices=["motif", "cross", "exact"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--room-size", type=int, default=8)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env = CompositionalGridWorld(size=args.grid_size, room_size=args.room_size,
                                 n_templates=args.n_templates, seed=args.seed)
    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=args.grid_size,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.variant} target={args.target} params={n_params:,} "
          f"n_steps={args.n_steps} seed={args.seed}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total)
    crit = nn.CrossEntropyLoss()

    mask_idx = {"exact": 2, "motif": 3, "cross": 4}[args.target]
    losses = []
    for ep in range(args.epochs):
        t0 = time.time(); model.train(); ep_loss = 0.0; nb = 0
        for _ in range(args.n_batches):
            batch = env.generate_batch(args.batch_size, args.n_steps)
            tokens = batch[0].to(args.device)
            target_mask_full = batch[mask_idx].to(args.device)
            inp = tokens[:, :-1]; tgt = tokens[:, 1:]
            tmask = target_mask_full[:, 1:]
            if tmask.sum() == 0:
                continue
            logits = model(inp)
            loss = crit(logits[tmask], tgt[tmask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item(); nb += 1
        avg = ep_loss / max(nb, 1); losses.append(avg)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:3d}/{args.epochs} loss={avg:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.1f}s")

    ckpt = out / f"{args.variant}.pt"
    torch.save({"model_state": model.state_dict(), "variant": args.variant,
                "target": args.target, "n_layers": args.n_layers,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "vocab_size": env.unified_vocab_size, "losses": losses}, ckpt)
    with open(out / f"{args.variant}_loss.json", "w") as f:
        json.dump(losses, f)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
