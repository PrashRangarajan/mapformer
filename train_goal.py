"""Train a MapFormer-style model on goal-directed action prediction.

Loss: cross-entropy on action token at navigate-phase positions, where the
ground truth is the BFS-optimal next action toward the goal.

Same model classes as standard training (vocab unchanged; model already
predicts every position). Different env (GoalDirectedGridWorld) and
different loss mask (action_target_mask instead of revisit_mask).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment_goal import GoalDirectedGridWorld
from mapformer.train_variant import VARIANT_MAP


def train_goal(
    model, env, n_epochs=20, lr=3e-4, batch_size=128,
    T_explore=64, T_navigate=64, n_batches=64, device="cuda",
    init_ckpt: str | None = None,
):
    if init_ckpt is not None:
        c = torch.load(init_ckpt, map_location=device, weights_only=False)
        msd = c["model_state_dict"]
        # Allow loading the body weights from a prediction-trained ckpt
        own = model.state_dict()
        loaded = 0
        for k, v in msd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        model.load_state_dict(own)
        print(f"Loaded {loaded}/{len(msd)} params from {init_ckpt}")

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                              total_iters=n_epochs * n_batches)

    losses = []
    rng = np.random.RandomState(0)
    for ep in range(n_epochs):
        ep_loss = 0.0; ep_correct = 0; ep_total = 0
        for b in range(n_batches):
            tokens, _, act_mask, _ = env.generate_goal_batch(
                batch_size, T_explore=T_explore, T_navigate=T_navigate, rng=rng,
            )
            tokens = tokens.to(device); act_mask = act_mask.to(device)

            # Predict NEXT token at every position
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            mask = act_mask[:, :-1]  # mask applies to PREDICTING positions

            logits = model(inp)                      # (B, L-1, V)
            lp = F.log_softmax(logits, dim=-1)
            # CE over action vocabulary only (first 4 tokens), but we let the
            # model express any token — the supervision keeps it on actions.
            tgt_flat = tgt[mask]
            lp_flat = lp[mask]
            if tgt_flat.numel() == 0:
                continue
            loss = F.nll_loss(lp_flat, tgt_flat)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            ep_loss += loss.item()
            preds = lp_flat.argmax(-1)
            ep_correct += (preds == tgt_flat).sum().item()
            ep_total += tgt_flat.numel()
        avg = ep_loss / max(1, n_batches)
        acc = ep_correct / max(1, ep_total)
        losses.append(avg)
        print(f"  Epoch {ep+1:3d}/{n_epochs} | Loss: {avg:.4f} | Acc: {acc:.3f} | "
              f"LR: {sched.get_last_lr()[0]:.2e}")
    return losses


def evaluate_goal(model, env, T_explore, T_navigate, n_trials, device="cuda", seed=0):
    rng = np.random.RandomState(seed)
    correct = total = 0
    nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, act_mask, _ = env.generate_goal_episode(
                T_explore=T_explore, T_navigate=T_navigate, rng=rng,
            )
            tokens = tokens.unsqueeze(0).to(device)
            act_mask = act_mask.unsqueeze(0).to(device)
            inp = tokens[:, :-1]; tgt = tokens[:, 1:]
            mask = act_mask[:, :-1]
            try:
                logits = model(inp)
            except Exception:
                return None, None
            lp = F.log_softmax(logits, dim=-1)
            tgt_flat = tgt[mask]; lp_flat = lp[mask]
            if tgt_flat.numel() == 0: continue
            preds = lp_flat.argmax(-1)
            correct += (preds == tgt_flat).sum().item()
            total += tgt_flat.numel()
            nll_sum += -lp_flat.gather(-1, tgt_flat.unsqueeze(-1)).sum().item()
    return (correct / max(1, total), nll_sum / max(1, total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-landmarks", type=int, default=200)
    parser.add_argument("--n-obs-types", type=int, default=16)
    parser.add_argument("--T-explore", type=int, default=64)
    parser.add_argument("--T-navigate", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="Optional: warm-start from a prediction-trained ckpt.")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env = GoalDirectedGridWorld(
        size=64, n_obs_types=args.n_obs_types, p_empty=0.5,
        n_landmarks=args.n_landmarks, seed=args.seed,
    )
    cls = VARIANT_MAP[args.variant]
    model = cls(
        vocab_size=env.unified_vocab_size,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, grid_size=64,
    )
    print(f"{args.variant} seed={args.seed} n_landmarks={args.n_landmarks} "
          f"T_explore={args.T_explore} T_navigate={args.T_navigate}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    losses = train_goal(
        model, env,
        n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        T_explore=args.T_explore, T_navigate=args.T_navigate,
        n_batches=args.n_batches, device=args.device,
        init_ckpt=args.init_ckpt,
    )

    # Eval on a held-out env seed
    env_test = GoalDirectedGridWorld(
        size=64, n_obs_types=args.n_obs_types, p_empty=0.5,
        n_landmarks=args.n_landmarks, seed=1000,
    )
    acc, nll = evaluate_goal(
        model, env_test, args.T_explore, args.T_navigate, n_trials=200,
        device=args.device, seed=2000,
    )
    print(f"Held-out goal-directed acc: {acc:.3f} | nll: {nll:.3f}")

    ckpt_path = out / f"{args.variant}_goal.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "variant": args.variant,
        "seed": args.seed,
        "test_acc": acc, "test_nll": nll,
        "config": {
            "vocab_size": env.unified_vocab_size,
            "d_model": args.d_model, "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "grid_size": 64, "n_obs_types": args.n_obs_types, "p_empty": 0.5,
            "n_landmarks": args.n_landmarks,
            "T_explore": args.T_explore, "T_navigate": args.T_navigate,
        },
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
