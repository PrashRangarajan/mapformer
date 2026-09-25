"""Train / evaluate a variant on hierarchical goal-directed navigation.

Loss: next-action cross-entropy at navigate-phase positions (BFS-optimal action
toward the (room, local) target). Chance = 0.25; BFS-oracle ceiling = 1.00.

Same model classes as everything else (forward(tokens) -> logits over vocab);
only the env and the loss mask differ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_hier_goal import HierGoalGridWorld
from mapformer.train_variant import VARIANT_MAP


def _run(model, tokens, act_mask, device):
    tokens = tokens.to(device); act_mask = act_mask.to(device)
    inp = tokens[:, :-1]; tgt = tokens[:, 1:]; mask = act_mask[:, :-1]
    logits = model(inp)
    lp = F.log_softmax(logits, dim=-1)
    return lp[mask], tgt[mask]


def train_hier_goal(model, env, epochs, lr, batch_size, T_explore, T_navigate,
                    n_batches, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, epochs * n_batches)
    rng = np.random.RandomState(0)
    losses = []
    for ep in range(epochs):
        ep_loss = 0.0; correct = 0; total = 0; nb = 0
        for _ in range(n_batches):
            toks, _, ams, _ = env.generate_hier_batch(batch_size, T_explore, T_navigate, rng)
            lp_flat, tgt_flat = _run(model, toks, ams, device)
            if tgt_flat.numel() == 0:
                continue
            loss = F.nll_loss(lp_flat, tgt_flat)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item(); nb += 1
            correct += (lp_flat.argmax(-1) == tgt_flat).sum().item(); total += tgt_flat.numel()
        losses.append(ep_loss / max(1, nb))
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}/{epochs} loss={losses[-1]:.4f} "
                  f"train_acc={correct/max(1,total):.3f} lr={sched.get_last_lr()[0]:.2e}")
    return losses


@torch.no_grad()
def evaluate(model, env, T_explore, T_navigate, n_trials, device, seed=0):
    rng = np.random.RandomState(seed); correct = total = 0; nll = 0.0
    for _ in range(n_trials):
        toks, _, ams, _ = env.generate_hier_batch(1, T_explore, T_navigate, rng)
        lp_flat, tgt_flat = _run(model, toks, ams, device)
        if tgt_flat.numel() == 0:
            continue
        correct += (lp_flat.argmax(-1) == tgt_flat).sum().item()
        total += tgt_flat.numel()
        nll += -lp_flat.gather(-1, tgt_flat.unsqueeze(-1)).sum().item()
    return correct / max(1, total), nll / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--room-size", type=int, default=8)
    ap.add_argument("--interleave-path", action="store_true",
                    help="deterministic balanced interleave of the BFS path -- kills the "
                         "copy-previous-action shortcut (0.969 -> 0.327)")
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--T-explore", type=int, default=64)
    ap.add_argument("--T-navigate", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-explore", type=int, nargs="+", default=None,
                    help="explore lengths to eval at (default: the training length). "
                         "Use e.g. 64 128 192 to test OOD explore length.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env = HierGoalGridWorld(size=64, room_size=args.room_size,
                            n_obs_types=args.n_obs_types, seed=args.seed,
                            interleave_path=args.interleave_path)
    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=64)
    print(f"{args.variant} seed={args.seed} vocab={env.unified_vocab_size} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"T_explore={args.T_explore} T_navigate={args.T_navigate}")

    losses = train_hier_goal(model, env, args.epochs, args.lr, args.batch_size,
                             args.T_explore, args.T_navigate, args.n_batches, args.device)

    env_test = HierGoalGridWorld(size=64, room_size=args.room_size,
                                 n_obs_types=args.n_obs_types, seed=10000,
                                 interleave_path=args.interleave_path)
    eval_lengths = args.eval_explore or [args.T_explore]
    results = {}
    for te in eval_lengths:
        acc, nll = evaluate(model, env_test, te, args.T_navigate,
                            n_trials=200, device=args.device, seed=2000)
        results[te] = {"acc": acc, "nll": nll}
        tag = "train" if te == args.T_explore else "OOD"
        print(f"HELD-OUT T_explore={te:4d} ({tag}): acc={acc:.3f} nll={nll:.3f} (chance=0.25)")

    ckpt = out / f"{args.variant}_hiergoal.pt"
    torch.save({"model_state": model.state_dict(), "variant": args.variant,
                "seed": args.seed, "results": results, "train_T_explore": args.T_explore,
                "losses": losses, "vocab_size": env.unified_vocab_size,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers}, ckpt)
    print(f"DONE {args.variant} -> {ckpt}")


if __name__ == "__main__":
    main()
