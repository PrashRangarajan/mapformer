"""Train / evaluate a variant on the modular-clock navigation task.

Next-action CE at navigate positions (greedy-optimal tick toward the target
time). Chance 0.25, oracle 1.00. Symbolic analog of train_hier_goal, so the
same variants run unchanged; grid_size is set to the clock period P.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_clock import ModularClockWorld
from mapformer.train_variant import VARIANT_MAP


def _run(model, tokens, act_mask, device):
    tokens = tokens.to(device); act_mask = act_mask.to(device)
    inp = tokens[:, :-1]; tgt = tokens[:, 1:]; mask = act_mask[:, :-1]
    lp = F.log_softmax(model(inp), dim=-1)
    return lp[mask], tgt[mask]


def train_clock(model, env, epochs, lr, batch_size, T_explore, T_navigate, n_batches, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, epochs * n_batches)
    rng = np.random.RandomState(0); losses = []
    for ep in range(epochs):
        el = 0.0; c = 0; t = 0; nb = 0
        for _ in range(n_batches):
            toks, _, ams, _ = env.generate_clock_batch(batch_size, T_explore, T_navigate, rng)
            lp, tg = _run(model, toks, ams, device)
            if tg.numel() == 0:
                continue
            loss = F.nll_loss(lp, tg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            el += loss.item(); nb += 1
            c += (lp.argmax(-1) == tg).sum().item(); t += tg.numel()
        losses.append(el / max(1, nb))
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}/{epochs} loss={losses[-1]:.4f} train_acc={c/max(1,t):.3f}")
    return losses


@torch.no_grad()
def evaluate(model, env, T_explore, T_navigate, n_trials, device, seed=0):
    rng = np.random.RandomState(seed); c = 0; t = 0; nll = 0.0
    for _ in range(n_trials):
        toks, _, ams, _ = env.generate_clock_batch(1, T_explore, T_navigate, rng)
        lp, tg = _run(model, toks, ams, device)
        if tg.numel() == 0:
            continue
        c += (lp.argmax(-1) == tg).sum().item(); t += tg.numel()
        nll += -lp.gather(-1, tg.unsqueeze(-1)).sum().item()
    return c / max(1, t), nll / max(1, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radix-fast", type=int, default=16)
    ap.add_argument("--radix-slow", type=int, default=8)
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--T-explore", type=int, default=64)
    ap.add_argument("--T-navigate", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-explore", type=int, nargs="+", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = ModularClockWorld(radix_fast=args.radix_fast, radix_slow=args.radix_slow,
                            n_obs_types=args.n_obs_types, seed=args.seed)
    model = VARIANT_MAP[args.variant](vocab_size=env.unified_vocab_size, d_model=args.d_model,
                                      n_heads=args.n_heads, n_layers=args.n_layers, grid_size=env.P)
    print(f"{args.variant} seed={args.seed} P={env.P} vocab={env.unified_vocab_size} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    losses = train_clock(model, env, args.epochs, args.lr, args.batch_size,
                         args.T_explore, args.T_navigate, args.n_batches, args.device)
    env_test = ModularClockWorld(radix_fast=args.radix_fast, radix_slow=args.radix_slow,
                                 n_obs_types=args.n_obs_types, seed=10000)
    lens = args.eval_explore or [args.T_explore]; results = {}
    for te in lens:
        acc, nll = evaluate(model, env_test, te, args.T_navigate, 200, args.device, seed=2000)
        results[te] = {"acc": acc, "nll": nll}
        tag = "train" if te == args.T_explore else "OOD"
        print(f"HELD-OUT T_explore={te:4d} ({tag}): acc={acc:.3f} nll={nll:.3f} (chance=0.25)")
    ckpt = out / f"{args.variant}_clock.pt"
    torch.save({"model_state": model.state_dict(), "variant": args.variant, "seed": args.seed,
                "results": results, "train_T_explore": args.T_explore, "losses": losses,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers, "P": env.P}, ckpt)
    print(f"DONE {args.variant} -> {ckpt}")


if __name__ == "__main__":
    main()
