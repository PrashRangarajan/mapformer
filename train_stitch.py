"""Trainer for the stitch (transitive-inference) environment.

Objective is MapFormer's own and nothing else: cross-entropy on the next token
at REVISITED observation positions. There is no invented scored test phase and
no auxiliary loss, because the evaluation for this task is an inspection of the
trained model's attention (`probe_stitch_attention.py`), not a held-out
accuracy.

Why an inspection rather than an accuracy: CSCG (George et al.) reports this
experiment qualitatively -- "Predictive performance on the stitching of the two
rooms is perfect" -- with no accuracy table, no baseline and no chance rate.
Their evaluation is to look at the learned transition matrix. There is no scored
metric of theirs to reproduce, and the scored one written for this repo earlier
had a negative control defeatable at 0.617 balanced accuracy, so it was dropped
rather than patched.

Two arms, and the second is the load-bearing one:
  MapWM-Flat  path integration (theta = omega * cumsum(Delta))
  PlainFlat   ordinary sequence-index RoPE, no path integration

PlainFlat is the control that absorbs every non-positional explanation of the
probe result (recency, visit frequency, token identity), because at layer 0 the
two look-alike patches have IDENTICAL key vectors up to their rotation, and
PlainFlat's rotation carries sequence index rather than place.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_stitch import StitchWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


@torch.no_grad()
def evaluate(model, env, T_a, T_b, T_t, n_batches, batch_size, device, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    ok = tot = 0
    nll = 0.0
    for _ in range(n_batches):
        toks, rev = env.generate_train_batch(batch_size, T_a, T_b, T_t, rng)
        toks = toks.to(device)
        logits = model(toks[:, :-1])
        tgt = toks[:, 1:]
        m = rev.to(device)[:, 1:]
        if not m.any():
            continue
        sl, tl = logits[m], tgt[m]
        ok += int((sl.argmax(-1) == tl).sum().item())
        nll += float(F.cross_entropy(sl, tl, reduction="sum").item())
        tot += int(tl.numel())
    model.train()
    return ok / max(tot, 1), nll / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=6)
    ap.add_argument("--n-obs", type=int, default=15)
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument("--T-a", type=int, default=256)
    ap.add_argument("--T-b", type=int, default=256)
    ap.add_argument("--T-t", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    kw = dict(h=args.h, w=args.w, n_obs_types=args.n_obs, patch=args.patch)
    env = StitchWorld(seed=args.seed, **kw)
    # Layouts are redrawn every episode from `rng`, so a "held-out" env differs
    # only in its evaluation RNG stream. Stated so the number is not read as a
    # transfer result: it is in-distribution held-out sampling.
    env_test = StitchWorld(seed=10000, **kw)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        grid_size=max(2 * args.h - args.patch, 2 * args.w - args.patch)).to(dev)
    print(f"{args.variant} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} chance={1/args.n_obs:.4f} "
          f"L={2*(args.T_a+args.T_b+args.T_t)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        t0 = time.time()
        run = 0.0
        for _ in range(args.n_batches):
            toks, rev = env.generate_train_batch(
                args.batch_size, args.T_a, args.T_b, args.T_t, rng)
            toks = toks.to(dev)
            logits = model(toks[:, :-1])
            tgt = toks[:, 1:]
            m = rev.to(dev)[:, 1:]
            loss = F.cross_entropy(logits[m], tgt[m])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        losses.append(run / args.n_batches)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs} loss={losses[-1]:.4f} "
                  f"(chance {np.log(args.n_obs):.3f}) ({time.time()-t0:.0f}s)",
                  flush=True)

    acc, nll, n = evaluate(model, env_test, args.T_a, args.T_b, args.T_t,
                           8, 8, dev, seed=5000 + args.seed)
    print(f"  [held-out] revisit acc={acc:.4f} nll={nll:.4f} (n={n}, "
          f"chance={1/args.n_obs:.4f})", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed,
                "results": {"revisit_acc": acc, "revisit_nll": nll, "n": n},
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "env": kw},
               out / f"{args.variant}_stitch.pt")
    json.dump({"revisit_acc": acc, "revisit_nll": nll, "n": n,
               "final_loss": losses[-1]},
              open(out / f"{args.variant}_stitch.json", "w"), indent=2)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
