"""Multi-digit addition trainer (ADDITION_DESIGN.md). Loss on sum digits and the closing '$' only.

    python3 -m mapformer.train_addition --variant Vanilla_r4 --fmt role --output-dir runs/addition_pilot/X
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_addition import AdditionWorld
from mapformer.train_variant import VARIANT_MAP


def run_model(model, T, P, dev):
    x = T[:, :-1].to(dev)
    if getattr(model, "wants_pos_ids", False):
        return model(x, pos_ids=P[:, :-1].to(dev))
    return model(x)


def loss_fn(model, T, M, P, dev):
    logits = run_model(model, T, P, dev)
    tgt = T[:, 1:].to(dev); m = M[:, 1:].to(dev)
    return F.cross_entropy(logits[m], tgt[m])


@torch.no_grad()
def evaluate(model, env, n_digits, n_examples, bs, dev, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    exact = dig_ok = dig_n = 0; done = 0
    while done < n_examples:
        b = min(bs, n_examples - done)
        T, M, P, _ = env.batch(b, rng, n_digits=n_digits)
        pred = run_model(model, T, P, dev).argmax(-1).cpu()
        tgt = T[:, 1:]; m = M[:, 1:]
        hit = (pred == tgt) | ~m
        exact += int(hit.all(dim=1).sum())
        dig_ok += int(((pred == tgt) & m).sum()); dig_n += int(m.sum())
        done += b
    model.train()
    return exact / n_examples, dig_ok / dig_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--fmt", default="shared", choices=["shared", "role"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmax", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--eval-digits", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64])
    ap.add_argument("--eval-examples", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    env = AdditionWorld(a.fmt)
    model = VARIANT_MAP[a.variant](vocab_size=env.vocab_size, d_model=a.d_model, n_heads=a.n_heads,
                                   n_layers=a.n_layers, grid_size=64).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.variant} fmt={a.fmt} seed={a.seed} params={n_params:,} L{a.n_layers} H{a.n_heads} d{a.d_model} "
          f"dmax={a.dmax}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.n_batches; w = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / w if st < w else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((st - w) / max(1, total - w), 1.0))))
    rng = np.random.RandomState(a.seed)
    rand_start = bool(getattr(model, "wants_pos_ids", False))   # coupled oracles: random starting ID
    losses, curve = [], []
    for ep in range(a.epochs):
        t0 = time.time(); run = 0.0
        for _ in range(a.n_batches):
            T, M, P, _ = env.batch(a.batch_size, rng, dmax=a.dmax, random_start=rand_start)
            loss = loss_fn(model, T, M, P, dev)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        losses.append(run / a.n_batches)
        if (ep + 1) % 10 == 0 or ep == 0:
            ex, dg = evaluate(model, env, a.dmax, 256, 128, dev, seed=999)
            curve.append((ep + 1, losses[-1], ex, dg))
            print(f"  epoch {ep+1}/{a.epochs} loss={losses[-1]:.4f} exact@{a.dmax}={ex:.3f} digit={dg:.3f} "
                  f"({time.time()-t0:.1f}s/ep)", flush=True)
    res = {}
    for n in a.eval_digits:
        ex, dg = evaluate(model, env, n, a.eval_examples, 64 if n > 32 else 128, dev, seed=10000 + n)
        res[str(n)] = {"exact": ex, "digit": dg}
        print(f"  [eval] {n:3d} digits: exact {ex:.3f}  digit {dg:.3f}", flush=True)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a), vocab_size=env.vocab_size, n_params=n_params)
    json.dump({"config": cfg, "eval": res, "curve": curve, "final_loss": losses[-1]},
              open(out / f"{a.variant}_addition.json", "w"), indent=1)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.variant,
                "seed": a.seed, "config": cfg}, out / f"{a.variant}_addition.pt")
    print(f"DONE {a.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
