"""Trainer for Flip-Flop LM (Liu et al. 2023) -- the published external check.

Modelled on train_recency.py. Two task-specific decisions, both forced by the
gates in FLIPFLOP_GATES.md and neither of them a redefinition of the benchmark:

- TRAIN on every read (dense signal; this is the LM formulation).
- EVALUATE on the FINAL read only. Scoring all reads leaves an order-1 n-gram at
  0.70-0.75 against a chance of 0.500, because two consecutive reads with no write
  between them return the same bit. That is the published task's own property, so
  we report both readings rather than altering it; final-read scoring puts every
  n-gram order back at chance (0.44-0.54).

Chance is 0.500 here, not 0.0625, so every margin is read against a high floor.
Architecture follows CoPE sec 5.1 for this task: 4 layers, d=256, 4 heads.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_flipflop import FlipFlopWorld
from mapformer.train_variant import VARIANT_MAP

SPLITS = {"train": (0.10, 0.80, 0.10), "ood_dense": (0.01, 0.98, 0.01),
          "ood_sparse": (0.45, 0.10, 0.45)}


def _bits(logits, env, b, p):
    return logits[b, p, env.bit_offset:env.bit_offset + 2]


def _loss(model, env, toks, sps, ans, dev):
    toks = toks.to(dev)
    logits = model(toks[:, :-1])
    tl, tt = [], []
    for b in range(toks.shape[0]):
        for p, a in zip(sps[b], ans[b]):
            if p < logits.shape[1]:
                tl.append(_bits(logits, env, b, p)); tt.append(a - env.bit_offset)
    if not tl:
        return logits.sum() * 0.0
    return F.cross_entropy(torch.stack(tl),
                           torch.tensor(tt, device=dev, dtype=torch.long))


@torch.no_grad()
def evaluate(model, split, T, n_ep, dev, seed):
    """Final read only -- see the module docstring."""
    pw, pi, pr = SPLITS[split]
    env = FlipFlopWorld(pw, pi, pr, score_final_only=True, seed=10000)
    rng = np.random.RandomState(seed)
    model.eval(); ok = tot = 0
    for _ in range(n_ep):
        tok, sp, ans, _i = env.generate_episode(T, rng)
        if not sp:
            continue
        logits = model(tok.unsqueeze(0)[:, :-1].to(dev))
        for p, a in zip(sp, ans):
            if p < logits.shape[1]:
                ok += int(_bits(logits, env, 0, p).argmax().item() == a - env.bit_offset)
                tot += 1
    model.train()
    return ok / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fast-attn", action="store_true")
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    if a.fast_attn:
        import mapformer.model as _M
        _M.USE_SDPA = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    pw, pi, pr = SPLITS["train"]
    env = FlipFlopWorld(pw, pi, pr, seed=a.seed)

    model = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size,
                                   d_model=a.d_model, n_heads=a.n_heads,
                                   n_layers=a.n_layers, grid_size=a.grid_size).to(dev)
    print(f"{a.variant} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} T={a.T} chance=0.500", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.n_batches
    if a.schedule == "cosine":
        w = max(1, int(0.05 * total))
        f = lambda st: ((st + 1) / w if st < w else
                        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((st - w) / max(1, total - w), 1.0))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))

    rng = np.random.RandomState(a.seed); losses = []
    for ep in range(a.epochs):
        t0 = time.time(); run = 0.0
        for _ in range(a.n_batches):
            toks, sps, ans, _ = env.generate_batch(a.batch_size, a.T, rng)
            loss = _loss(model, env, toks, sps, ans, dev)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        losses.append(run / a.n_batches)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{a.epochs} loss={losses[-1]:.4f} "
                  f"(chance {math.log(2):.3f}) ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for split in SPLITS:
        acc, n = evaluate(model, split, a.T, 400, dev, seed=5000 + a.seed)
        results[split] = {"acc": acc, "err_pct": 100 * (1 - acc), "n": n}
        print(f"  [{split}] acc={acc:.4f}  err={100*(1-acc):.2f}%  (n={n})", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": a.variant, "seed": a.seed, "results": results,
                "vocab_size": env.unified_vocab_size, "d_model": a.d_model,
                "n_heads": a.n_heads, "n_layers": a.n_layers,
                "grid_size": a.grid_size}, out / f"{a.variant}_flipflop.pt")
    json.dump(results, open(out / f"{a.variant}_flipflop.json", "w"), indent=2)
    print(f"DONE {a.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
