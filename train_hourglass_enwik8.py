"""
enwik8 char-level LM training for the Hourglass SCAFFOLD SANITY-CHECK.

Adapted from lucidrains/hourglass-transformer-pytorch train.py. Trains a plain
(non-MapFormer) Hourglass and flat baselines so we can confirm the scaffold
reproduces Hourglass's published efficiency property (equal/better bpc at less
compute) before swapping in MapFormer layers.

Reports validation bits-per-character (bpc) vs iteration and wall-clock, plus
an attention-FLOP proxy, so the efficiency comparison is legible:
  Hourglass(4,2,4) and Flat-10 have IDENTICAL params; Hourglass uses ~15%
  fewer attention FLOPs (valley runs at seq/2). If Hourglass matches Flat-10
  bpc at equal iterations, that is a strict efficiency win.

Usage:
  python3 -m mapformer.train_hourglass_enwik8 --model hourglass --device cuda:0
  python3 -m mapformer.train_hourglass_enwik8 --model flat10    --device cuda:1
  python3 -m mapformer.train_hourglass_enwik8 --model flat9     --device cuda:0
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .hourglass_plain import HourglassPlainLM, FlatPlainLM
from .train_variant import VARIANT_MAP

LN2 = 0.6931471805599453
# Repo root = the package dir this file lives in. Makes default paths portable
# across servers (no hardcoded absolute paths).
_REPO = os.path.dirname(os.path.abspath(__file__))


def get_batch(data, batch_size, seq_len, device, generator=None):
    n = data.size(0)
    idx = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + seq_len] for i in idx]).long().to(device)
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in idx]).long().to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, batch_size, seq_len, device, n_batches=40):
    """Deterministic validation.

    The val batches were previously drawn from the GLOBAL torch RNG, so they were
    re-sampled every checkpoint AND differed between models (each model's __init__
    consumes a different number of draws). Measured consequence: within-model
    checkpoint-to-checkpoint val swings of 0.02-0.07 bpc against a between-model
    spread of 0.011 -- and taking min-over-last-6 instead of the final point
    REORDERED the arms. The comparison could not resolve its own effect.

    Fixed here with a dedicated generator seeded identically for every model and
    every checkpoint, so all arms are scored on the SAME val batches. This also
    stops evaluation from perturbing the training data stream.
    """
    model.eval()
    gen = torch.Generator().manual_seed(1234)      # same val set for every arm
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, seq_len, device, generator=gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += loss.item()
    model.train()
    return (tot / n_batches) / LN2  # bpc


def build(model_name, shorten=2, dim=512, heads=8, n_layers=9, grid_size=512,
          bottleneck_r=2):
    if model_name == "hourglass":
        return HourglassPlainLM(num_tokens=256, dim=512, depth=(4, 2, 4),
                                shorten_factor=shorten, heads=8)
    if model_name == "flat10":
        return FlatPlainLM(num_tokens=256, dim=512, n_layers=10, heads=8)
    if model_name == "flat9":
        return FlatPlainLM(num_tokens=256, dim=512, n_layers=9, heads=8)
    # Any registered MapFormer variant, on byte-level enwik8 (vocab 256).
    # grid_size: MapFormer initialises omega geometrically so the LOWEST frequency
    # completes one cycle over the largest traversable distance (omega_min =
    # 2*pi/grid_size). Language has no grid, so we set grid_size = seq_len, the
    # direct analogue: the slowest rotation spans exactly one context window.
    # (Selective RoPE instead DERIVES its ladder from random-Fourier-feature theory
    # -- the untested alternative; see LANGUAGE_LANDSCAPE.md.)
    if model_name in VARIANT_MAP:
        kw = dict(vocab_size=256, d_model=dim, n_heads=heads,
                  n_layers=n_layers, grid_size=grid_size)
        # bottleneck_r = dimensionality of the learned "action" subspace. MapFormer
        # uses r=2 for 2D navigation and r=4 for its OpenWebText run; rank is
        # decisive in their own data (2D: r=1 -> 0.66, r=2 -> 1.00). Only the
        # path-integrating variants accept it.
        import inspect
        if 'bottleneck_r' in inspect.signature(VARIANT_MAP[model_name].__init__).parameters:
            kw['bottleneck_r'] = bottleneck_r
        return VARIANT_MAP[model_name](**kw)
    raise ValueError(model_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)   # plain names or any VARIANT_MAP key
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--data", default=os.path.join(_REPO, "data", "enwik8"))
    ap.add_argument("--out", default=os.path.join(_REPO, "hourglass_enwik8"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shorten", type=int, default=2)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=9)
    ap.add_argument("--tag", default="")   # output filename suffix
    ap.add_argument("--bottleneck-r", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw = np.fromfile(args.data, dtype=np.uint8)
    data = torch.from_numpy(raw.copy())
    train_data = data[:90_000_000]
    val_data = data[90_000_000:95_000_000]

    device = args.device
    model = build(args.model, shorten=args.shorten, dim=args.dim,
                  heads=args.heads, n_layers=args.n_layers,
                  grid_size=args.seq_len, bottleneck_r=args.bottleneck_r).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    flop = (model.flops_proxy(args.seq_len) / (args.seq_len ** 2)
            if hasattr(model, 'flops_proxy') else None)
    print(f"model={args.model} params={n_params:,} attn_flop_proxy={flop} "
          f"seq={args.seq_len} bs={args.batch_size} iters={args.iters}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    log = {"model": args.model, "params": n_params, "flop_proxy": flop,
           "seq_len": args.seq_len, "batch_size": args.batch_size,
           "curve": []}

    t0 = time.time()
    model.train()
    for it in range(1, args.iters + 1):
        x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            val_bpc = evaluate(model, val_data, args.batch_size, args.seq_len, device)
            wall = time.time() - t0
            train_bpc = loss.item() / LN2
            log["curve"].append({"iter": it, "val_bpc": val_bpc,
                                 "train_bpc": train_bpc, "wall_s": wall})
            print(f"  it={it:6d} train_bpc={train_bpc:.4f} val_bpc={val_bpc:.4f} "
                  f"wall={wall:.0f}s ({it/wall:.1f} it/s)")
            with open(outdir / f"{args.model}{args.tag}.partial.json", "w") as f:
                json.dump(log, f, indent=2)

    log["wall_total_s"] = time.time() - t0
    with open(outdir / f"{args.model}{args.tag}.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"DONE {args.model} val_bpc={log['curve'][-1]['val_bpc']:.4f} "
          f"wall={log['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
