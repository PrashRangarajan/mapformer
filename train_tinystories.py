"""Word-level TinyStories LM (`tinystories_data.py`), for the step-table question on natural text. A word-level twin of
`train_hourglass_enwik8.py` (which is fixed to 256 bytes): any VARIANT_MAP model at vocab = len(vocab.json).

Differences from the enwik8 trainer, each deliberate: uint16 memmap data; the TRAINING stream is drawn from a dedicated
generator seeded by the run seed, so every arm at a seed sees the same batches; AdamW (wd 0.05, the project default)
with linear warmup + cosine to 0.1x (new work uses cosine, CLAUDE.md); validation on fixed batches (seed 1234) in nats
per token, plus a final larger validation; checkpoints always saved. grid_size = seq_len (as on enwik8 / code).

  python3 -m mapformer.train_tinystories --model Vanilla --seed 0 --device cuda:0 --out runs/tinystories/p0
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"
D = f"{REPO}/data/tinystories"


def batch(data, bs, T, gen, device):
    idx = torch.randint(0, len(data) - T - 1, (bs,), generator=gen).numpy()
    xy = np.stack([data[i:i + T + 1] for i in idx]).astype(np.int64)
    xy = torch.from_numpy(xy).to(device, non_blocking=True)
    return xy[:, :-1], xy[:, 1:]


@torch.no_grad()
def evaluate(model, data, bs, T, device, n_batches):
    model.eval()
    gen = torch.Generator().manual_seed(1234)            # same validation batches for every arm and checkpoint
    tot = 0.0
    for _ in range(n_batches):
        x, y = batch(data, bs, T, gen, device)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return tot / n_batches                                # nats per token


def build(name, vocab, dim, heads, n_layers, seq_len, rank):
    kw = dict(vocab_size=vocab, d_model=dim, n_heads=heads, n_layers=n_layers, grid_size=seq_len)
    return VARIANT_MAP[name](**kw, bottleneck_r=rank)      # RoPE takes **kwargs and ignores the rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vocab = len(json.load(open(f"{D}/vocab.json")))
    train = np.memmap(f"{D}/train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(f"{D}/val.bin", dtype=np.uint16, mode="r")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = build(a.model, vocab, a.dim, a.heads, a.n_layers, a.seq_len, a.rank).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(10_000 + a.seed)  # training stream: a function of the seed only
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / a.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / a.iters)))))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_s{a.seed}"
    cfg = dict(vars(a), vocab=vocab, params=n_params)
    print(f"{tag}: params {n_params:,} vocab {vocab} train tokens {len(train):,} val tokens {len(val):,}", flush=True)
    log = {"cfg": cfg, "curve": []}
    best, t0 = float("inf"), time.time()
    model.train()
    for it in range(1, a.iters + 1):
        x, y = batch(train, a.batch_size, a.seq_len, gen, a.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % a.eval_every == 0 or it == a.iters:
            v = evaluate(model, val, a.batch_size, a.seq_len, a.device, 40)
            wall = time.time() - t0
            log["curve"].append({"iter": it, "train": loss.item(), "val": v, "lr": sched.get_last_lr()[0], "wall": wall})
            print(f"  it {it:6d} train {loss.item():.4f} val {v:.4f} nats/token  ({it / wall:.2f} it/s)", flush=True)
            json.dump(log, open(out / f"{tag}.partial.json", "w"), indent=1)
            if v < best:
                best = v
                torch.save({"cfg": cfg, "iter": it, "val": v, "state_dict": model.state_dict()}, out / f"{tag}.best.pt")
    torch.save({"cfg": cfg, "iter": a.iters, "state_dict": model.state_dict()}, out / f"{tag}.final.pt")
    log["final_val_200"] = evaluate(model, val, a.batch_size, a.seq_len, a.device, 200)
    log["best_val"] = best
    log["wall"] = time.time() - t0
    json.dump(log, open(out / f"{tag}.json", "w"), indent=1)
    print(f"DONE {tag} final val (200 batches) {log['final_val_200']:.4f} best {best:.4f} wall {log['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
