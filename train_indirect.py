"""Indirect Indexing (PoPE paper sec 5.1) with path integration added as a second factor.

The PoPE paper compares RoPE vs PoPE and reports 11.16 +/- 2.45 vs 94.82 +/- 2.91 final-token
accuracy (Table 1). This runs the 2x2 that the paper does not: {index position, path integration}
x {RoPE encoding, PoPE encoding}.

Their recipe (App B.2/B.3), followed: d_model 512, 8 heads, 8 layers, dropout 0, base wavelength
10,000, delta init range 2pi; batch 64, lr 2e-4 cosine to 2e-5, weight decay 0.01, grad clip 1.0,
AdamW beta2 0.99, 100,000 iterations, 4,000 warmup; fixed train/val/test splits of 1M/10k/10k;
cross-entropy on the final (target) token only.
Deviations, all recorded in INDIRECT_PREREG.md: LayerNorm instead of RMSNorm (the repo's layers),
block size 56 rather than their stated 40 (their own strings reach 40 characters before the
", c, -15, " suffix), and the path-integration arms need an omega base, set to the block size.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_indirect import IndirectWorld, VOCAB_SIZE, BLOCK
from mapformer.model import MapFormerWM
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_pope import MapFormerWM_PoPE, MapFormerWM_RoPEIndex_PoPE, DELTA_MIN

ARCH = {"RoPE": MapFormerWM_RoPE, "PoPE": MapFormerWM_RoPEIndex_PoPE,
        "MapWM": MapFormerWM, "MapPoPE": MapFormerWM_PoPE}


def build(arch, d_model, n_heads, n_layers, rank, base, delta_init, gen):
    kw = dict(vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
              dropout=0.0, grid_size=base)
    m = ARCH[arch](**kw) if arch in ("RoPE", "PoPE") else ARCH[arch](bottleneck_r=rank, **kw)
    if delta_init == "uniform":       # paper Table 7: "Init. range for delta: 2pi"
        for mod in m.modules():
            if hasattr(mod, "pope_delta"):
                with torch.no_grad():
                    mod.pope_delta.uniform_(DELTA_MIN, 0.0, generator=gen)
    return m


@torch.no_grad()
def evaluate(model, X, Y, dev, bs=256):
    model.eval(); ok = 0
    for i in range(0, len(X), bs):
        lg = model(X[i:i + bs].to(dev))[:, -1]
        ok += int((lg.argmax(-1).cpu() == Y[i:i + bs]).sum())
    model.train()
    return ok / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--base", type=int, default=BLOCK)
    ap.add_argument("--delta-init", default="uniform", choices=["uniform", "zero"])
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    w = IndirectWorld()
    Xtr, Ytr = w.batch(a.n_train, np.random.default_rng(1234))          # fixed splits, as in the paper
    Xva, Yva = w.batch(10_000, np.random.default_rng(5678))
    Xte, Yte = w.batch(10_000, np.random.default_rng(9012))
    model = build(a.arch, a.d_model, a.n_heads, a.n_layers, a.rank, a.base, a.delta_init, gen).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    name = a.arch + (f"_r{a.rank}" if a.arch in ("MapWM", "MapPoPE") else "")
    print(f"{name} seed={a.seed} params={n_par:,} train={len(Xtr):,} chance={1/52:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.weight_decay)
    floor = a.min_lr / a.lr
    f = lambda s: ((s + 1) / a.warmup if s < a.warmup else
                   floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min((s - a.warmup) / max(1, a.iters - a.warmup), 1.0))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    rng = np.random.default_rng(a.seed)
    curve, run, t0 = [], [], time.time()
    for step in range(a.iters):
        idx = rng.integers(0, len(Xtr), a.batch_size)
        lg = model(Xtr[idx].to(dev))[:, -1]
        loss = F.cross_entropy(lg, Ytr[idx].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if (step + 1) % 500 == 0:
            curve.append(float(np.mean(run))); run = []
        if (step + 1) % a.eval_every == 0:
            va = evaluate(model, Xva[:2000], Yva[:2000], dev)
            print(f"step {step+1}/{a.iters} loss {(curve[-1] if curve else float('nan')):.4f} val {va:.4f} ({time.time()-t0:.0f}s)", flush=True)
    test = evaluate(model, Xte, Yte, dev)
    val = evaluate(model, Xva, Yva, dev)
    od = Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), od / f"{name}.pt")
    json.dump(dict(arch=a.arch, name=name, seed=a.seed, params=n_par, iters=a.iters,
                   test_acc=test, val_acc=val, final_loss=float(np.mean(curve[-max(2, len(curve)//10):])),
                   loss_curve=curve, wall_s=time.time() - t0, base=a.base, rank=a.rank,
                   delta_init=a.delta_init), open(od / f"{name}.json", "w"), indent=1)
    print(f"FINAL {name} seed {a.seed}: test {test:.4f} val {val:.4f}", flush=True)


if __name__ == "__main__":
    main()
