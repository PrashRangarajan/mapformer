"""Train one arm on Dyck-2 (MapFormer v4 App. B.4) and evaluate the paper's L x D grid.

Paper recipe (stated): train L=32, D=4, 560,000 sequences, AdamW lr 1e-4, weight decay 0.01,
cosine schedule with warmup; 1-layer models have 1 head of size h=64, 2-layer models 2 heads of
size 64; eval L in {32,64,96,128} x D in {4,6,8,12}; metric F1 of Goodale et al.
Unstated and chosen here (see DYCK_PREREG.md): batch 128 (the paper's navigation batch), 5%
linear warmup then cosine to 10% of lr, dropout 0.1 (the models' default), omega base = 32
(the training context; the paper's only non-grid base is its context size), next-token
cross-entropy on every position.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_dyck import DyckWorld, f1_valid
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_baselines_extra import CoPEBaseline
from mapformer.model_pope import MapFormerWM_PoPE, MapFormerWM_RoPEIndex_PoPE
from mapformer.model_pope_t3 import (MapFormerWM_PoPE_T3, MapFormerWM_PoPE_T3_Inert,
                                     MapFormerWM_PoPE_T3_PI01)
from mapformer.model_pope_decay import (MapFormerWM_PoPE_Decay,
                                        MapFormerWM_RoPEIndex_PoPE_Decay,
                                        MapFormerWM_PoPE_Decay_IdxMetric,
                                        MapFormerWM_RoPEIndex_PoPE_Decay_StateMetric,
                                        MapFormerWM_RoPEIndex_PoPE_Decay_FrozenMetric)
from mapformer.model_dyck_monotone import (MapFormerWM_Abs, MapFormerWM_PoPE_Abs,
                                           MapFormerWM_PoPE_T3_PI01_Abs,
                                           MapFormerWM_PoPE_T3_Inert_Abs,
                                           MapFormerWM_PoPE_T3_Zero_Abs)

LS, DS = [32, 64, 96, 128], [4, 6, 8, 12]


def build(arch, vocab, n_layers, n_heads, rank, base, lam_init=None):
    d = 64 * n_heads
    kw = dict(vocab_size=vocab, d_model=d, n_heads=n_heads, n_layers=n_layers, grid_size=base)
    if arch == "MapWM": return MapFormerWM(bottleneck_r=rank, **kw)
    if arch == "MapEM": return MapFormerEM(bottleneck_r=rank, **kw)
    if arch == "RoPE": return MapFormerWM_RoPE(**kw)
    if arch == "CoPE": return CoPEBaseline(max_pos=max(LS) + 1, **kw)
    if arch == "PoPE": return MapFormerWM_RoPEIndex_PoPE(**kw)
    if arch == "MapPoPE": return MapFormerWM_PoPE(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_T3": return MapFormerWM_PoPE_T3(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_T3inert": return MapFormerWM_PoPE_T3_Inert(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_T3pi01": return MapFormerWM_PoPE_T3_PI01(bottleneck_r=rank, **kw)
    if lam_init is not None:
        kw["lam_init"] = lam_init
    if arch == "MapPoPE_decay": return MapFormerWM_PoPE_Decay(bottleneck_r=rank, **kw)
    if arch == "PoPE_decay": return MapFormerWM_RoPEIndex_PoPE_Decay(**kw)
    if arch == "MapPoPE_decay_idxmetric": return MapFormerWM_PoPE_Decay_IdxMetric(bottleneck_r=rank, **kw)
    if arch == "PoPE_decay_statemetric": return MapFormerWM_RoPEIndex_PoPE_Decay_StateMetric(bottleneck_r=rank, **kw)
    if arch == "PoPE_decay_frozenmetric": return MapFormerWM_RoPEIndex_PoPE_Decay_FrozenMetric(bottleneck_r=rank, **kw)
    if arch == "MapWM_abs": return MapFormerWM_Abs(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_abs": return MapFormerWM_PoPE_Abs(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_abs_T3": return MapFormerWM_PoPE_T3_PI01_Abs(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_abs_T3inert": return MapFormerWM_PoPE_T3_Inert_Abs(bottleneck_r=rank, **kw)
    if arch == "MapPoPE_abs_T3zero": return MapFormerWM_PoPE_T3_Zero_Abs(bottleneck_r=rank, **kw)
    raise ValueError(arch)


@torch.no_grad()
def evaluate(model, world, dev, n_per_cell, seed):
    model.eval()
    res = {}
    for L in LS:
        for D in DS:
            rng = np.random.default_rng(seed + 1000 * L + D)
            inp, tgt, valid, ent = world.batch(n_per_cell, L, D, rng)
            f1s, pvs, bts, ces = [], [], [], []
            for i in range(0, n_per_cell, 256):
                logits = model(inp[i:i + 256].to(dev)).float()
                p = logits.softmax(-1).cpu()
                f, pv, bt = f1_valid(p, valid[i:i + 256])
                f1s.append(f); pvs.append(pv); bts.append(bt)
                ces.append(F.cross_entropy(logits.transpose(1, 2), tgt[i:i + 256].to(dev),
                                           reduction="none").cpu())
            res[f"L{L}_D{D}"] = dict(F1=float(torch.cat(f1s).mean()), PV=float(torch.cat(pvs).mean()),
                                    BT=float(torch.cat(bts).mean()), CE=float(torch.cat(ces).mean()),
                                    CE_floor=float(ent.mean()))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["MapWM", "MapEM", "RoPE", "CoPE", "PoPE", "MapPoPE", "MapPoPE_T3", "MapPoPE_T3inert", "MapPoPE_T3pi01",
                         "MapPoPE_decay", "PoPE_decay", "MapWM_abs", "MapPoPE_abs", "MapPoPE_decay_idxmetric", "PoPE_decay_statemetric", "PoPE_decay_frozenmetric",
                         "MapPoPE_abs_T3", "MapPoPE_abs_T3inert", "MapPoPE_abs_T3zero"])
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lam-init", type=float, default=None,
                    help="decay arms: override the ALiBi geometric init with a constant, to match "
                         "the EFFECTIVE penalty of another arm at a given token distance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sequences", type=int, default=560_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    world = DyckWorld()
    model = build(a.arch, world.vocab_size, a.n_layers, a.n_heads, a.rank, a.base, a.lam_init).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    name = f"{a.arch}-{a.n_layers}L" + (f"_r{a.rank}" if a.arch.startswith(("MapWM", "MapEM", "MapPoPE")) else "")
    if a.lam_init is not None:
        name += f"_lam{a.lam_init:g}"
    print(f"{name} seed={a.seed} params={n_par:,} d_model={64 * a.n_heads}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.n_sequences // a.batch_size
    warm = max(1, int(a.warmup_frac * total))
    f = lambda s: ((s + 1) / warm if s < warm else
                   0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total - warm), 1.0))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)

    curve, run, t0 = [], [], time.time()
    for step in range(total):
        inp, tgt, _, _ = world.batch(a.batch_size, 32, 4, rng)
        logits = model(inp.to(dev))
        loss = F.cross_entropy(logits.transpose(1, 2), tgt.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if (step + 1) % 50 == 0:
            curve.append(float(np.mean(run))); run = []
        if (step + 1) % 500 == 0:
            print(f"step {step + 1}/{total} loss {curve[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    tail = curve[-max(2, len(curve) // 10):]
    slope = float(np.polyfit(np.arange(len(tail)) * 50, tail, 1)[0] * 1000)  # per 1k steps
    grid = evaluate(model, world, dev, a.n_eval, seed=424242)
    out = dict(arch=a.arch, name=name, n_layers=a.n_layers, n_heads=a.n_heads, rank=a.rank,
               base=a.base, seed=a.seed, params=n_par, steps=total, batch_size=a.batch_size,
               lr=a.lr, weight_decay=a.weight_decay, loss_curve=curve,
               final_loss=float(np.mean(tail)), final_slope_per_1k=slope, grid=grid,
               wall_s=time.time() - t0)
    od = Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), od / f"{name}.pt")
    json.dump(out, open(od / f"{name}.json", "w"), indent=1)
    g = grid
    print(f"final loss {out['final_loss']:.4f} (floor {g['L32_D4']['CE_floor']:.4f}) slope/1k {slope:+.4f}")
    print("F1 L32D4 %.3f  L128D4 %.3f  L32D12 %.3f  L128D12 %.3f" % tuple(
        g[k]["F1"] for k in ["L32_D4", "L128_D4", "L32_D12", "L128_D12"]), flush=True)


if __name__ == "__main__":
    main()
