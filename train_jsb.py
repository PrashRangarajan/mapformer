"""Bach Chorales sequence modelling: the PoPE paper's setup, crossed with path integration.

Paper (App B.2/B.3) for JSB: d_model 256, 8 heads, 6 layers, RMSNorm, base wavelength 10,000,
delta init range 2pi, dropout 0.2; batch 4, sequence length 2048, lr 6e-4 cosine to 6e-5, weight
decay 0.01, grad clip 1.0, AdamW beta2 0.99, 3,000 iterations, 10 warmup. Reported metric:
"Best NLL on the test split" -- RoPE 0.5081, PoPE 0.4889 (Table 2).
Deviations recorded in JSB_PREREG.md: LayerNorm rather than RMSNorm (the repo's layers); the
path-integration arms need an omega base, set to 2048 (the sequence length).
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_jsb import splits, VOCAB_SIZE, MAXLEN, PAD
from mapformer.model import MapFormerWM
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_pope import MapFormerWM_PoPE, MapFormerWM_RoPEIndex_PoPE, DELTA_MIN
from mapformer.model_pope_t3 import MapFormerWM_PoPE_T3, MapFormerWM_PoPE_T3_Inert
from mapformer.model_centered import center_model, token_frequencies

ARCH = {"RoPE": MapFormerWM_RoPE, "PoPE": MapFormerWM_RoPEIndex_PoPE,
        "MapWM": MapFormerWM, "MapPoPE": MapFormerWM_PoPE,
        "MapPoPE_T3": MapFormerWM_PoPE_T3, "MapPoPE_T3inert": MapFormerWM_PoPE_T3_Inert}


def build(arch, d_model, n_heads, n_layers, rank, base, dropout, delta_init, gen):
    kw = dict(vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
              dropout=dropout, grid_size=base)
    m = ARCH[arch](**kw) if arch in ("RoPE", "PoPE") else ARCH[arch](bottleneck_r=rank, **kw)
    if delta_init == "uniform":
        for mod in m.modules():
            if hasattr(mod, "pope_delta"):
                with torch.no_grad():
                    mod.pope_delta.uniform_(DELTA_MIN, 0.0, generator=gen)
    return m


BUCKETS = [(0, 512), (512, 1024), (1024, 2048)]


@torch.no_grad()
def nll_buckets(model, X, M, dev, bs=4):
    """NLL by POSITION bucket on full pieces -- isolates extrapolation past the training context."""
    model.eval(); tot = {b: 0.0 for b in BUCKETS}; cnt = {b: 0.0 for b in BUCKETS}
    for i in range(0, len(X), bs):
        x, m = X[i:i + bs].to(dev), M[i:i + bs].to(dev)
        l = F.cross_entropy(model(x)[:, :-1].transpose(1, 2), x[:, 1:], reduction="none") * m[:, 1:]
        for lo, hi in BUCKETS:
            sl = slice(max(lo - 1, 0), hi - 1)
            tot[(lo, hi)] += float(l[:, sl].sum()); cnt[(lo, hi)] += float(m[:, 1:][:, sl].sum())
    model.train()
    return {f"{lo}-{hi}": (tot[(lo, hi)] / cnt[(lo, hi)] if cnt[(lo, hi)] else float("nan"))
            for lo, hi in BUCKETS}


@torch.no_grad()
def nll(model, X, M, dev, bs=4):
    model.eval(); tot = n = 0.0
    for i in range(0, len(X), bs):
        x, m = X[i:i + bs].to(dev), M[i:i + bs].to(dev)
        lg = model(x)[:, :-1]
        tgt, mm = x[:, 1:], m[:, 1:]
        l = F.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        tot += float((l * mm).sum()); n += float(mm.sum())
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--base", type=int, default=MAXLEN)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--delta-init", default="uniform", choices=["uniform", "zero"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--center", action="store_true",
                    help="THEORY_MAPPOPE T1: centre the increment so the accumulator is a "
                         "mean-zero random walk instead of a clock")
    ap.add_argument("--train-len", type=int, default=MAXLEN,
                    help="training context; < 2048 trains on random crops and makes the "
                         "later position buckets an extrapolation test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
    S = splits()
    Xtr, Mtr = S["train"]; Xva, Mva = S["valid"]; Xte, Mte = S["test"]
    model = build(a.arch, a.d_model, a.n_heads, a.n_layers, a.rank, a.base,
                  a.dropout, a.delta_init, gen)
    name = a.arch + (f"_r{a.rank}" if a.arch.startswith(("MapWM", "MapPoPE")) else "")
    if a.center:
        model = center_model(model, token_frequencies(Xtr, Mtr, VOCAB_SIZE))
        name += "_centered"
    model = model.to(dev)
    print(f"{name} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.weight_decay)
    floor = a.min_lr / a.lr
    f = lambda s: ((s + 1) / a.warmup if s < a.warmup else
                   floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min((s - a.warmup) / max(1, a.iters - a.warmup), 1.0))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    rng = np.random.default_rng(a.seed)
    best_va, best_te, hist, t0 = 1e9, None, [], time.time()
    for step in range(a.iters):
        idx = rng.integers(0, len(Xtr), a.batch_size)
        x, m = Xtr[idx], Mtr[idx]
        if a.train_len < MAXLEN:                      # random crop, per example
            xs, ms = [], []
            for j in range(len(idx)):
                n = int(m[j].sum())
                st = int(rng.integers(0, max(1, n - a.train_len + 1)))
                xs.append(x[j, st:st + a.train_len]); ms.append(m[j, st:st + a.train_len])
            x = torch.stack([F.pad(t, (0, a.train_len - len(t))) for t in xs])
            m = torch.stack([F.pad(t, (0, a.train_len - len(t))) for t in ms])
        x, m = x.to(dev), m.to(dev)
        lg = model(x)[:, :-1]
        l = F.cross_entropy(lg.transpose(1, 2), x[:, 1:], reduction="none")
        loss = (l * m[:, 1:]).sum() / m[:, 1:].sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % a.eval_every == 0:
            va, te = nll(model, Xva, Mva, dev), nll(model, Xte, Mte, dev)
            hist.append(dict(step=step + 1, train=float(loss), valid=va, test=te))
            if va < best_va:
                best_va, best_te = va, te
            print(f"step {step+1}/{a.iters} train {float(loss):.4f} valid {va:.4f} test {te:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    od = Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), od / f"{name}.pt")
    buckets = nll_buckets(model, Xte, Mte, dev)
    print("test NLL by position bucket:", {k: round(v, 4) for k, v in buckets.items()}, flush=True)
    json.dump(dict(arch=a.arch, name=name, seed=a.seed, train_len=a.train_len,
                   base=a.base, rank=a.rank, dropout=a.dropout, centered=a.center,
                   test_buckets=buckets, best_valid=best_va, test_at_best_valid=best_te,
                   best_test=min(h["test"] for h in hist), final_test=hist[-1]["test"],
                   final_valid=hist[-1]["valid"], history=hist, wall_s=time.time() - t0,
                   params=sum(p.numel() for p in model.parameters())),
              open(od / f"{name}.json", "w"), indent=1)
    print(f"FINAL {name} seed {a.seed}: test at best valid {best_te:.4f}, best test "
          f"{min(h['test'] for h in hist):.4f}", flush=True)


if __name__ == "__main__":
    main()
