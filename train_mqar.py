"""Trainer for MQAR. Scored at query positions over the VALUE block, so chance is
1/n_values. Pre-registered as a probable ceiling: see MQAR_PREREG.md."""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_mqar import MQARWorld
from mapformer.train_variant import VARIANT_MAP


def _vals(logits, env, b, p):
    return logits[b, p, env.val_offset:env.val_offset + env.n_values]


def _loss(model, env, toks, sps, ans, dev):
    toks = toks.to(dev); logits = model(toks[:, :-1])
    tl, tt = [], []
    for b in range(toks.shape[0]):
        for p, a in zip(sps[b], ans[b]):
            if p < logits.shape[1]:
                tl.append(_vals(logits, env, b, p)); tt.append(a - env.val_offset)
    if not tl:
        return logits.sum() * 0.0
    return F.cross_entropy(torch.stack(tl), torch.tensor(tt, device=dev, dtype=torch.long))


@torch.no_grad()
def evaluate(model, env, T, n_ep, dev, seed):
    rng = np.random.RandomState(seed); model.eval(); ok = tot = 0
    for _ in range(n_ep):
        tok, sp, ans, _i = env.generate_episode(T, rng)
        logits = model(tok.unsqueeze(0)[:, :-1].to(dev))
        for p, a in zip(sp, ans):
            if p < logits.shape[1]:
                ok += int(_vals(logits, env, 0, p).argmax().item() == a - env.val_offset); tot += 1
    model.train()
    return ok / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--eval-T", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--n-kv", type=int, default=16)
    ap.add_argument("--n-queries", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    env = MQARWorld(n_kv=a.n_kv, n_queries=a.n_queries, seed=a.seed)
    model = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size, d_model=a.d_model,
                                   n_heads=a.n_heads, n_layers=a.n_layers,
                                   grid_size=a.grid_size).to(dev)
    print(f"{a.variant} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} n_kv={a.n_kv} chance={1/env.n_values:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.n_batches; w = max(1, int(0.05 * total))
    f = lambda st: ((st + 1) / w if st < w else
                    0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((st - w) / max(1, total - w), 1.0))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
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
            print(f"  epoch {ep+1}/{a.epochs} loss={losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for T in a.eval_T:
        acc, n = evaluate(model, env, T, 200, dev, seed=5000 + a.seed)
        results[str(T)] = {"acc": acc, "n": n}
        print(f"  [held-out] T={T}: acc={acc:.4f} (n={n})", flush=True)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses, "variant": a.variant,
                "seed": a.seed, "results": results, "vocab_size": env.unified_vocab_size,
                "d_model": a.d_model, "n_heads": a.n_heads, "n_layers": a.n_layers,
                "grid_size": a.grid_size}, out / f"{a.variant}_mqar.pt")
    json.dump(results, open(out / f"{a.variant}_mqar.json", "w"), indent=2)
    print(f"DONE {a.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
