"""Train one arm on an algorithmic task and evaluate LENGTH GENERALIZATION.

Train short, test long -- the evaluation the looped-transformer literature actually
uses (arXiv 2409.15647). Eval lengths are runtime arguments, so extrapolation costs
no retraining.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_algorithmic import WORLDS
from mapformer.train_variant import VARIANT_MAP


def make_env(task, n_symbols, seed):
    W = WORLDS[task]
    return W(seed=seed) if task == "parity" else W(n_symbols, seed=seed)


@torch.no_grad()
def evaluate(model, env, L, n_batches, bs, dev, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    ok = tot = 0
    for _ in range(n_batches):
        t, y, m = env.batch(bs, L, rng)
        logits = model(t.to(dev))
        pred = logits.argmax(-1).cpu()
        ok += int((pred[m] == y[m]).sum()); tot += int(m.sum())
    model.train()
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--task", default="parity", choices=list(WORLDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-length", type=int, default=16)
    ap.add_argument("--eval-lengths", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-symbols", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    env = make_env(a.task, a.n_symbols, seed=a.seed)
    env_test = make_env(a.task, a.n_symbols, seed=99991)   # held out generator

    model = VARIANT_MAP[a.variant](vocab_size=env.vocab_size, d_model=a.d_model,
                                   n_heads=a.n_heads, n_layers=a.n_layers,
                                   grid_size=64).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{a.variant} task={a.task} seed={a.seed} params={n_par:,} "
          f"vocab={env.vocab_size} L_train={a.train_length} chance={env.chance:.4f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.n_batches
    if a.schedule == "cosine":
        warm = max(1, int(0.05 * total))
        f = lambda s: ((s + 1) / warm if s < warm else
                       0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi *
                                                       min((s - warm) / max(1, total - warm), 1.0))))
    else:
        f = lambda s: max(0.0, 1 - s / total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)

    rng = np.random.RandomState(a.seed)
    losses = []
    for ep in range(a.epochs):
        t0 = time.time(); acc = 0.0
        for _ in range(a.n_batches):
            t, y, m = env.batch(a.batch_size, a.train_length, rng)
            logits = model(t.to(dev))
            loss = F.cross_entropy(logits[m.to(dev)], y.to(dev)[m.to(dev)])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); acc += loss.item()
        losses.append(acc / a.n_batches)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{a.epochs} loss={losses[-1]:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    res = {}
    for L in a.eval_lengths:
        res[L] = evaluate(model, env_test, L, 8, 64, dev, seed=5000 + a.seed)
        print(f"  [held-out] L={L}: acc={res[L]:.4f}", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "vocab_size": env.vocab_size,
                "d_model": a.d_model, "n_heads": a.n_heads, "n_layers": a.n_layers,
                "task": a.task, "params": n_par},
               out / f"{a.variant}_{a.task}.pt")
    json.dump({"acc": res, "final_loss": losses[-1], "params": n_par,
               "chance": env.chance}, open(out / f"{a.variant}_{a.task}.json", "w"),
              indent=2)
    print(f"DONE {a.variant} {a.task} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
