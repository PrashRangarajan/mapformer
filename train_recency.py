"""Trainer for the Recency (k-back) task -- the CLOCK half of the clock/map 2x2.

  L = CE at each scored query step, over the SYMBOL block of the logits only.

There is deliberately NO auxiliary next-token loss, and that is a design decision
rather than an omission. `train_match_query.py` carries `L_obs` because without
it nothing pressures the model to build the map the query phase reads. Here the
symbols are drawn IID UNIFORM, so next-symbol prediction has irreducible entropy
log(n_symbols) and its gradient is pure noise -- adding it would spend capacity
on an unlearnable objective and dilute the only signal that exists. The query
loss is the whole loss.

Consequence: supervision is sparse (~20 scored positions per 256-token episode,
the price of the `min_gap = k_max` shortcut guarantee), so this needs batches,
not a short budget.

`--schedule cosine` is the DEFAULT here, unlike the older trainers. Standing rule
10: `LinearLR` from step one decays with no warmup and can trap a run on a
plateau it then cannot escape -- it moved one arm from 0.448 to 0.990 on the same
task and inverted a headline. There are no legacy checkpoints for this task, so
there is nothing to stay bit-compatible with and no reason to inherit the trap.

Per-offset accuracy is reported because it is the mechanism readout, not a
nicety. A monotone accumulator addresses every k by the same mechanism and should
be roughly FLAT in k; a signed one collides more the further back it reaches and
should DECAY in k. That shape distinguishes "the arm is worse" from "the arm is
worse for the predicted reason".

Gates: validate_recency.py, PASS at the default `min_gap = k_max` for
T = 256..2048 (`RECENCY_GATES.md`).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_recency import RecencyWorld
from mapformer.train_variant import VARIANT_MAP


def _sym_logits(logits, env, b, p):
    """Logit slice over the symbol block. Keeps chance at 1/n_symbols rather
    than 1/vocab, since the answer is a symbol by construction."""
    lo = env.sym_offset
    return logits[b, p, lo:lo + env.n_symbols]


def _loss(model, env, toks, sps, ans, device):
    toks = toks.to(device)
    logits = model(toks[:, :-1])
    tl, tt = [], []
    for b in range(toks.shape[0]):
        for p, a in zip(sps[b], ans[b]):
            if p >= logits.shape[1]:
                continue
            tl.append(_sym_logits(logits, env, b, p))
            tt.append(a - env.sym_offset)
    if not tl:
        return logits.sum() * 0.0
    return F.cross_entropy(torch.stack(tl),
                           torch.tensor(tt, device=device, dtype=torch.long))


@torch.no_grad()
def evaluate(model, env, T, n_batches, batch_size, device, seed):
    """Held-out here means a fresh RNG stream, NOT a fresh environment: the task
    has no map to hold out, only sequences. Generalisation is to unseen streams,
    and the OOD axis is T."""
    model.eval()
    rng = np.random.RandomState(seed)
    ok = tot = 0
    nll = 0.0
    per_k_ok = np.zeros(env.k_max + 1); per_k_n = np.zeros(env.k_max + 1)
    for _ in range(n_batches):
        toks, sps, ans, infos = env.generate_batch(batch_size, T, rng)
        logits = model(toks[:, :-1].to(device))
        for b in range(toks.shape[0]):
            for p, a, k in zip(sps[b], ans[b], infos[b]["offsets"]):
                if p >= logits.shape[1]:
                    continue
                sl = _sym_logits(logits, env, b, p)
                t = a - env.sym_offset
                hit = int(sl.argmax().item() == t)
                ok += hit; tot += 1
                nll += -F.log_softmax(sl.float(), dim=-1)[t].item()
                per_k_ok[k] += hit; per_k_n[k] += 1
    model.train()
    per_k = {int(k): (per_k_ok[k] / per_k_n[k]) if per_k_n[k] else None
             for k in range(1, env.k_max + 1)}
    return ok / max(tot, 1), nll / max(tot, 1), tot, per_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-symbols", type=int, default=16)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--p-query", type=float, default=0.25)
    ap.add_argument("--min-gap", type=int, default=None,
                    help="default None = k_max, the value that makes the "
                         "answer-stream shortcut structurally impossible "
                         "(k2 == k1 + g needs k2 > k_max). See RECENCY_GATES.md; "
                         "lowering it re-opens a gate that FAILS at 0.147.")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--eval-T", nargs="+", type=int, default=[256, 512, 1024])
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid-size", type=int, default=64,
                    help="not spatial here -- it sets the PathIntegrator's omega "
                         "frequency range. Shared by every arm in a batch, so it "
                         "is a constant of the comparison and not a confound.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fast-attn", action="store_true",
                    help="SDPA + TF32; verified equivalent (logits 1.4e-06, grad "
                         "cosine 1.0000000000), 2.56x faster. NOT valid for MapEM. "
                         "Every arm in a batch must share the setting.")
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"],
                    help="cosine (DEFAULT) = 5%% warmup + cosine to 10%%. See the "
                         "module docstring for why linear is not the default here.")
    ap.add_argument("--k-fixed", type=int, default=None,
                    help="every query asks this k (train AND eval). SEARCH_PREREG.md S3.")
    ap.add_argument("--k-curriculum", default=None,
                    help="'K0,EVERY': draw k from 1..K0*2^(epoch//EVERY), capped at "
                         "k_max. Eval always uses the full 1..k_max. SEARCH_PREREG.md S3.")
    ap.add_argument("--k-set", default=None,
                    help="comma-separated offsets to draw k from, e.g. '1,4,16,64' (train AND "
                         "eval). Varies the number of query tokens at fixed k_max. SPREAD_PREREG.md")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if args.fast_attn:
        import mapformer.model as _M
        _M.USE_SDPA = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[fast-attn] SDPA + TF32 enabled", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    kw = dict(n_symbols=args.n_symbols, k_max=args.k_max,
              p_query=args.p_query, min_gap=args.min_gap, k_fixed=args.k_fixed,
              k_set=[int(v) for v in args.k_set.split(",")] if args.k_set else None)
    curric = tuple(int(v) for v in args.k_curriculum.split(",")) if args.k_curriculum else None
    env = RecencyWorld(seed=args.seed, **kw)
    env_test = RecencyWorld(seed=10000, **kw)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        grid_size=args.grid_size).to(dev)
    chance = 1.0 / env.n_symbols
    mr_floor = 1.0 / env.k_max + chance * (1.0 - 1.0 / env.k_max)
    print(f"{args.variant} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} T={args.T} k_max={env.k_max} "
          f"min_gap={env.min_gap} chance={chance:.4f} "
          f"most-recent floor={mr_floor:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    if args.schedule == "cosine":
        w = max(1, int(0.05 * total))
        def f(st):
            if st < w:
                return (st + 1) / w
            p_ = (st - w) / max(1, total - w)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p_, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: max(0.0, 1 - s / total))

    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        t0 = time.time(); run = 0.0
        if curric is not None:
            env.k_active = min(env.k_max, curric[0] * 2 ** (ep // curric[1]))
        for _ in range(args.n_batches):
            toks, sps, ans, _ = env.generate_batch(args.batch_size, args.T, rng)
            loss = _loss(model, env, toks, sps, ans, dev)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
        losses.append(run / args.n_batches)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs} loss={losses[-1]:.4f} "
                  f"(chance {math.log(env.n_symbols):.2f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for T in args.eval_T:
        a, nll, n, per_k = evaluate(model, env_test, T, 8, 8, dev,
                                    seed=5000 + args.seed)
        results[str(T)] = {"acc": a, "nll": nll, "n": n, "per_k": per_k}
        pk = " ".join(f"k{k}={v:.2f}" for k, v in per_k.items() if v is not None)
        print(f"  [held-out] T={T}: acc={a:.4f} nll={nll:.4f} (n={n})\n"
              f"             per-offset: {pk}", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed, "results": results,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "grid_size": args.grid_size, "k_max": args.k_max,
                "n_symbols": args.n_symbols, "min_gap": env.min_gap,
                "k_fixed": args.k_fixed, "k_curriculum": args.k_curriculum,
                "k_set": args.k_set},
               out / f"{args.variant}_recency.pt")
    json.dump(results, open(out / f"{args.variant}_recency.json", "w"), indent=2)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
