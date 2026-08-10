"""Trainer for the Match-Query (blind continuation) task.

  L = L_match + obs_coef * L_obs

  L_match  CE at each scored query step over the NON-BLANK observation types
           only (answers are non-blank by construction, so restricting the logit
           slice keeps chance at 1/n_obs_types = 0.0625 rather than the blank
           majority rate).
  L_obs    the paper's objective on the explore phase: predict the observation at
           REVISITED cells. Without it there is no pressure to build the map that
           the query phase then reads.

obs_coef defaults to 1.0 (equal weight), stated rather than tuned.

Gates for this task are in validate_match_query.py and PASS at T_explore>=256
with dedup. The 25-epoch budget that undertrained Map-Query is NOT reused here:
default is 200, the budget the Map-Query diagnostic showed was actually needed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_match_query import MatchQueryGridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


def _match_logits(logits, env, b, p):
    """Logit slice over the non-blank observation types."""
    lo = env.obs_offset
    return logits[b, p, lo:lo + env.n_obs_types]


def _losses(model, env, toks, rev, sps, ans, device):
    toks = toks.to(device); rev = rev.to(device)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    logits = model(inp)
    ml = []
    for b in range(toks.shape[0]):
        for p, a in zip(sps[b], ans[b]):
            if p >= logits.shape[1]:
                continue
            t = a - env.obs_offset            # index within the non-blank block
            ml.append(F.cross_entropy(_match_logits(logits, env, b, p).unsqueeze(0),
                                      torch.tensor([t], device=device)))
    L_match = torch.stack(ml).mean() if ml else logits.sum() * 0.0
    m = rev[:, 1:]
    L_obs = F.cross_entropy(logits[m], tgt[m]) if m.any() else logits.sum() * 0.0
    return L_match, L_obs


@torch.no_grad()
def evaluate(model, env, T_explore, T_query, n_batches, batch_size, device, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    ok = tot = 0
    nll = 0.0
    for _ in range(n_batches):
        toks, rev, sps, ans, _i = env.generate_match_batch(
            batch_size, T_explore, T_query, rng)
        logits = model(toks[:, :-1].to(device))
        for b in range(toks.shape[0]):
            for p, a in zip(sps[b], ans[b]):
                if p >= logits.shape[1]:
                    continue
                sl = _match_logits(logits, env, b, p)
                t = a - env.obs_offset
                ok += int(sl.argmax().item() == t)
                nll += -F.log_softmax(sl, dim=-1)[t].item()
                tot += 1
    model.train()
    return ok / max(tot, 1), nll / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--T-explore", type=int, default=512)
    ap.add_argument("--T-query", type=int, default=256)
    ap.add_argument("--eval-query", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--obs-coef", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    env = MatchQueryGridWorld(size=64, n_obs_types=16, seed=args.seed)
    env_test = MatchQueryGridWorld(size=64, n_obs_types=16, seed=10000)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=64).to(dev)
    print(f"{args.variant} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} TE={args.T_explore} TQ={args.T_query} "
          f"chance={1/env.n_obs_types:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        t0 = time.time(); acc = 0.0; pm = 0.0; po = 0.0
        for _ in range(args.n_batches):
            toks, rev, sps, ans, _i = env.generate_match_batch(
                args.batch_size, args.T_explore, args.T_query, rng)
            L_match, L_obs = _losses(model, env, toks, rev, sps, ans, dev)
            loss = L_match + args.obs_coef * L_obs
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            acc += loss.item(); pm += L_match.item(); po += L_obs.item()
        losses.append(acc / args.n_batches)
        nb = args.n_batches
        if (ep + 1) % 5 == 0 or ep == 0:
            # chance-level match loss = -log(1/16) = 2.77
            print(f"  epoch {ep+1}/{args.epochs} loss={losses[-1]:.4f} "
                  f"[match={pm/nb:.4f} (chance 2.77) obs={po/nb:.4f}] "
                  f"({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for TQ in args.eval_query:
        a, nll, n = evaluate(model, env_test, args.T_explore, TQ, 8, 8, dev,
                             seed=5000 + args.seed)
        results[TQ] = {"match_acc": a, "match_nll": nll, "n": n}
        print(f"  [held-out] T_query={TQ}: acc={a:.4f} nll={nll:.4f} (n={n})",
              flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed, "results": results,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers},
               out / f"{args.variant}_matchquery.pt")
    json.dump(results, open(out / f"{args.variant}_matchquery.json", "w"), indent=2)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
