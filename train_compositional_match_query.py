"""Trainer for Compositional Match-Query (blind continuation, repeated motifs).

  L = L_match + obs_coef * L_obs
  L_match  CE at every scored query step (BOTH categories) over the non-blank
           observation types (chance = 1/n_obs_types = 0.0625).
  L_obs    the paper's objective on the explore phase: predict the observation at
           REVISITED cells, forcing the map to be built.

Held-out eval reports accuracy + NLL SPLIT BY CATEGORY:
  exact  cell exactly seen in explore (path-integration matching).
  cross  cell not seen, room explored + motif seen elsewhere
         (path-integration AND motif abstraction -- the synergy target).

Any registered variant works via VARIANT_MAP (Vanilla, Hourglass_k2,
Hourglass_CoarseIdx, PlainFlat, PlainHourglass, ...). Gates:
validate_compositional_match_query.py.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_compositional_match_query import CompositionalMatchQueryGridWorld
from mapformer.train_variant import VARIANT_MAP


def _gather_scored(sps, ans, cats, Lp, obs_offset, device):
    """Flatten ragged (per-batch) scored positions into index tensors for a
    single batched cross_entropy. Returns (b_idx, p_idx, t_idx, cats_list) or
    (None, ..) if empty. t_idx is the target index within the non-blank block."""
    b_idx, p_idx, t_idx, cs = [], [], [], []
    for b in range(len(sps)):
        clist = cats[b] if cats is not None else [None] * len(sps[b])
        for p, a, c in zip(sps[b], ans[b], clist):
            if p < Lp:
                b_idx.append(b); p_idx.append(p); t_idx.append(a - obs_offset); cs.append(c)
    if not b_idx:
        return None, None, None, cs
    return (torch.tensor(b_idx, device=device), torch.tensor(p_idx, device=device),
            torch.tensor(t_idx, device=device), cs)


def _losses(model, env, toks, rev, sps, ans, device):
    toks = toks.to(device); rev = rev.to(device)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    logits = model(inp)
    lo, K = env.obs_offset, env.n_obs_types
    b_t, p_t, t_t, _ = _gather_scored(sps, ans, None, logits.shape[1], lo, device)
    if b_t is not None:
        sl = logits[b_t, p_t][:, lo:lo + K]        # (M, K) -- one batched CE
        L_match = F.cross_entropy(sl, t_t)
    else:
        L_match = logits.sum() * 0.0
    m = rev[:, 1:]
    L_obs = F.cross_entropy(logits[m], tgt[m]) if m.any() else logits.sum() * 0.0
    return L_match, L_obs


@torch.no_grad()
def evaluate(model, env, T_explore, T_query, n_batches, batch_size, device, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    ok = defaultdict(int); tot = defaultdict(int); nll = defaultdict(float)
    lo, K = env.obs_offset, env.n_obs_types
    for _ in range(n_batches):
        toks, rev, sps, ans, cats, _i = env.generate_cmq_batch(
            batch_size, T_explore, T_query, rng)
        logits = model(toks[:, :-1].to(device))
        b_t, p_t, t_t, cs = _gather_scored(sps, ans, cats, logits.shape[1], lo, device)
        if b_t is None:
            continue
        sl = logits[b_t, p_t][:, lo:lo + K]                  # (M, K)
        hit = (sl.argmax(-1) == t_t)
        ll = -F.log_softmax(sl, dim=-1).gather(1, t_t.unsqueeze(1)).squeeze(1)
        cs = np.array(cs)
        for key in ("exact", "cross", "all"):
            sel = np.ones(len(cs), bool) if key == "all" else (cs == key)
            if not sel.any():
                continue
            idx = torch.tensor(np.nonzero(sel)[0], device=device)
            ok[key] += int(hit[idx].sum().item()); tot[key] += int(sel.sum())
            nll[key] += float(ll[idx].sum().item())
    model.train()
    return {k: {"acc": ok[k] / max(tot[k], 1), "nll": nll[k] / max(tot[k], 1),
                "n": tot[k]} for k in ("exact", "cross", "all")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--room-size", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--T-explore", type=int, default=512)
    ap.add_argument("--T-query", type=int, default=256)
    ap.add_argument("--eval-query", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--obs-coef", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="fraction of total steps for linear LR warmup 0->1 "
                         "before linear decay. Stabilises the bimodal training "
                         "basin on this multi-hop task.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    kw = dict(size=args.size, room_size=args.room_size, n_obs_types=args.n_obs,
              n_templates=args.n_templates)
    env = CompositionalMatchQueryGridWorld(seed=args.seed, **kw)
    env_test = CompositionalMatchQueryGridWorld(seed=10000, **kw)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=args.size).to(dev)
    print(f"{args.variant} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} TE={args.T_explore} TQ={args.T_query} "
          f"chance={1/env.n_obs_types:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    warmup = int(args.warmup_frac * total)

    def _lr(s):
        if warmup > 0 and s < warmup:
            return s / warmup                        # linear 0 -> 1
        return max(0.0, (total - s) / max(total - warmup, 1))  # linear -> 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        t0 = time.time(); acc = 0.0; pm = 0.0; po = 0.0
        for _ in range(args.n_batches):
            toks, rev, sps, ans, cats, _i = env.generate_cmq_batch(
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
            print(f"  epoch {ep+1}/{args.epochs} loss={losses[-1]:.4f} "
                  f"[match={pm/nb:.4f} (chance 2.77) obs={po/nb:.4f}] "
                  f"({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for TQ in args.eval_query:
        r = evaluate(model, env_test, args.T_explore, TQ, 8, 8, dev,
                     seed=5000 + args.seed)
        results[TQ] = r
        print(f"  [held-out] T_query={TQ}: "
              f"exact={r['exact']['acc']:.4f}(n={r['exact']['n']}) "
              f"cross={r['cross']['acc']:.4f}(n={r['cross']['n']}) "
              f"all={r['all']['acc']:.4f}", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed, "results": results,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers},
               out / f"{args.variant}_cmq.pt")
    json.dump(results, open(out / f"{args.variant}_cmq.json", "w"), indent=2)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
