"""Trainer for the Map-Query task.

Objective is three terms, all on the same forward pass:

  L = L_direction + L_room + obs_coef * L_obs

  L_direction  at each direction query, -log( sum_{a in S} p_a ) where S is the
               set of optimal first actions. This maximises P(prediction is in S),
               which is EXACTLY the eval metric -- several first actions can be
               simultaneously optimal on a torus, so training against one
               canonical choice would penalise a correct model.
  L_room       plain CE over the 64 room tokens ("which room are you in?").
  L_obs        the paper's objective: predict the observation at REVISITED cells
               during explore. Without it the queries alone would not force a map
               to be built.

obs_coef is a free weighting parameter, defaulted to 1.0 (equal weight) rather
than tuned, so it is a stated choice rather than a silent one.

Answer slots are fed back as MASK, never as the answer -- see environment_map_query.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_map_query import MapQueryGridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


def _losses(model, env, toks, rev, dirs, rooms, device):
    toks = toks.to(device); rev = rev.to(device)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    logits = model(inp)
    B = toks.shape[0]

    # --- direction queries: maximise total probability on the optimal set ---
    dl = []
    for b in range(B):
        pos, ans = dirs[b]
        for p, S in zip(pos, ans):
            if not S:
                continue
            lp = F.log_softmax(logits[b, p, :env.N_ACTIONS], dim=-1)
            dl.append(-torch.logsumexp(lp[list(S)], dim=0))
    L_dir = torch.stack(dl).mean() if dl else logits.sum() * 0.0

    # --- room queries: CE over the room-token block ---
    rl = []
    for b in range(B):
        pos, ans = rooms[b]
        for p, r in zip(pos, ans):
            sl = logits[b, p, env.room_tok0:env.room_tok0 + env.n_rooms]
            rl.append(F.cross_entropy(sl.unsqueeze(0),
                                      torch.tensor([r], device=device)))
    L_room = torch.stack(rl).mean() if rl else logits.sum() * 0.0

    # --- paper objective: observation prediction at revisited cells ---
    m = rev[:, 1:]
    L_obs = (F.cross_entropy(logits[m], tgt[m]) if m.any()
             else logits.sum() * 0.0)
    return L_dir, L_room, L_obs


@torch.no_grad()
def evaluate(model, env, T_explore, n_q, n_rq, n_batches, batch_size, device, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    d_ok = d_tot = r_ok = r_tot = 0
    for _ in range(n_batches):
        toks, rev, dirs, rooms, _i = env.generate_query_batch(
            batch_size, T_explore, n_q, n_rq, rng)
        logits = model(toks[:, :-1].to(device))
        for b in range(toks.shape[0]):
            pos, ans = dirs[b]
            for p, S in zip(pos, ans):
                if not S:
                    continue
                d_ok += int(logits[b, p, :env.N_ACTIONS].argmax().item() in S)
                d_tot += 1
            pos, ans = rooms[b]
            for p, r in zip(pos, ans):
                sl = logits[b, p, env.room_tok0:env.room_tok0 + env.n_rooms]
                r_ok += int(sl.argmax().item() == r)
                r_tot += 1
    model.train()
    return d_ok / max(d_tot, 1), r_ok / max(r_tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--T-explore", type=int, default=256,
                    help="gates PASS only at >=256; 64 fails assume-start at 0.623")
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--n-room-queries", type=int, default=4)
    ap.add_argument("--eval-explore", nargs="+", type=int, default=[256, 512, 1024])
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
    env = MapQueryGridWorld(size=64, room_size=8, seed=args.seed)
    env_test = MapQueryGridWorld(size=64, room_size=8, seed=10000)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=64).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.variant} seed={args.seed} params={n_par:,} "
          f"vocab={env.unified_vocab_size} T_explore={args.T_explore}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        t0 = time.time(); acc = 0.0
        for _ in range(args.n_batches):
            toks, rev, dirs, rooms, _i = env.generate_query_batch(
                args.batch_size, args.T_explore, args.n_queries,
                args.n_room_queries, rng)
            L_dir, L_room, L_obs = _losses(model, env, toks, rev, dirs, rooms, dev)
            loss = L_dir + L_room + args.obs_coef * L_obs
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            acc += loss.item()
        losses.append(acc / args.n_batches)
        print(f"  epoch {ep+1}/{args.epochs} loss={losses[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for T in args.eval_explore:
        d, r = evaluate(model, env_test, T, args.n_queries, args.n_room_queries,
                        8, 16, dev, seed=4000 + args.seed)
        results[T] = {"direction_acc": d, "room_acc": r}
        print(f"  [held-out] T_explore={T}: direction={d:.4f} room={r:.4f}", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed,
                "results": results, "vocab_size": env.unified_vocab_size,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers},
               out / f"{args.variant}_mapquery.pt")
    json.dump(results, open(out / f"{args.variant}_mapquery.json", "w"), indent=2)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
