"""Trainer for the Lap task (CSCG event-specific representations).

  L = L_next + dec_coef * L_dec

  L_next  next-token CE at every valid position. Learns the track itself, which
          is easy after lap 1 (copy the observation from loop_len steps ago).
  L_dec   next-token CE at the K lap-boundary decision positions only. Without
          this the K decisions are ~K out of ~8*loop_len tokens and the signal is
          diluted to nothing -- the failure mode that made Map-Query look like a
          negative result when it was undertrained.

dec_coef defaults to 1.0, stated rather than tuned.

METRICS (see validate_lap.py for why boundary accuracy alone is unusable):
  hit          REWARD is argmax at the K-th boundary
  false_alarm  fraction of boundaries 1..K-1 where REWARD is argmax
  exact        hit AND no false alarms   <- HEADLINE
Floors: always-no 0.000, always-yes 0.000, random-boundary 0.250 (K=4),
oracle 1.000. Gates require VARIABLE loop_len -- fixed gives a positional
shortcut that scores 1.000.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_lap import LapWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


def _losses(model, env, toks, valid, dec_pos, dec_lab, device):
    toks = toks.to(device); valid = valid.to(device)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    logits = model(inp)
    m = valid[:, 1:]
    L_next = F.cross_entropy(logits[m], tgt[m]) if m.any() else logits.sum() * 0.0

    dl = []
    for b in range(toks.shape[0]):
        for p in dec_pos[b]:
            if p < logits.shape[1]:
                dl.append(F.cross_entropy(logits[b, p].unsqueeze(0),
                                          tgt[b, p].unsqueeze(0)))
    L_dec = torch.stack(dl).mean() if dl else logits.sum() * 0.0
    return L_next, L_dec


@torch.no_grad()
def evaluate(model, env, n_batches, batch_size, device, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    hits = fas = exacts = n_ep = 0
    fa_tot = 0
    for _ in range(n_batches):
        toks, valid, dec_pos, dec_lab, _info = env.generate_lap_batch(batch_size, rng)
        logits = model(toks[:, :-1].to(device))
        for b in range(toks.shape[0]):
            hit, fa = 0, 0
            for p, lab in zip(dec_pos[b], dec_lab[b]):
                if p >= logits.shape[1]:
                    continue
                is_rew = int(logits[b, p].argmax().item() == env.reward_tok)
                if lab == 1:
                    hit = is_rew
                else:
                    fa += is_rew; fa_tot += 1
            hits += hit; fas += fa
            exacts += int(hit == 1 and fa == 0)
            n_ep += 1
    model.train()
    return (hits / max(n_ep, 1), fas / max(fa_tot, 1), exacts / max(n_ep, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-laps", type=int, default=4)
    ap.add_argument("--eval-laps", nargs="+", type=int, default=[4, 6],
                    help="6 = OOD lap count, never seen in training")
    ap.add_argument("--fixed-loop", action="store_true",
                    help="VOID: gates give a positional shortcut of 1.000 here")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dec-coef", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    env = LapWorld(n_laps=args.n_laps, fixed_loop=args.fixed_loop, seed=args.seed)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=64).to(dev)
    print(f"{args.variant} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={env.unified_vocab_size} K={args.n_laps} "
          f"fixed_loop={args.fixed_loop} (exact floors: random 0.25, oracle 1.0)",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        t0 = time.time(); acc = pn = pd = 0.0
        for _ in range(args.n_batches):
            toks, valid, dp, dl, _i = env.generate_lap_batch(args.batch_size, rng)
            L_next, L_dec = _losses(model, env, toks, valid, dp, dl, dev)
            loss = L_next + args.dec_coef * L_dec
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            acc += loss.item(); pn += L_next.item(); pd += L_dec.item()
        losses.append(acc / args.n_batches)
        if (ep + 1) % 10 == 0 or ep == 0:
            nb = args.n_batches
            print(f"  epoch {ep+1}/{args.epochs} loss={losses[-1]:.4f} "
                  f"[next={pn/nb:.4f} dec={pd/nb:.4f}] ({time.time()-t0:.0f}s)",
                  flush=True)

    results = {}
    for K in args.eval_laps:
        env_t = LapWorld(n_laps=K, fixed_loop=args.fixed_loop, seed=10000)
        h, f, e = evaluate(model, env_t, 8, 32, dev, seed=6000 + args.seed)
        results[K] = {"hit": h, "false_alarm": f, "exact": e}
        print(f"  [held-out] K={K}: hit={h:.4f} false_alarm={f:.4f} exact={e:.4f}",
              flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed, "results": results,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers},
               out / f"{args.variant}_lap.pt")
    json.dump(results, open(out / f"{args.variant}_lap.json", "w"), indent=2)
    print(f"DONE {args.variant} final_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
