"""Train a variant on the MiniWorld continuous-3D cognitive-map task.

MiniWorld trajectory generation is slow (~0.6 s each), so training samples from a
pre-generated, disk-cached buffer (built once per (env,grid,T,n_obs,n_dir,allo,
seed,N); pre-build with run_miniworld.sh to avoid concurrent-build races).

Objective: the paper's revisit-masked next-token CE. Held-out eval reports BOTH
overall accuracy and NON-BLANK accuracy (the validity gate flagged that ~41% of
scored cells are blank, so overall is inflated; non-blank is the primary metric,
chance 1/n_obs).

--allocentric selects the world-fixed displacement action encoding.
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.miniworld_env import MiniWorldWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_REPO, "runs", "_miniworld_cache")


def build_or_load_buffer(env, n_steps, buffer_size, seed):
    os.makedirs(_CACHE, exist_ok=True)
    key = (f"{env.env_name}|G{env.grid_size}|T{n_steps}|obs{env.n_obs_types}|"
           f"dir{env.n_dir}|allo{int(env.allocentric)}|fix{int(env.fixed_map)}|"
           f"seed{seed}|N{buffer_size}")
    path = os.path.join(_CACHE, "mw_" + hashlib.sha1(key.encode()).hexdigest()[:12] + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        print(f"[buffer] loaded {len(d['tokens'])} trajectories from {path}", flush=True)
        return d["tokens"], d["revisit"]
    print(f"[buffer] building {buffer_size} trajectories T={n_steps} (one-time)...", flush=True)
    rng = np.random.RandomState(seed)
    tok = np.zeros((buffer_size, 2 * n_steps), dtype=np.int64)
    rev = np.zeros((buffer_size, 2 * n_steps), dtype=bool)
    t0 = time.time()
    for i in range(buffer_size):
        tk, _om, rm = env.generate_trajectory(n_steps, rng=rng)
        tok[i] = tk.numpy(); rev[i] = rm.numpy()
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{buffer_size} ({(time.time()-t0)/(i+1)*1000:.0f} ms/traj)", flush=True)
    with open(path, "wb") as f:
        pickle.dump({"tokens": tok, "revisit": rev}, f)
    print(f"[buffer] built + cached at {path}", flush=True)
    return tok, rev


@torch.no_grad()
def evaluate(model, env, n_steps, n_batches, batch_size, device, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    blank = env.obs_offset + env.blank_token
    ok = tot = 0; ok_nb = tot_nb = 0; nll_nb = 0.0
    for _ in range(n_batches):
        toks, om, rm, _l = env.generate_batch(batch_size, n_steps, rng=rng)
        toks = toks.to(device)
        inp, tgt = toks[:, :-1], toks[:, 1:]
        m = rm[:, 1:].to(device)
        if m.sum() == 0:
            continue
        logits = model(inp)
        pred = logits.argmax(-1)
        ok += int((pred[m] == tgt[m]).sum()); tot += int(m.sum())
        nb = m & (tgt != blank)
        if nb.sum() > 0:
            lp = F.log_softmax(logits, -1)
            ok_nb += int((pred[nb] == tgt[nb]).sum()); tot_nb += int(nb.sum())
            nll_nb += -float(lp[nb].gather(-1, tgt[nb].unsqueeze(-1)).sum())
    model.train()
    return {"acc": ok / max(tot, 1), "nb_acc": ok_nb / max(tot_nb, 1),
            "nb_nll": nll_nb / max(tot_nb, 1), "n": tot, "n_nb": tot_nb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allocentric", action="store_true")
    ap.add_argument("--fixed-map", action="store_true",
                    help="reuse one obs_map per seed (path integration on a known "
                         "map, data-efficient) instead of fresh-per-episode "
                         "(in-context map building, data-hungry / memorises)")
    ap.add_argument("--env-name", default="MiniWorld-OneRoom-v0")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--n-dir", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--buffer-size", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-batches", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--eval-lengths", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    kw = dict(env_name=args.env_name, grid_size=args.grid_size, n_obs_types=args.n_obs,
              n_dir=args.n_dir, allocentric=args.allocentric, fixed_map=args.fixed_map)
    env = MiniWorldWorld(seed=args.seed, **kw)
    # fixed_map: eval on the SAME map (novel walks, known layout); fresh_map:
    # eval on a held-out map (tests in-context generalisation).
    env_test = MiniWorldWorld(seed=args.seed if args.fixed_map else 10000, **kw)

    tok, rev = build_or_load_buffer(env, args.n_steps, args.buffer_size, args.seed)
    tok_t = torch.from_numpy(tok); rev_t = torch.from_numpy(rev)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=args.grid_size).to(dev)
    print(f"{args.variant} allo={args.allocentric} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} vocab={env.unified_vocab_size} "
          f"chance(non-blank)={1/args.n_obs:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total)
    crit = nn.CrossEntropyLoss()
    N = tok_t.shape[0]; losses = []
    for ep in range(args.epochs):
        t0 = time.time(); acc = 0.0
        for _ in range(args.n_batches):
            idx = torch.randint(0, N, (args.batch_size,))
            batch = tok_t[idx].to(dev); rmask = rev_t[idx].to(dev)
            inp, tgt = batch[:, :-1], batch[:, 1:]; m = rmask[:, 1:]
            if m.sum() == 0:
                continue
            logits = model(inp)
            loss = crit(logits[m], tgt[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); acc += loss.item()
        losses.append(acc / args.n_batches)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{args.epochs} loss={losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for T in args.eval_lengths:
        r = evaluate(model, env_test, T, 8, 16, dev, seed=5000 + args.seed)
        results[T] = r
        print(f"  [held-out] T={T}: acc={r['acc']:.4f} nb_acc={r['nb_acc']:.4f} "
              f"nb_nll={r['nb_nll']:.3f} (n_nb={r['n_nb']})", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "variant": args.variant,
                "allocentric": args.allocentric, "results": results, "losses": losses,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers},
               out / f"{args.variant}_{'allo' if args.allocentric else 'raw'}.pt")
    json.dump(results, open(out / f"{args.variant}_{'allo' if args.allocentric else 'raw'}.json", "w"), indent=2)
    print(f"DONE {args.variant} allo={args.allocentric}", flush=True)


if __name__ == "__main__":
    main()
