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
import math
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
# Bump on ANY data-affecting change to the env (policy, _macro, discretization,
# bounds). 2026-08-25 v2: bounds-from-geometry + macro wall-break (changes the
# cell discretization vs the v1 random-walk bounds), so v2 buffers must not
# collide with v1. Included in the cache key.
CODE_VERSION = "v2"


def _env_kwargs(env, seed):
    """Reconstruct the MiniWorldWorld constructor kwargs (for spawning parallel
    build workers, each of which needs its own env / EGL context)."""
    return dict(env_name=env.env_name, grid_size=env.grid_size,
                n_obs_types=env.n_obs_types, p_empty=env.p_empty,
                n_dir=env.n_dir, seed=seed, allocentric=env.allocentric,
                fixed_map=env.fixed_map, oracle=getattr(env, "oracle", False))


def _build_chunk(payload):
    """Worker: build `count` trajectories in a fresh process. Top-level + picklable
    for multiprocessing 'spawn'. worker_seed makes each worker draw distinct
    fresh maps / start poses, so the pooled buffer is diverse."""
    import numpy as _np
    from mapformer.miniworld_env import MiniWorldWorld
    env_kwargs, n_steps, count, worker_seed = payload
    env = MiniWorldWorld(**env_kwargs)
    rng = _np.random.RandomState(worker_seed)
    tok = _np.zeros((count, 2 * n_steps), dtype=_np.int64)
    rev = _np.zeros((count, 2 * n_steps), dtype=bool)
    for i in range(count):
        tk, _om, rm = env.generate_trajectory(n_steps, rng=rng)
        tok[i] = tk.numpy(); rev[i] = rm.numpy()
    return tok, rev


def build_or_load_buffer(env, n_steps, buffer_size, seed, n_workers=1, tag=""):
    os.makedirs(_CACHE, exist_ok=True)
    key = (f"{CODE_VERSION}|{tag}|{env.env_name}|G{env.grid_size}|T{n_steps}|"
           f"obs{env.n_obs_types}|pe{env.p_empty}|dir{env.n_dir}|"
           f"allo{int(env.allocentric)}|orc{int(getattr(env,'oracle',False))}|"
           f"fix{int(env.fixed_map)}|seed{seed}|N{buffer_size}|w{n_workers}")
    path = os.path.join(_CACHE, "mw_" + hashlib.sha1(key.encode()).hexdigest()[:12] + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        print(f"[buffer] loaded {len(d['tokens'])} trajectories from {path}", flush=True)
        return d["tokens"], d["revisit"]
    t0 = time.time()
    if n_workers > 1:
        import multiprocessing as mp
        # split buffer_size across workers; each gets a distinct worker_seed
        base = buffer_size // n_workers
        counts = [base + (1 if i < buffer_size % n_workers else 0) for i in range(n_workers)]
        payloads = [(_env_kwargs(env, seed), n_steps, c, seed * 100003 + i)
                    for i, c in enumerate(counts) if c > 0]
        print(f"[buffer] building {buffer_size} trajectories T={n_steps} across "
              f"{len(payloads)} workers (one-time)...", flush=True)
        ctx = mp.get_context("spawn")             # clean EGL context per worker
        with ctx.Pool(len(payloads)) as pool:
            parts = pool.map(_build_chunk, payloads)
        tok = np.concatenate([p[0] for p in parts], axis=0)
        rev = np.concatenate([p[1] for p in parts], axis=0)
    else:
        print(f"[buffer] building {buffer_size} trajectories T={n_steps} (one-time)...", flush=True)
        rng = np.random.RandomState(seed)
        tok = np.zeros((buffer_size, 2 * n_steps), dtype=np.int64)
        rev = np.zeros((buffer_size, 2 * n_steps), dtype=bool)
        for i in range(buffer_size):
            tk, _om, rm = env.generate_trajectory(n_steps, rng=rng)
            tok[i] = tk.numpy(); rev[i] = rm.numpy()
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{buffer_size} ({(time.time()-t0)/(i+1)*1000:.0f} ms/traj)", flush=True)
    with open(path, "wb") as f:
        pickle.dump({"tokens": tok, "revisit": rev}, f)
    print(f"[buffer] built {len(tok)} + cached at {path} "
          f"({time.time()-t0:.0f}s total)", flush=True)
    return tok, rev


def build_or_load_eval_buffer(env_test, n_steps, n_trials, n_workers=1):
    """Held-out eval trajectories, built ONCE per (env_test config, T) and shared by
    every arm at that config. Removes the ~7/8 redundant live regeneration across
    the 4 variants AND makes the position effect exactly paired (identical held-out
    walks for path-int and index). env_test carries its own seed (10000 for
    fresh-map -> a NEW map; =train seed for fixed-map)."""
    return build_or_load_buffer(env_test, n_steps, n_trials, env_test.seed,
                                n_workers=n_workers, tag=f"eval{n_trials}")


@torch.no_grad()
def evaluate(model, eval_tok, eval_rev, blank, batch_size, device):
    """Score a PRE-BUILT eval set (tokens, revisit) -- deterministic + shared."""
    model.eval()
    ok = tot = 0; ok_nb = tot_nb = 0; nll_nb = 0.0
    N = eval_tok.shape[0]
    for s in range(0, N, batch_size):
        toks = torch.from_numpy(eval_tok[s:s + batch_size]).to(device)
        rm = torch.from_numpy(eval_rev[s:s + batch_size])
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
    ap.add_argument("--oracle", action="store_true",
                    help="exact-cell-transition action encoding (9 classes); makes "
                         "cumsum reconstruct position exactly. Overrides allocentric.")
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
    ap.add_argument("--n-workers", type=int, default=1,
                    help="parallel processes for one-time buffer build (each gets "
                         "its own env/EGL context); ignored if the buffer is cached")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-batches", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--eval-lengths", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--schedule", default="linear", choices=["linear", "cosine"],
                    help="cosine = 5%% warmup then cosine decay to 10%% of peak")
    ap.add_argument("--eval-trials", type=int, default=128,
                    help="held-out eval trajectories per length (pre-built + shared "
                         "across arms; was 8 batches x 16 = 128 live)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fast-attn", action="store_true",
                    help="use F.scaled_dot_product_attention + TF32 instead of the "
                         "explicit softmax(QK^T)V. Mathematically identical (logits "
                         "1.4e-06, grads 2.4e-08, grad cosine 1.0000000000) but 2.56x "
                         "faster at 37%% of the memory. Attention-dropout RNG draws "
                         "differ, so a run is NOT bit-identical to one without it -- "
                         "always include a same-budget control arm when mixing.")
    args = ap.parse_args()

    if args.fast_attn:
        import mapformer.model as _M
        _M.USE_SDPA = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[fast-attn] SDPA + TF32 enabled", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    kw = dict(env_name=args.env_name, grid_size=args.grid_size, n_obs_types=args.n_obs,
              n_dir=args.n_dir, allocentric=args.allocentric, fixed_map=args.fixed_map,
              oracle=args.oracle)
    env = MiniWorldWorld(seed=args.seed, **kw)
    # fixed_map: eval on the SAME map (novel walks, known layout); fresh_map:
    # eval on a held-out map (tests in-context generalisation).
    env_test = MiniWorldWorld(seed=args.seed if args.fixed_map else 10000, **kw)

    tok, rev = build_or_load_buffer(env, args.n_steps, args.buffer_size, args.seed,
                                    n_workers=args.n_workers)
    tok_t = torch.from_numpy(tok); rev_t = torch.from_numpy(rev)

    model = VARIANT_MAP[args.variant](
        vocab_size=env.unified_vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, grid_size=args.grid_size).to(dev)
    print(f"{args.variant} allo={args.allocentric} seed={args.seed} "
          f"params={sum(p.numel() for p in model.parameters()):,} vocab={env.unified_vocab_size} "
          f"chance(non-blank)={1/args.n_obs:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * args.n_batches
    # LinearLR(1.0 -> 0.0) decays from step ONE with no warmup. On a task with a
    # plateau-then-cliff loss transition that is actively harmful: by the time a run
    # could escape the plateau its LR is already near zero, so the budget measures
    # "did the transition fire early" rather than "can this model solve the task".
    # --schedule cosine adds a warmup then a cosine decay to 10% (not 0), leaving
    # usable LR late in training. Default stays linear for backward compatibility.
    if args.schedule == "cosine":
        warm = max(1, int(0.05 * total))
        def lr_fn(step):
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(1, total - warm)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    else:
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

    blank = env.obs_offset + env.blank_token
    results = {}
    for T in args.eval_lengths:
        et, er = build_or_load_eval_buffer(env_test, T, args.eval_trials,
                                           n_workers=args.n_workers)
        r = evaluate(model, et, er, blank, 16, dev)
        results[T] = r
        print(f"  [held-out] T={T}: acc={r['acc']:.4f} nb_acc={r['nb_acc']:.4f} "
              f"nb_nll={r['nb_nll']:.3f} (n_nb={r['n_nb']})", flush=True)

    enc = "oracle" if args.oracle else ("allo" if args.allocentric else "raw")
    if args.oracle and env._oracle_steps:
        print(f"  oracle multi-cell-jump clamp rate: "
              f"{env._oracle_clamped/max(env._oracle_steps,1):.4f}", flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "variant": args.variant,
                "allocentric": args.allocentric, "oracle": args.oracle,
                "results": results, "losses": losses,
                "vocab_size": env.unified_vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers},
               out / f"{args.variant}_{enc}.pt")
    json.dump(results, open(out / f"{args.variant}_{enc}.json", "w"), indent=2)
    print(f"DONE {args.variant} enc={enc}", flush=True)


if __name__ == "__main__":
    main()
