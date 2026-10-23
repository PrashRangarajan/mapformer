"""Trainer for SIGNSEP_PREREG.md: the 2x2 that separates the missing collective route (S3) from the wrap barrier
(S1 + P3 + P4) as accounts of the |.| learnability cost on the ring.

Recipe = LRSWEEP / SIGNROUTE verbatim (`train_signroute.py`): ring 32 cells, moves +1/-1 (GridWorldSteps, p_first 0.5),
observation-step-free rank-4 MapWM, T = 128, batch 128, 98 batches, 300 epochs, cosine, AdamW wd 0.05, 1 layer, 2 heads,
d 128, 3 data workers. Only `--arm` and `--lr` vary across the batch. Every arm starts at the COMMON START
(`model_signsep.build`: MECHANISM's balanced-gauge |.| start), so the four cells are function-identical at init.

Arms (`model_signsep.ARMS`): signed (S), abs (A), abs_wrap (W), signed_frozen (F), abs_wrap_past (control C1).
`--init-from` warm-starts from a checkpoint of the SAME arm (rule 14's trainable test; pilots only), and then the walk
stream is offset by 10**6 (fresh walks, as WRAP_EXISTENCE's continuations).

  cd /home/prashr && python3 -m mapformer.train_signsep --arm abs_wrap --lr 1e-3 --seed 2300 \
      --device cuda:0 --output-dir /home/prashr/mapformer/runs/signsep/p0/abs_wrap_lr1e-3_s2300
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_signsep import ARMS, build, diagnostics, start_stats
from mapformer.train import train
from mapformer.train_cancel import evaluate

EVAL_SEED = 10 ** 6
WARM_OFFSET = 10 ** 6


def exempt_from_decay(params):
    """Make AdamW's decoupled weight decay skip `params` (Amendment 2): `train()` builds AdamW over model.parameters()
    with one weight decay, so the decay is undone after each step (exact up to float32 rounding, ~1e-6). torch's AdamW applies
    p <- p (1 - lr wd) - lr * adam_update; adding lr wd p_before afterwards leaves p_before - lr * adam_update, the update
    with wd = 0. Global optimizer hooks: this process has one optimizer. Returns the two hook handles."""
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
    params = list(params)
    saved = {}

    def pre(opt, args, kwargs):
        saved.clear()
        for g in opt.param_groups:
            for p in g["params"]:
                if any(p is q for q in params) and p.grad is not None:
                    saved[id(p)] = (p.detach().clone(), g["lr"] * g["weight_decay"])

    def post(opt, args, kwargs):
        with torch.no_grad():
            for p in params:
                if id(p) in saved:
                    before, f = saved[id(p)]
                    p.add_(before, alpha=f)
    return register_optimizer_step_pre_hook(pre), register_optimizer_step_post_hook(post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--deltas", default="1,-1")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--init-from", default=None, help="warm start from a checkpoint of the SAME arm (pilots only)")
    ap.add_argument("--tau-decay", choices=("off", "on"), default="off",
                    help="abs_wrap arms: weight decay on tau (Amendment 2: registered OFF -- decay pulls the collective "
                         "coordinate toward T = 0, the near side of the barrier; 'on' reproduces the first pilots)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    deltas = tuple(int(x) for x in a.deltas.split(","))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = GridWorldSteps(size=a.size, seed=a.seed, p_first=0.5, deltas=deltas)
    model, info = build(a.arm, env.unified_vocab_size, (env.action_offset, env.action_offset + 1), grid_size=a.size)
    assert len(model.layers) == 1
    start = {"start_angle_err": info["start_angle_err"], **start_stats(model)}
    if a.init_from:
        b = torch.load(a.init_from, map_location="cpu", weights_only=False)
        assert b["variant"] == a.arm, f"--init-from is a {b['variant']!r} checkpoint, arm is {a.arm!r}"
        model.load_state_dict(b["model_state_dict"])
        start.update({"warm_from": a.init_from, **{f"warm_{k}": v for k, v in start_stats(model).items()}})
        print(f"warm-started from {a.init_from}", flush=True)
    if hasattr(model, "wrap_tau") and a.tau_decay == "off":
        exempt_from_decay([model.wrap_tau])
    npar = sum(p.numel() for p in model.parameters())
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{a.arm} lr={a.lr} seed={a.seed} params={npar:,} trainable={ntr:,} vocab={env.unified_vocab_size} "
          f"start {start}", flush=True)

    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers,
                   weight_decay=a.weight_decay, data_seed_offset=WARM_OFFSET if a.init_from else 0)

    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size, "deltas": deltas,
                           "action_offset": env.action_offset, "d_model": 128, "n_heads": 2, "n_layers": 1,
                           "grid_size": a.size}},
               out / f"{a.arm}.pt")

    te = GridWorldSteps(size=a.size, seed=10000, p_first=0.5, deltas=deltas)
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)

    n_tail = max(1, len(losses) // 20)
    json.dump({"eval": ev, "arm": a.arm, "lr": a.lr, "seed": a.seed,
               "final_loss": float(np.mean(losses[-n_tail:])), "args": dict(vars(a)),
               "losses": [float(x) for x in losses], "n_params": npar, "n_trainable": ntr, "start": start,
               **diagnostics(model)},
              open(out / "eval.json", "w"), indent=2)
    print(f"DONE {a.arm} lr={a.lr} s{a.seed}", flush=True)


if __name__ == "__main__":
    main()
