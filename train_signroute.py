"""Trainer for SIGNROUTE_PREREG.md (G-2 Part A): the sign ROUTE on the ring.

Recipe is LRSWEEP's verbatim, so the signed and abs arms here are read against numbers already
on record (RING_DIAL: 32 cells, moves +1/-1, observation-step-free, T = 128, batch 128, 98
batches, cosine, wd 0.05, 300 epochs, 1 layer, 2 heads, d 128, rank 4). Only `--arm` and `--lr`
vary across the batch.

Arms (`model_signroute.ARMS`): signed, abs, hybrid, abs_tied, exp, softplus. The first two are
bitwise-identical to model_wrap_exist's SignedAO_r4 / AbsAO_r4 (`test_signroute.py`), which is
what makes them usable as anchors.

Diagnostics written to eval.json, because the batch's reading depends on them:
  u_ratio      |z_x + z_y| / (|z_x| + |z_y|)   the collective coordinate of S2(a); ~0 means the
                                               model found the route (trained signed ring models
                                               sit at 0.044-0.053, SEARCH_THEORY Sec. 2)
  latent_cos   cos(z_x, z_y)                   -1 is the antipodal end state (signed: -0.994)
  clock_absmax max_c |omega_c (a_c(x) + a_c(y))|   the clock in angle units; 0 = cancelling
  wrap_share   share of channels cancelling by a 2 pi wrap rather than at 0 (model_wrap_exist)

  python3 -m mapformer.train_signroute --arm hybrid --lr 1e-3 --seed 1800 \
      --device cuda:1 --output-dir runs/signroute/hybrid_lr1e-3_s1800
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_signroute import ARMS
from mapformer.model_wrap_exist import wrap_share
from mapformer.train import train
from mapformer.train_cancel import evaluate

EVAL_SEED = 10 ** 6


@torch.no_grad()
def diagnostics(m):
    """S2/S3 readouts. Computed on CPU after training, so they cost nothing and cannot perturb it."""
    z = m.latents().double()                                  # (2, r)
    u = (z[0] + z[1]).norm()
    denom = z[0].norm() + z[1].norm()
    cos = torch.nn.functional.cosine_similarity(z[0], z[1], dim=0)
    clock = m.clock().abs().max()
    out = {"u_ratio": float(u / denom) if float(denom) > 0 else float("nan"),
           "latent_cos": float(cos), "clock_absmax": float(clock),
           "step_absmax": float(m.step_table()[m.act].abs().max()),
           "step_frac_negative": float((m.step_table()[m.act] < 0).double().mean())}
    # The EFFECTIVE readout (Amendment 1 finding 12): the hybrid applies |W_out|, so its stored W_out's
    # sign says nothing; report the matrix the step actually uses. omega's sign is free, so the
    # angle-space readout diag(omega) W_eff is sign-coherent per ROW rather than non-negative.
    W = m.action_to_lie.w_out.weight.double()
    W_eff = W.abs() if m.action_to_lie.mode == "hybrid" else W
    om = m.path_integrator.omega.detach().reshape(-1, 1).double()
    rows = om * W_eff
    coherent = ((rows >= 0).all(1) | (rows <= 0).all(1)).double().mean()
    out["readout_frac_negative"] = float((W_eff < 0).double().mean())
    out["readout_row_sign_coherent"] = float(coherent)
    try:
        out["wrap_share"] = wrap_share(m)
    except Exception as e:                      # diagnostics must never fail a run
        out["wrap_share_error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
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
    ap.add_argument("--freeze-move-latents", action="store_true",
                    help="G-2 torus part / ACCOUNTS T2; unused by Part A")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a checkpoint of the SAME arm (rule 14's trainable test: "
                         "a construction that scores 1.000 frozen has fallen to 0.642 trainable "
                         "before). Unused by the registered batch, which trains from scratch.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    deltas = tuple(int(x) for x in a.deltas.split(","))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = GridWorldSteps(size=a.size, seed=a.seed, p_first=0.5, deltas=deltas)
    model = ARMS[a.arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1,
                        grid_size=a.size, action_ids=(env.action_offset, env.action_offset + 1))
    assert len(model.layers) == 1
    if a.init_from:
        b = torch.load(a.init_from, map_location="cpu", weights_only=False)
        assert b["variant"] == a.arm, f"--init-from is a {b['variant']!r} checkpoint, arm is {a.arm!r}"
        model.load_state_dict(b["model_state_dict"])
        print(f"warm-started from {a.init_from}", flush=True)
    if a.freeze_move_latents:
        model.freeze_move_latents()
    npar = sum(p.numel() for p in model.parameters())
    print(f"{a.arm} lr={a.lr} seed={a.seed} deltas={deltas} params={npar:,} "
          f"vocab={env.unified_vocab_size}", flush=True)

    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                   n_steps=a.n_steps, n_batches=a.n_batches, device=a.device,
                   schedule="cosine", data_workers=a.data_workers, weight_decay=a.weight_decay)

    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm,
                "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size,
                           "deltas": deltas, "action_offset": env.action_offset,
                           "d_model": 128, "n_heads": 2, "n_layers": 1, "grid_size": a.size}},
               out / f"{a.arm}.pt")

    te = GridWorldSteps(size=a.size, seed=10000, p_first=0.5, deltas=deltas)
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)

    model.cpu()
    n_tail = max(1, len(losses) // 20)
    json.dump({"eval": ev, "arm": a.arm, "lr": a.lr, "seed": a.seed,
               "final_loss": float(np.mean(losses[-n_tail:])),
               "args": dict(vars(a)),                   # analyze_signroute checks every run's recipe
               "losses": [float(x) for x in losses],   # full curve: classify_run uses the final 5% (SOLVED)
                                                       # and the final 2 x 10% (STALLED / DESCENDING)
               "n_params": npar, **diagnostics(model)},
              open(out / "eval.json", "w"), indent=2)
    print(f"DONE {a.arm} lr={a.lr} s{a.seed}", flush=True)


if __name__ == "__main__":
    main()
