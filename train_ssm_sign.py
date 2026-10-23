"""Trainer for G3_SSM_PREREG.md (G-3): the sign test in a one-layer diagonal complex SSM (model_ssm_sign.py).

Task and recipe are the attention ring anchors' (LRSWEEP / SIGNROUTE, train_signroute.py) verbatim: GridWorldSteps ring of
--size cells, moves +1 / -1 (p_first 0.5, runs of 1-10), observation map redrawn per walk, loss on revisited observations
only (train.train), T = 128, batch 128, 98 batches, cosine with 5% warmup, AdamW wd 0.05, 300 epochs, d 128, 2 heads,
rank 4, 3 data workers. Evaluation: train_cancel.evaluate on a fresh stream (np seed 10^6, env seed 10000) at T and 4T.

Arms: --arm signed | abs | softplus | exp | index (every token steps 1: input-independent rotation) | none (memoryless:
mixer output 0); --steps moves (observation steps 0 by construction; registered primary) | all.

Rule 14 continuations: --init-built PATH loads a construct_ssm_sign.py checkpoint (class = --arm), records its FROZEN
evaluation before any update, then trains on a fresh walk stream (data_seed_offset 10^6) with the same schedule.

  python3 -m mapformer.train_ssm_sign --arm abs --steps moves --size 32 --lr 1e-3 --seed 2100 \
      --device cuda:1 --output-dir runs/ssm_sign/A/abs_m32_lr1e-3_s2100
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_ssm_sign import MODES, STEPS, build
from mapformer.train import train
from mapformer.train_cancel import evaluate

EVAL_SEED = 10 ** 6
CONT_OFFSET = 10 ** 6


@torch.no_grad()
def diagnostics(m):
    """Step-table readouts (CPU, after training; they cannot perturb it)."""
    if m.mode == "index":
        return {}
    A = m.angle_table()[m.act].reshape(len(m.act), -1).double()        # (2, C) move angles
    z = m.latents().double()
    clock = torch.angle(torch.exp(1j * (A[0] + A[1])))                  # wrapped clock per channel
    big = torch.angle(torch.exp(1j * A[0])).abs() > 0.05                # wrap-invariant non-blind channels
    out = {"latent_cos": float(torch.nn.functional.cosine_similarity(z[0], z[1], dim=0)),
           "u_ratio": float((z[0] + z[1]).norm() / (z[0].norm() + z[1].norm())),
           "clock_absmax_raw": float((A[0] + A[1]).abs().max()),        # unwrapped: 0 = cancel at k = 0
           "clock_wrapped_median_nonblind": float(clock[big].abs().median()) if big.any() else float("nan"),
           "cancel_share_nonblind": float((clock[big].abs() < 0.1).double().mean()) if big.any() else float("nan"),
           "wrap_cancel_share_nonblind": float(((clock[big].abs() < 0.1) & ((A[0] + A[1])[big].abs() > 1.0))
                                               .double().mean()) if big.any() else float("nan"),
           "move_step_min": float(m.step_table()[m.act].min()),
           "move_step_absmax": float(m.step_table()[m.act].abs().max()),
           "move_step_frac_negative": float((m.step_table()[m.act] < 0).double().mean())}
    if m.steps == "all":
        obs = [i for i in range(m.vocab_size) if i not in m.act.tolist()]
        Ao = m.angle_table()[obs].double()
        out["obs_angle_wrapped_absmax"] = float(torch.angle(torch.exp(1j * Ao)).abs().max())
        out["obs_angle_wrapped_median"] = float(torch.angle(torch.exp(1j * Ao)).abs().median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(MODES) + ["none"])
    ap.add_argument("--steps", default="moves", choices=STEPS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--deltas", default="1,-1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--decay", default="none", choices=("none", "learned"))
    ap.add_argument("--init-built", default=None, help="rule 14 continuation from a construct_ssm_sign checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    deltas = tuple(int(x) for x in a.deltas.split(","))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = GridWorldSteps(size=a.size, seed=a.seed, p_first=0.5, deltas=deltas)
    act = tuple(env.action_offset + i for i in range(len(deltas)))
    mode, mixer = (a.arm, "ssm") if a.arm != "none" else ("signed", "none")
    model = build(mode, a.steps, env.unified_vocab_size, a.size, action_ids=act, mixer=mixer, decay=a.decay,
                  bottleneck_r=a.rank)
    te = GridWorldSteps(size=a.size, seed=10000, p_first=0.5, deltas=deltas)
    frozen = None
    if a.init_built:
        b = torch.load(a.init_built, map_location="cpu", weights_only=False)
        assert b["variant"] == a.arm, f"--init-built is a {b['variant']!r} construction, arm is {a.arm!r}"
        assert b["config"]["steps"] == a.steps and b["config"]["size"] == a.size, b["config"]
        model.load_state_dict(b["model_state_dict"])
        model.to(a.device)
        frozen = {"acc": evaluate(model, te, a.n_steps, a.n_trials, a.device, seed=EVAL_SEED)[0],
                  "construction": b["config"].get("construction"), "built_from": b["config"].get("built_from")}
        print(f"built {a.arm}: frozen acc T={a.n_steps} {frozen['acc']:.4f}", flush=True)
    npar = sum(p.numel() for p in model.parameters())
    print(f"SSM {a.arm} steps={a.steps} m={a.size} lr={a.lr} seed={a.seed} params={npar:,} "
          f"vocab={env.unified_vocab_size}", flush=True)
    t0 = time.time()
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers,
                   weight_decay=a.weight_decay, data_seed_offset=CONT_OFFSET if a.init_built else 0)
    wall = time.time() - t0
    cfg = {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size, "deltas": deltas, "action_ids": act,
           "steps": a.steps, "size": a.size, "grid_size": a.size, "n_steps": a.n_steps, "mixer": mixer,
           "decay": a.decay, "d_model": 128, "n_heads": 2, "rank": a.rank}
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": mode if mixer == "ssm" else "none",
                "seed": a.seed, "config": cfg}, out / f"{a.arm}.pt")
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)
    model.cpu()
    n_tail = max(1, len(losses) // 20)
    json.dump({"eval": ev, "arm": a.arm, "steps": a.steps, "size": a.size, "lr": a.lr, "seed": a.seed,
               "final_loss": float(np.mean(losses[-n_tail:])), "args": dict(vars(a)),
               "losses": [float(x) for x in losses], "n_params": npar, "wall_s": wall,
               "s_per_epoch": wall / max(1, a.epochs), "frozen": frozen,
               "torch": torch.__version__, **diagnostics(model)},
              open(out / "eval.json", "w"), indent=2)
    print(f"DONE {a.arm} {a.steps} m{a.size} lr={a.lr} s{a.seed} ({wall / a.epochs:.2f} s/epoch)", flush=True)


if __name__ == "__main__":
    main()
