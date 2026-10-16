"""Trainer for GAUGE_PREREG.md: the relative-sign gauge's prediction on the ring. Moves with steps (+1, -1) (opposite: a sign
cost expected) or (+1, +2) (same direction: no sign cost predicted by T1's corollary). RING_DIAL recipe (32 cells, T = 128,
batch 128, 98 batches, lr 1e-3, cosine, 300 epochs, 1 layer, 2 heads, d 128). Arms: SO / AO = SignedAO_r4 / AbsAO_r4
(observation-step-free: only the two moves step, so T1's corollary applies exactly), RoPE (index; a reference for whether the
task needs a map). Evaluation: train_cancel.evaluate on a fresh stream (np seed 10**6, maps redrawn) at T and 4T.
  python3 -m mapformer.train_gauge --arm AO --deltas 1,2 --seed 1300 --device cuda:0 --output-dir runs/gauge/p0/AO_d12_s1300
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, wrap_share
from mapformer.train import train
from mapformer.train_cancel import evaluate
from mapformer.train_variant import VARIANT_MAP

EVAL_SEED = 10 ** 6


def build(arm, env, size):
    kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=size)
    if arm == "RoPE":
        return VARIANT_MAP["RoPE"](**kw)
    cls = MapFormerWM_SignedAO_r4 if arm == "SO" else MapFormerWM_AbsAO_r4
    return cls(action_ids=(env.action_offset, env.action_offset + 1), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["SO", "AO", "RoPE"])
    ap.add_argument("--deltas", required=True, help="e.g. 1,-1 or 1,2")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)   # ROBUST_PREREG.md; 0.05 = train()'s default
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    deltas = tuple(int(x) for x in a.deltas.split(","))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = GridWorldSteps(size=a.size, seed=a.seed, p_first=0.5, deltas=deltas)
    model = build(a.arm, env, a.size)
    assert len(model.layers) == 1
    print(f"{a.arm} deltas={deltas} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers,
                   weight_decay=a.weight_decay)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size, "deltas": deltas,
                           "action_offset": env.action_offset}}, out / f"{a.arm}.pt")
    te = GridWorldSteps(size=a.size, seed=10000, p_first=0.5, deltas=deltas)
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)
    model.cpu()
    extra = {} if a.arm == "RoPE" else {"wrap": wrap_share(model)}
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):])), **extra},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
