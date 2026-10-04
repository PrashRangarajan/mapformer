"""Trainer for NormStep on the text world (TW_NORMSTEP_PREREG.md). train_textworld.py with four step arms; the
data, recipe and held-out map are train_textworld's. Eval stream: np seed 10**6 (no overlap with any training
batch), held-out map env seed 10000, T = --n-steps and 2x. Writes <arm>.pt and eval.json in --output-dir."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_textworld import TextWorld
from mapformer.model_codes import MapWM_NormStep
from mapformer.model_textstep import MapWM_NormStepNB, MapWM_DirOnly
from mapformer.train import train
from mapformer.train_textworld import evaluate
from mapformer.train_variant import VARIANT_MAP

ARMS = {"MapWM": VARIANT_MAP["Vanilla_r4"], "NormStep": MapWM_NormStep, "NormStepNB": MapWM_NormStepNB,
        "DirOnly": MapWM_DirOnly}
EVAL_SEED, HELDOUT = 10**6, 10000


def build(arm, env, n_layers=1, size=64):
    m = ARMS[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=n_layers, grid_size=size)
    if arm == "DirOnly":
        m.set_step_ids([i for a in range(4) for i in env.dir_ids[a]])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = TextWorld(size=a.size, seed=a.seed)
    model = build(a.arm, env, a.n_layers, a.size)
    assert len(model.layers) == a.n_layers, (len(model.layers), a.n_layers)   # rule 17
    print(f"{a.arm} L{a.n_layers} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}")
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "arm": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size}}, out / f"{a.arm}.pt")
    te = TextWorld(size=a.size, seed=HELDOUT)
    ev = {}
    for T in (a.n_steps, 2 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"held-out map T={T}: acc {acc:.4f} nll {nll:.4f}")
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):]))},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
