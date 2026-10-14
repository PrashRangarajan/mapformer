"""Ring trainer for GENERALITY_PREREG.md: a copy of train_ring_dial.py (used by a registered batch, so not edited) with the
published monotone arms added. Arms from VARIANT_MAP: Signed_r4, Abs_r4, RoPE, CARoPEF, CoPEF (model_mono_baselines.py).
Recipe = RING_DIAL's (ring 32, T = 128 moves, batch 128, 98 batches, lr 1e-3, cosine, 300 epochs, 1 layer, 2 heads,
d 128); evaluation on a fresh stream (np seed 10**6; maps redrawn per trajectory) at T and 4T. Writes <arm>.pt, eval.json.
  python3 -m mapformer.train_ring_variant --arm CARoPEF --p-plus 0.5 --seed 30 --device cuda:0 --output-dir runs/...
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_cancel import GridWorldCancel
from mapformer.train import train
from mapformer.train_cancel import evaluate
from mapformer.train_variant import VARIANT_MAP

ARMS = ("Signed_r4", "Abs_r4", "RoPE", "CARoPEF", "CoPEF")
EVAL_SEED = 10 ** 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--p-plus", type=float, required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = GridWorldCancel(size=a.size, seed=a.seed, p_plus=a.p_plus)
    model = VARIANT_MAP[a.arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=a.size)
    assert len(model.layers) == 1
    print(f"{a.arm} p_plus={a.p_plus} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size}}, out / f"{a.arm}.pt")
    te = GridWorldCancel(size=a.size, seed=10000, p_plus=a.p_plus)
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):]))},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
