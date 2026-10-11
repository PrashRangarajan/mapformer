"""Trainer for RANK_TESTS_PREREG.md experiment 1: per-head rank 1 vs 2 on the 1D ring (environment_cancel, p_plus 0.5).
A separate entry point; train_cancel.py is unchanged so the committed CANCEL runs still reproduce. Same recipe as CANCEL
(`run_cancel.sh`: T = 128, batch 128, 98 batches, lr 1e-3, cosine, 300 epochs, 1 layer, 2 heads, d 128); evaluation on a
fresh stream (np seed 10**6; the ring redraws its map per trajectory, so every eval map is unseen) at T and 4T.
Writes <arm>.pt and eval.json in --output-dir.
  python3 -m mapformer.train_ring_rank --arm R1 --seed 20 --device cuda:0 --output-dir runs/ring_rank/p0/R1_s20
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_cancel import GridWorldCancel
from mapformer.model import MapFormerWM
from mapformer.model_ring_rank import MapFormerWM_Ring_R1, MapFormerWM_Ring_R2, MapFormerWM_Ring_R1DF, MapFormerWM_Ring_R1AO
from mapformer.train import train
from mapformer.train_cancel import evaluate

ARMS = {"R1": MapFormerWM_Ring_R1, "R2": MapFormerWM_Ring_R2, "R1DF": MapFormerWM_Ring_R1DF,
        "R1AO": MapFormerWM_Ring_R1AO, "R2S": MapFormerWM}   # R2S = shared rank 2 (Vanilla): per-head rank 2, 2 latent dims
EVAL_SEED = 10 ** 6


def build(arm, env, size):
    kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=size)
    if arm in ("R1DF", "R1AO"):
        kw["action_ids"] = (env.action_offset + 0, env.action_offset + 1)     # +1, -1 (environment_cancel)
    if arm == "R2S":
        kw["bottleneck_r"] = 2
    return ARMS[arm](**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--p-plus", type=float, default=0.5)
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
    model = build(a.arm, env, a.size)
    assert len(model.layers) == 1
    print(f"{a.arm} p_plus={a.p_plus} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size, "n_obs_types": env.n_obs_types,
                           "p_empty": env.p_empty, "action_offset": env.action_offset, "obs_offset": env.obs_offset}},
               out / f"{a.arm}.pt")
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
