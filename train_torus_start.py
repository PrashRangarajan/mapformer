"""Trainer for TORUS_START_PREREG.md. Torus (environment.GridWorld 64x64), the GENERALITY / SIGN_MATCHED recipe: T = 1024
moves, batch 16, 98 batches, lr 1e-3, cosine 5% warm-up, 900 epochs, 1 layer, 2 heads, d 128, --n-landmarks 0, 3 data
workers, SDPA attention with TF32 off (train_variant's --fast-attn, applied the same way). A separate entry point so
train_variant.py (checksummed by a running batch) is not edited. Evaluation: eval_noise_refine.evaluate on the held-out map
(env seed 10000) at T = 1024 and 2048 (100 trials, seed 1234 + s, as eval_noise_refine's main), and eval_rank_strata.evaluate
(T = 1024, 100 trials). Writes <arm>.pt and eval.json.
  python3 -m mapformer.train_torus_start --arm AO4_high --seed 1200 --device cuda:0 --output-dir runs/torus_start/p0/AO4_high_s1200
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import mapformer.model as _M
from mapformer.environment import GridWorld
from mapformer.eval_noise_refine import evaluate as eval_acc
from mapformer.eval_rank_strata import evaluate as eval_strata
from mapformer.model_torus_start import ARMS
from mapformer.model_wrap_exist import wrap_share, action_angles
from mapformer.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    _M.USE_SDPA = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=a.seed)
    model = ARMS[a.arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    assert len(model.layers) == 1
    pairs = ((0, 1), (2, 3))
    init_angles = action_angles(model)
    print(f"{a.arm} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,} init angle err "
          f"{getattr(model, 'init_angle_err', None)} init |angle| mean {float(init_angles.abs().mean()):.3f} init wrap "
          f"{wrap_share(model, pairs=pairs)} |W_out| max {float(model.action_to_lie.w_out.weight.abs().max()):.3f}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size, "d_model": 128, "n_heads": 2,
                           "n_layers": 1, "grid_size": 64, "n_obs_types": 16, "p_empty": 0.5, "n_landmarks": 0,
                           "n_steps": a.n_steps}}, out / f"{a.arm}.pt")
    model.eval()
    te = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=10000)
    ev = {}
    for T in (1024, 2048):
        acc, nll = eval_acc(model, te, T, a.n_trials, 0.0, a.device, seed=1234 + a.seed)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"held-out map T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)
    st = eval_strata(model, te, 1024, a.n_trials, 1234 + a.seed, a.device)
    model.cpu()
    json.dump({"eval": ev, "strata": st, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):])),
               "wrap": wrap_share(model, pairs=pairs), "init_angles": init_angles.tolist(),
               "init_angle_err": getattr(model, "init_angle_err", None)}, open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
