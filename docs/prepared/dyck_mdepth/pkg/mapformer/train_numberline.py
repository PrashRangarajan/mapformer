"""Train a variant on NumberLineWorld — arithmetic as navigation.

Trains, then evaluates at the training chain length and at 4x OOD chain
length (the key test: does the path-integration accumulator extrapolate
to longer arithmetic chains?). Saves a checkpoint with eval_results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.number_line_env import NumberLineWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train import train as train_loop


@torch.no_grad()
def evaluate(model, env, T, n_trials, device, seed=2000):
    """Accuracy + NLL on revisited-value positions of held-out trajectories."""
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll = 0.0
    for _ in range(n_trials):
        tokens, _, rm = env.generate_trajectory(T)
        tt = tokens.unsqueeze(0).to(device)
        logits = model(tt[:, :-1])
        lp = F.log_softmax(logits, dim=-1)
        preds = lp.argmax(-1)[0]
        tgts = tt[0, 1:]
        mask = rm[1:].to(device)
        if mask.sum() == 0: continue
        correct += (preds[mask] == tgts[mask]).sum().item()
        total += mask.sum().item()
        idx = torch.arange(lp.shape[1], device=device)[mask]
        nll += -lp[0, idx, tgts[mask]].sum().item()
    return (correct / total if total else None,
            nll / total if total else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=64, help="N — number-line modulus.")
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--n-landmarks", type=int, default=0)
    ap.add_argument("--p-empty", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128, help="train chain length (ops).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env = NumberLineWorld(size=args.size, n_obs_types=args.n_obs_types,
                          p_empty=args.p_empty, n_landmarks=args.n_landmarks,
                          seed=args.seed)
    cls = VARIANT_MAP[args.variant]
    model = cls(vocab_size=env.unified_vocab_size, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers,
                grid_size=args.size)
    print(f"{args.variant} seed={args.seed} N={args.size} "
          f"lm={args.n_landmarks} params={sum(p.numel() for p in model.parameters()):,}")

    losses = train_loop(model, env, n_epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, n_steps=args.n_steps,
                        n_batches=args.n_batches, device=args.device,
                        verbose=True)

    model = model.to(args.device).eval()
    acc_t1, nll_t1 = evaluate(model, env, args.n_steps, 100, args.device, seed=2000)
    acc_t4, nll_t4 = evaluate(model, env, args.n_steps * 4, 50, args.device, seed=2001)
    print(f"\nIn-dist  T={args.n_steps}:  acc={acc_t1:.3f} nll={nll_t1:.3f}")
    print(f"OOD chain T={args.n_steps*4}: acc={acc_t4:.3f} nll={nll_t4:.3f}")

    ckpt = out / f"{args.variant}_numberline.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "losses": losses,
        "variant": args.variant, "seed": args.seed,
        "test_acc": acc_t1, "test_nll": nll_t1,
        "test_acc_T2": acc_t4, "test_nll_T2": nll_t4,
        "config": {"vocab_size": env.unified_vocab_size,
                   "d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "grid_size": args.size,
                   "n_obs_types": args.n_obs_types, "p_empty": args.p_empty,
                   "n_landmarks": args.n_landmarks, "size": args.size},
    }, ckpt)
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
