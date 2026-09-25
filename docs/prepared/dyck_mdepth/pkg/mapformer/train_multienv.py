"""Train any variant on multi-environment data (TEM-style cognitive-map test).

Each batch trajectory is sampled from a randomly-chosen training env.
Eval uses held-out test envs the model has never seen. Tests whether the
model has learned a META-strategy for cognitive-map building, rather than
memorising per-env content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment_multienv import MultiEnvGridWorld
from mapformer.train_variant import VARIANT_MAP


def train(model, world, n_epochs, lr, batch_size, n_steps, n_batches, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=n_epochs * n_batches)
    losses = []
    rng = np.random.RandomState(0)
    for ep in range(n_epochs):
        ep_loss = 0.0
        for b in range(n_batches):
            tokens, _, rm, _ = world.generate_batch(batch_size, n_steps, train=True, rng=rng)
            tokens = tokens.to(device); rm = rm.to(device)
            inp = tokens[:, :-1]; tgt = tokens[:, 1:]; mask = rm[:, 1:]
            logits = model(inp)
            lp = F.log_softmax(logits, dim=-1)
            tgt_flat = tgt[mask]; lp_flat = lp[mask]
            if tgt_flat.numel() == 0: continue
            loss = F.nll_loss(lp_flat, tgt_flat)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item()
        avg = ep_loss / max(1, n_batches); losses.append(avg)
        print(f"  Epoch {ep+1:3d}/{n_epochs} | Loss: {avg:.4f} | LR: {sched.get_last_lr()[0]:.2e}")
    return losses


def evaluate(model, world, T, n_trials, device, seed=2000, train_envs=False):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm, _ = world.generate_trajectory(T, train=train_envs, rng=rng)
            tt = tokens.unsqueeze(0).to(device)
            try: logits = model(tt[:, :-1])
            except Exception: return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].to(device)
            if mask.sum() == 0: continue
            correct += (preds[mask] == tgts[mask]).sum().item()
            total += mask.sum().item()
            idx = torch.arange(lp.shape[1], device=device)[mask]
            nll += -lp[0, idx, tgts[mask]].sum().item()
    return (correct / total if total else None, nll / total if total else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-landmarks", type=int, default=200)
    ap.add_argument("--n-train-envs", type=int, default=50)
    ap.add_argument("--n-test-envs", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    world = MultiEnvGridWorld(
        size=64, n_obs_types=16, p_empty=0.5, n_landmarks=args.n_landmarks,
        n_train_envs=args.n_train_envs, n_test_envs=args.n_test_envs,
        seed=args.seed,
    )
    cls = VARIANT_MAP[args.variant]
    model = cls(vocab_size=world.unified_vocab_size, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers, grid_size=64)
    print(f"{args.variant} seed={args.seed} n_train_envs={args.n_train_envs} "
          f"n_test_envs={args.n_test_envs} lm={args.n_landmarks}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    losses = train(model, world, args.epochs, args.lr, args.batch_size,
                   args.n_steps, args.n_batches, args.device)

    # Eval on train envs AND held-out test envs
    acc_train, nll_train = evaluate(model, world, args.n_steps, 100, args.device,
                                     seed=2000, train_envs=True)
    acc_test, nll_test = evaluate(model, world, args.n_steps, 100, args.device,
                                   seed=2000, train_envs=False)
    acc_test_T2, nll_test_T2 = evaluate(model, world, args.n_steps * 4, 50, args.device,
                                         seed=2000, train_envs=False)
    print(f"\nTrain envs: acc={acc_train:.3f} nll={nll_train:.3f}")
    print(f"Held-out envs at T={args.n_steps}: acc={acc_test:.3f} nll={nll_test:.3f}")
    print(f"Held-out envs at T={args.n_steps*4}: acc={acc_test_T2:.3f} nll={nll_test_T2:.3f}")

    ckpt = out / f"{args.variant}_multienv.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "losses": losses,
        "variant": args.variant, "seed": args.seed,
        "train_acc": acc_train, "test_acc": acc_test, "test_acc_T2": acc_test_T2,
        "train_nll": nll_train, "test_nll": nll_test, "test_nll_T2": nll_test_T2,
        "config": {"vocab_size": world.unified_vocab_size,
                   "d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "grid_size": 64,
                   "n_obs_types": 16, "p_empty": 0.5,
                   "n_landmarks": args.n_landmarks,
                   "n_train_envs": args.n_train_envs,
                   "n_test_envs": args.n_test_envs},
    }, ckpt)
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
