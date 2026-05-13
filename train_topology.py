"""Train any variant on multi-topology data: torus + open + walls grids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment_topology import MultiTopologyGridWorld
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


def evaluate_topology(model, world, topo, T, n_trials, device, seed=2000, train_envs=False):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm, _ = world.generate_trajectory(T, train=train_envs, topology=topo, rng=rng)
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
    ap.add_argument("--topologies", nargs="+", default=["torus", "open", "walls"])
    ap.add_argument("--n-envs-per-topology", type=int, default=20)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-landmarks", type=int, default=200)
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

    world = MultiTopologyGridWorld(
        topologies=tuple(args.topologies),
        size=args.size, n_obs_types=16, p_empty=0.5,
        n_landmarks=args.n_landmarks,
        n_envs_per_topology=args.n_envs_per_topology,
        n_test_envs_per_topology=args.n_envs_per_topology,
        seed=args.seed,
    )
    cls = VARIANT_MAP[args.variant]
    model = cls(vocab_size=world.unified_vocab_size, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers, grid_size=args.size)
    print(f"{args.variant} seed={args.seed} topos={args.topologies} "
          f"n_train={len(args.topologies) * args.n_envs_per_topology}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    losses = train(model, world, args.epochs, args.lr, args.batch_size,
                   args.n_steps, args.n_batches, args.device)

    eval_results = {}
    for topo in args.topologies:
        print(f"\n--- Eval on {topo} ---")
        acc_128, nll_128 = evaluate_topology(model, world, topo, args.n_steps, 100,
                                              args.device, seed=2000, train_envs=False)
        acc_512, nll_512 = evaluate_topology(model, world, topo, args.n_steps * 4, 50,
                                              args.device, seed=2000, train_envs=False)
        eval_results[topo] = {
            "held_T128_acc": acc_128, "held_T128_nll": nll_128,
            "held_T512_acc": acc_512, "held_T512_nll": nll_512,
        }
        print(f"  {topo}: T=128 acc={acc_128:.3f} nll={nll_128:.3f}")
        print(f"  {topo}: T=512 acc={acc_512:.3f} nll={nll_512:.3f}")

    ckpt = out / f"{args.variant}_topology.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "variant": args.variant, "seed": args.seed,
        "eval_results": eval_results,
        "config": {"vocab_size": world.unified_vocab_size,
                   "d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "grid_size": args.size,
                   "n_obs_types": 16, "p_empty": 0.5,
                   "n_landmarks": args.n_landmarks,
                   "topologies": args.topologies,
                   "n_envs_per_topology": args.n_envs_per_topology},
    }, ckpt)
    print(f"\nSaved: {ckpt}")


if __name__ == "__main__":
    main()
