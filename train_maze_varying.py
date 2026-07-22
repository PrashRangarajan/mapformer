"""
Train/eval on the VARYING maze: build a cognitive map, then plan on it.

A fresh maze + fresh landmark layout every episode, so memorisation is
impossible by construction (the fixed-maze version collapsed 0.94 -> 0.68 on a
novel maze, proving it had memorised rather than planned).

Loss: CE on the next token at positions whose next token is a BFS-optimal
navigate action. Chance = 0.25. A greedy wall-ignoring policy scores ~0.73, so
ANY score meaningfully above 0.73 means the model is using the map it built
during exploration.

Eval buckets by BFS path length (planning depth). The falsifiable prediction
for hierarchy: its advantage should GROW with path length.
"""
from __future__ import annotations
import argparse, sys, collections
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapformer.environment_maze_varying import VaryingMazeWorld
from mapformer.train_variant import VARIANT_MAP


def evaluate(model, env, Te, Tn, n, device, seed=4321):
    model.eval()
    rng = np.random.RandomState(seed)
    per = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for _ in range(n):
            r = env.generate_episode(Te, Tn, rng)
            if r is None:
                continue
            tk, am, info = r
            tok = tk.unsqueeze(0).to(device)
            logits = model(tok[:, :-1])
            pred = logits.argmax(-1)[0]
            tgt = tok[0, 1:]
            m = am[:-1].to(device)
            if m.sum() == 0:
                continue
            c = (pred[m] == tgt[m]).sum().item()
            L = info["bfs_len"]
            b = "1-5" if L <= 5 else ("6-10" if L <= 10 else ("11-15" if L <= 15 else "16+"))
            per[b][0] += c; per[b][1] += m.sum().item()
    out = {k: v[0] / v[1] for k, v in per.items() if v[1] > 0}
    tc = sum(v[0] for v in per.values()); tn = sum(v[1] for v in per.values())
    out["all"] = tc / tn if tn else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--rooms-per-side", type=int, default=4)
    ap.add_argument("--n-obs-types", type=int, default=8)
    ap.add_argument("--n-landmarks", type=int, default=24)
    ap.add_argument("--T-explore", type=int, default=256)
    ap.add_argument("--T-navigate", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = VaryingMazeWorld(size=args.size, rooms_per_side=args.rooms_per_side,
                           n_obs_types=args.n_obs_types,
                           n_landmarks=args.n_landmarks, seed=args.seed)
    model = VARIANT_MAP[args.variant](vocab_size=env.unified_vocab_size, d_model=128,
                                      n_heads=2, n_layers=1,
                                      grid_size=args.size).to(args.device)
    print(f"{args.variant} varying-maze size={args.size} Te={args.T_explore} "
          f"Tn={args.T_navigate} params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0,
                                              total_iters=args.epochs * args.n_batches)
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        model.train(); el = 0.0
        for _ in range(args.n_batches):
            tk, am = env.generate_batch(args.batch_size, args.T_explore,
                                        args.T_navigate, rng)
            tk = tk.to(args.device); am = am.to(args.device)
            logits = model(tk[:, :-1])
            tgt = tk[:, 1:]; m = am[:, :-1]
            loss = F.cross_entropy(logits[m], tgt[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); el += loss.item()
        losses.append(el / args.n_batches)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{args.epochs} | Loss: {losses[-1]:.4f}", flush=True)

    acc = evaluate(model, env, args.T_explore, args.T_navigate, 300, args.device)
    print("  " + " ".join(f"{k}={v:.3f}" for k, v in sorted(acc.items())))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses,
                "variant": args.variant, "acc": acc}, out / f"{args.variant}_vmaze.pt")
    print(f"Saved: {out}/{args.variant}_vmaze.pt")


if __name__ == "__main__":
    main()
