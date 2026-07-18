"""
Train/eval on the nested-room (hierarchical space) environment.

Loss = next-token CE masked to positions that are at least partially
predictable: revisits (retrievable) OR novel cells in an already-visited room
(theme-inferable). Evaluation reports the two regimes SEPARATELY from the same
trained model:

  revisit acc     : needle / retrieval        -> expect flat attention to win
  room-novel acc  : theme inference (spatial aggregate) -> expect hierarchy to win

Ceiling on room-novel is ~1/theme_size (0.333 for theme_size=3); blind chance
is 1/n_obs_types (0.0625). Any lift above ~0.06 means the model inferred the
room theme rather than merely retrieving.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapformer.environment_rooms import RoomsGridWorld
from mapformer.train_variant import VARIANT_MAP


def gen_batch(env, B, T, device):
    tk, rev, nov = [], [], []
    for _ in range(B):
        t_, _om, rm, rn = env.generate_trajectory(T)
        tk.append(t_); rev.append(rm); nov.append(rn)
    return (torch.stack(tk).to(device), torch.stack(rev).to(device),
            torch.stack(nov).to(device))


def evaluate(model, env, T, n, device):
    model.eval(); cr = tr_ = cn = tn = 0
    with torch.no_grad():
        for _ in range(n):
            tokens, rev, nov = gen_batch(env, 1, T, device)
            logits = model(tokens[:, :-1])
            pred = logits.argmax(-1)[0]
            tgt = tokens[0, 1:]
            mr, mn = rev[0, 1:], nov[0, 1:]
            if mr.any():
                cr += (pred[mr] == tgt[mr]).sum().item(); tr_ += mr.sum().item()
            if mn.any():
                cn += (pred[mn] == tgt[mn]).sum().item(); tn += mn.sum().item()
    return (cr / tr_ if tr_ else float('nan'), cn / tn if tn else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--rooms-per-side", type=int, default=8)
    ap.add_argument("--theme-size", type=int, default=3)
    ap.add_argument("--p-empty", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = RoomsGridWorld(size=64, n_obs_types=16, rooms_per_side=args.rooms_per_side,
                         theme_size=args.theme_size, p_empty=args.p_empty, seed=0)
    model = VARIANT_MAP[args.variant](vocab_size=env.unified_vocab_size, d_model=128,
                                      n_heads=2, n_layers=1, grid_size=64).to(args.device)
    print(f"{args.variant} rooms task rooms={args.rooms_per_side}^2 theme={args.theme_size} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0,
                                              total_iters=args.epochs * args.n_batches)
    losses = []
    for ep in range(args.epochs):
        model.train(); el = 0.0
        for _ in range(args.n_batches):
            tokens, rev, nov = gen_batch(env, args.batch_size, args.n_steps, args.device)
            logits = model(tokens[:, :-1])
            tgt = tokens[:, 1:]
            mask = (rev | nov)[:, 1:]
            loss = F.cross_entropy(logits[mask], tgt[mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); el += loss.item()
        losses.append(el / args.n_batches)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{args.epochs} | Loss: {losses[-1]:.4f}", flush=True)

    res = {}
    for T in args.eval_lens:
        a_rev, a_nov = evaluate(model, env, T, 100 if T <= 1024 else 30, args.device)
        res[T] = {"revisit": a_rev, "room_novel": a_nov}
        print(f"  T={T}: revisit={a_rev:.3f}  room_novel={a_nov:.3f}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses,
                "variant": args.variant, "rooms_acc": res}, out / f"{args.variant}_rooms.pt")
    print(f"Saved: {out}/{args.variant}_rooms.pt")


if __name__ == "__main__":
    main()
