"""
Train/eval hierarchical goal-directed navigation (rooms + distant goals).

Loss: CE on the next token at positions whose next token is a BFS-optimal
navigate action. Chance = 1/N_ACTIONS = 0.25.

Eval reports action accuracy BUCKETED BY ROOM DISTANCE (how many room-hops to
the goal). The falsifiable prediction is that a hierarchical model's advantage
GROWS with room distance, because that is where flat planning degrades and
coarse (room-level) routing pays off.
"""
from __future__ import annotations
import argparse, sys, collections
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapformer.environment_rooms_goal import RoomsGoalWorld
from mapformer.environment_rooms_maze import RoomsMazeWorld
from mapformer.train_variant import VARIANT_MAP


def gen_batch(env, B, Te, Tn, rng, device):
    tk, am = [], []
    for _ in range(B):
        t_, a_, _ = env.generate_goal_episode(Te, Tn, rng)
        tk.append(t_); am.append(a_)
    return torch.stack(tk).to(device), torch.stack(am).to(device)


def evaluate(model, env, Te, Tn, n, device, seed=1234):
    model.eval()
    rng = np.random.RandomState(seed)
    per = collections.defaultdict(lambda: [0, 0])   # rdist -> [correct, total]
    with torch.no_grad():
        for _ in range(n):
            t_, a_, info = env.generate_goal_episode(Te, Tn, rng)
            tokens = t_.unsqueeze(0).to(device); am = a_.unsqueeze(0).to(device)
            logits = model(tokens[:, :-1])
            pred = logits.argmax(-1)[0]
            tgt = tokens[0, 1:]
            m = am[0, :-1]
            if m.sum() == 0:
                continue
            c = (pred[m] == tgt[m]).sum().item(); n_ = m.sum().item()
            d = info["room_distance"]
            per[d][0] += c; per[d][1] += n_
    out = {d: (v[0] / v[1]) for d, v in sorted(per.items()) if v[1] > 0}
    tot_c = sum(v[0] for v in per.values()); tot_n = sum(v[1] for v in per.values())
    out["all"] = tot_c / tot_n if tot_n else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rooms-per-side", type=int, default=8)
    ap.add_argument("--theme-size", type=int, default=3)
    ap.add_argument("--T-explore", type=int, default=64)
    ap.add_argument("--T-navigate", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eval-explore", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--env-kind", choices=["open", "maze"], default="open",
                    help="open = no walls (greedy-solvable, non-test); maze = walls+doors")
    ap.add_argument("--connectivity", choices=["full", "tree"], default="tree")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.env_kind == "maze":
        env = RoomsMazeWorld(size=64, n_obs_types=16, rooms_per_side=args.rooms_per_side,
                             theme_size=args.theme_size,
                             connectivity=args.connectivity, seed=0)
    else:
        env = RoomsGoalWorld(size=64, n_obs_types=16, rooms_per_side=args.rooms_per_side,
                             theme_size=args.theme_size, p_empty=0.0, seed=0)
    model = VARIANT_MAP[args.variant](vocab_size=env.unified_vocab_size, d_model=128,
                                      n_heads=2, n_layers=1, grid_size=64).to(args.device)
    print(f"{args.variant} rooms-goal Te={args.T_explore} Tn={args.T_navigate} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0,
                                              total_iters=args.epochs * args.n_batches)
    rng = np.random.RandomState(args.seed)
    losses = []
    for ep in range(args.epochs):
        model.train(); el = 0.0
        for _ in range(args.n_batches):
            tokens, am = gen_batch(env, args.batch_size, args.T_explore, args.T_navigate,
                                   rng, args.device)
            logits = model(tokens[:, :-1])
            tgt = tokens[:, 1:]; m = am[:, :-1]
            loss = F.cross_entropy(logits[m], tgt[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); el += loss.item()
        losses.append(el / args.n_batches)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{args.epochs} | Loss: {losses[-1]:.4f}", flush=True)

    res = {}
    for Te in args.eval_explore:
        r = evaluate(model, env, Te, args.T_navigate, 200, args.device)
        res[Te] = r
        pretty = " ".join(f"d{k}={v:.3f}" for k, v in r.items())
        print(f"  T_explore={Te}: {pretty}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses,
                "variant": args.variant, "goal_acc": res}, out / f"{args.variant}_rgoal.pt")
    print(f"Saved: {out}/{args.variant}_rgoal.pt")


if __name__ == "__main__":
    main()
