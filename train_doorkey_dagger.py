"""DAgger (Dataset Aggregation) on MiniGrid-DoorKey.

Algorithm (Ross & Bagnell 2011):
  Round 0: BC on expert (BFS solver) trajectories — same as train_doorkey_bc.
  Round k>0:
    1. Roll out the current model closed-loop in the env, collecting visited
       states.
    2. For each visited state, query the BFS solver for the expert action.
    3. Add (state-prefix, expert_action) pairs to the aggregated dataset.
    4. Retrain (from current weights) on the aggregated dataset.
  After K rounds, eval closed-loop success.

Closing the BC distribution-shift gap. Each round teaches the model how to
recover from its own mistakes, because the visited-state distribution shifts
toward where the model actually goes.

Warm-starts from the round-0 BC checkpoint if `--init-ckpt` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.minigrid_env import MiniGridWorld
from mapformer.doorkey_solver import solve_doorkey
from mapformer.train_variant import VARIANT_MAP


def rollout_and_relabel(model, world: MiniGridWorld, n_episodes: int, max_steps: int,
                        device: str = "cuda"):
    """Run model closed-loop; at each visited state, ask the expert for the
    optimal action; emit (prefix_tokens, expert_action) training pairs.

    Returns a list of (tokens_tensor, action_target_mask_tensor) where the
    action_target_mask is True at positions whose NEXT TOKEN is the expert's
    suggestion at the corresponding state."""
    pairs = []
    model.eval()
    for _ in range(n_episodes):
        obs, info = world.env.reset(seed=world.seed + np.random.randint(1_000_000))
        prefix = []  # interleaved (a, o, a, o, ...) tokens so far
        expert_targets = []  # parallel list: expert action at each step
        for step in range(max_steps):
            # Ask the expert what to do FROM the current env state
            plan = solve_doorkey(world.env)
            if plan is None or len(plan) == 0:
                break
            expert_a = plan[0]

            # Forward the model on the prefix (if any) to pick its own action
            if len(prefix) > 0:
                with torch.no_grad():
                    inp = torch.tensor(prefix, device=device).unsqueeze(0)
                    logits = model(inp)
                    a_model = int(logits[0, -1].argmax().item()) % world.N_ACTIONS
            else:
                # No prefix yet → just use the expert's first move
                a_model = expert_a

            # Record the supervision target: expert action at this state
            expert_targets.append(expert_a)
            # Execute the MODEL's action (so we collect states the model
            # actually visits — the whole point of DAgger).
            prefix.append(a_model + world.action_offset)
            obs, r, term, trunc, info = world.env.step(a_model)
            prefix.append(world._front_cell_token(obs) + world.obs_offset)
            if term:
                break

        if len(expert_targets) == 0:
            continue

        # Build a single (tokens, mask) example for this episode where the
        # tokens are what the model SAW (its own actions + the resulting obs)
        # and the targets at each obs position is the EXPERT'S action.
        tokens = torch.tensor(prefix, dtype=torch.long)
        L = tokens.shape[0]
        am = torch.zeros(L, dtype=torch.bool)
        # action at flat index 2k, predicted from position 2k-1 (an obs).
        # We have len(expert_targets) total navigate steps; the action at step
        # k is at index 2k.
        # For the supervision: we want CE between prediction at position 2k-1
        # and expert_targets[k]. So we OVERWRITE tokens at position 2k with
        # expert_targets[k] so that tgt = tokens[1:] gives expert at the right
        # spot, then mask the corresponding pred positions.
        n_steps = len(expert_targets)
        for k in range(n_steps):
            tokens[2 * k] = expert_targets[k] + world.action_offset
            if 2 * k - 1 >= 0:
                am[2 * k - 1] = True
        pairs.append((tokens, am))
    return pairs


def collate(pairs, pad_token=0):
    Lmax = max(p[0].shape[0] for p in pairs)
    B = len(pairs)
    T = torch.full((B, Lmax), pad_token, dtype=torch.long)
    M = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, (t, m) in enumerate(pairs):
        T[i, :t.shape[0]] = t
        M[i, :m.shape[0]] = m
    return T, M


def expert_episode(world: MiniGridWorld, max_steps: int):
    """Pure expert trajectory: BFS solver dictates actions."""
    obs, info = world.env.reset(seed=world.seed + np.random.randint(1_000_000))
    plan = solve_doorkey(world.env)
    if plan is None: return None
    plan = plan[:max_steps]
    tokens = []
    for a in plan:
        tokens.append(a + world.action_offset)
        obs, r, term, trunc, info = world.env.step(a)
        tokens.append(world._front_cell_token(obs) + world.obs_offset)
        if term: break
    tokens = torch.tensor(tokens, dtype=torch.long)
    L = tokens.shape[0]
    am = torch.zeros(L, dtype=torch.bool); am[1::2] = True
    return tokens, am


def closed_loop_success(model, world: MiniGridWorld, n_trials: int, max_steps: int,
                        device: str = "cuda"):
    model.eval()
    succ = 0; total = 0
    with torch.no_grad():
        for _ in range(n_trials):
            obs, info = world.env.reset(seed=world.seed + np.random.randint(1_000_000))
            prefix = []
            for step in range(max_steps):
                if len(prefix) == 0:
                    a = 2  # forward as a fallback
                else:
                    inp = torch.tensor(prefix, device=device).unsqueeze(0)
                    logits = model(inp)
                    a = int(logits[0, -1].argmax().item()) % world.N_ACTIONS
                prefix.append(a + world.action_offset)
                obs, r, term, trunc, info = world.env.step(a)
                prefix.append(world._front_cell_token(obs) + world.obs_offset)
                if term:
                    if r > 0: succ += 1
                    total += 1
                    break
            else:
                total += 1
    return succ / max(1, total)


def train_one_round(model, dataset, n_epochs, lr, batch_size, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0,
                                              total_iters=n_epochs * (len(dataset) // batch_size + 1))
    losses = []
    for ep in range(n_epochs):
        np.random.shuffle(dataset)
        ep_loss = 0.0; ep_correct = 0; ep_total = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            T, M = collate(batch)
            T = T.to(device); M = M.to(device)
            inp = T[:, :-1]; tgt = T[:, 1:]; mask = M[:, :-1]
            logits = model(inp)
            lp = F.log_softmax(logits, dim=-1)
            if mask.sum() == 0: continue
            loss = F.nll_loss(lp[mask], tgt[mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item()
            ep_correct += (lp[mask].argmax(-1) == tgt[mask]).sum().item()
            ep_total += mask.sum().item()
        avg = ep_loss / max(1, len(dataset) // batch_size); acc = ep_correct / max(1, ep_total)
        losses.append(avg)
        print(f"    Inner ep {ep+1}/{n_epochs} | Loss {avg:.4f} | Acc {acc:.3f}")
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0")
    ap.add_argument("--tokenization", type=str, default="obj_color")
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--ep-per-round", type=int, default=512)
    ap.add_argument("--inner-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-ckpt", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    world = MiniGridWorld(env_name=args.env, tokenization=args.tokenization, seed=args.seed)
    cls = VARIANT_MAP[args.variant]
    model = cls(vocab_size=world.unified_vocab_size, d_model=128, n_heads=2,
                n_layers=1, grid_size=world.size).cuda()

    if args.init_ckpt:
        c = torch.load(args.init_ckpt, map_location="cuda", weights_only=False)
        own = model.state_dict()
        loaded = 0
        for k, v in c["model_state_dict"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded += 1
        model.load_state_dict(own)
        print(f"Warm-start: loaded {loaded} keys from {args.init_ckpt}")

    # Bootstrap dataset with expert trajectories
    print("Bootstrap with expert trajectories...")
    dataset = []
    for _ in range(args.ep_per_round):
        ep = expert_episode(world, args.max_steps)
        if ep is not None: dataset.append(ep)

    round_metrics = []
    for r in range(args.n_rounds):
        print(f"\n=== DAgger round {r+1}/{args.n_rounds} ===")
        # Train on aggregated dataset
        print(f"  Train on {len(dataset)} episodes...")
        train_one_round(model, dataset, args.inner_epochs, args.lr,
                        args.batch_size, "cuda")
        # Evaluate closed-loop
        world_eval = MiniGridWorld(env_name=args.env, tokenization=args.tokenization,
                                   seed=args.seed + 1000)
        succ = closed_loop_success(model, world_eval, 100, args.max_steps, "cuda")
        print(f"  Closed-loop success after round {r+1}: {succ:.3f}")
        round_metrics.append(succ)
        # Roll out + relabel; aggregate
        if r < args.n_rounds - 1:
            print(f"  Rollout + relabel {args.ep_per_round} episodes...")
            new_pairs = rollout_and_relabel(model, world, args.ep_per_round,
                                            args.max_steps, "cuda")
            print(f"    Got {len(new_pairs)} new pairs.")
            dataset.extend(new_pairs)

    ckpt = out / f"{args.variant}_dagger.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "round_success": round_metrics, "variant": args.variant,
                "seed": args.seed,
                "config": {"vocab_size": world.unified_vocab_size, "d_model": 128,
                           "n_heads": 2, "n_layers": 1, "grid_size": world.size,
                           "env": args.env, "tokenization": args.tokenization}},
               ckpt)
    print(f"\nSaved: {ckpt}")
    print(f"Round-by-round success: {round_metrics}")


if __name__ == "__main__":
    main()
