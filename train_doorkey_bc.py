"""Behavioural cloning on MiniGrid-DoorKey solver trajectories.

Generates expert trajectories using `doorkey_solver.solve_doorkey`, trains
the model to predict the expert's action at each step. Same model classes
as standard training (VARIANT_MAP from train_variant.py), but loss is on
action prediction with mask aligned to action positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.minigrid_env import MiniGridWorld
from mapformer.doorkey_solver import solve_doorkey
from mapformer.train_variant import VARIANT_MAP


def generate_solver_episode(world: MiniGridWorld, max_steps: int = 64):
    """Reset env, solve, execute. Returns (tokens, action_target_mask).

    tokens layout: interleaved (a_0, o_0, a_1, o_1, ..., a_T, o_T).
    Action positions: 0, 2, 4, ... Obs positions: 1, 3, 5, ...
    action_target_mask True at obs position k-1 predicting action at k.
    For position 0 there's no prefix, we skip that supervision target.
    """
    obs, info = world.env.reset(seed=world.seed + np.random.randint(1_000_000))
    plan = solve_doorkey(world.env)
    if plan is None or len(plan) == 0:
        return None, None
    plan = plan[:max_steps]
    tokens = []
    for a in plan:
        tokens.append(a + world.action_offset)
        obs, r, term, trunc, info = world.env.step(a)
        front = world._front_cell_token(obs)
        tokens.append(front + world.obs_offset)
        if term: break
    tokens = torch.tensor(tokens, dtype=torch.long)
    L = tokens.shape[0]
    am = torch.zeros(L, dtype=torch.bool)
    # supervise action prediction at every action position except the first.
    # action token at index 2k (k=0..); predicted from position 2k-1.
    am[1::2] = True
    # Drop the very last position since it predicts something that doesn't exist.
    # We'll just shift via tokens[:-1] / tokens[1:] in the loss.
    return tokens, am


def gen_batch(world: MiniGridWorld, batch_size: int, max_steps: int = 64):
    toks, ams = [], []
    Lmax = 0
    for _ in range(batch_size):
        t, am = generate_solver_episode(world, max_steps=max_steps)
        if t is None: continue
        toks.append(t); ams.append(am); Lmax = max(Lmax, t.shape[0])
    # Pad with zeros; mask out padded positions.
    B = len(toks)
    padded_t = torch.zeros(B, Lmax, dtype=torch.long)
    padded_m = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, (t, m) in enumerate(zip(toks, ams)):
        padded_t[i, :t.shape[0]] = t
        padded_m[i, :m.shape[0]] = m
    return padded_t, padded_m


def train(model, world, n_epochs=30, lr=3e-4, batch_size=64, n_batches=64,
          max_steps=64, device="cuda"):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=n_epochs * n_batches)
    losses = []
    for ep in range(n_epochs):
        ep_loss = 0.0; ep_correct = 0; ep_total = 0
        for b in range(n_batches):
            tokens, am = gen_batch(world, batch_size, max_steps=max_steps)
            tokens = tokens.to(device); am = am.to(device)
            inp = tokens[:, :-1]; tgt = tokens[:, 1:]; mask = am[:, :-1]
            logits = model(inp)
            lp = F.log_softmax(logits, dim=-1)
            tgt_flat = tgt[mask]; lp_flat = lp[mask]
            if tgt_flat.numel() == 0: continue
            loss = F.nll_loss(lp_flat, tgt_flat)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item()
            preds = lp_flat.argmax(-1)
            ep_correct += (preds == tgt_flat).sum().item()
            ep_total += tgt_flat.numel()
        avg = ep_loss / max(1, n_batches); acc = ep_correct / max(1, ep_total)
        losses.append(avg)
        print(f"  Epoch {ep+1:3d}/{n_epochs} | Loss: {avg:.4f} | Acc: {acc:.3f} | LR: {sched.get_last_lr()[0]:.2e}")
    return losses


def evaluate(model, world, n_trials=200, max_steps=64, device="cuda"):
    """Action-match accuracy + closed-loop success rate."""
    model.eval()
    match_correct = 0; match_total = 0
    rollout_success = 0; rollout_total = 0
    with torch.no_grad():
        for _ in range(n_trials):
            obs, info = world.env.reset(seed=world.seed + np.random.randint(1_000_000))
            plan = solve_doorkey(world.env)
            if plan is None: continue
            # Closed-loop rollout: model picks actions, env executes
            history = []
            for step in range(max_steps):
                if len(history) == 0:
                    # No history yet: just use the optimal first action
                    # (we have no input to feed); skip supervised match here.
                    a_model = plan[0] if plan else 0
                else:
                    inp = torch.tensor(history, device=device).unsqueeze(0)
                    logits = model(inp)
                    a_model = int(logits[0, -1].argmax().item())
                    # Restrict to action vocab
                    if a_model >= world.action_offset + world.N_ACTIONS:
                        a_model = 0
                obs, r, term, trunc, info = world.env.step(a_model % world.N_ACTIONS)
                front = world._front_cell_token(obs)
                history.append(a_model % world.N_ACTIONS + world.action_offset)
                history.append(front + world.obs_offset)
                if term:
                    if r > 0: rollout_success += 1
                    rollout_total += 1
                    break
            else:
                rollout_total += 1
            # Action-match: compare model action at each step to solver
            # (open-loop, on expert trajectory)
            obs, info = world.env.reset(seed=info.get("seed", 0) if isinstance(info, dict) else 0)
            plan2 = solve_doorkey(world.env)
            if plan2:
                tokens = []
                for a in plan2[:max_steps]:
                    if tokens:
                        inp = torch.tensor(tokens, device=device).unsqueeze(0)
                        logits = model(inp)
                        a_pred = int(logits[0, -1].argmax().item())
                        a_pred_action = a_pred % world.N_ACTIONS
                        match_correct += (a_pred_action == a)
                        match_total += 1
                    tokens.append(a + world.action_offset)
                    obs, r, term, trunc, info = world.env.step(a)
                    front = world._front_cell_token(obs)
                    tokens.append(front + world.obs_offset)
                    if term: break
    match_acc = match_correct / max(1, match_total)
    success = rollout_success / max(1, rollout_total)
    return match_acc, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--tokenization", type=str, default="obj_color")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    world = MiniGridWorld(env_name=args.env, tokenization=args.tokenization, seed=args.seed)
    cls = VARIANT_MAP[args.variant]
    model = cls(vocab_size=world.unified_vocab_size, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers, grid_size=world.size)
    print(f"{args.variant} seed={args.seed} env={args.env}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}, vocab={world.unified_vocab_size}, grid={world.size}")

    losses = train(model, world, n_epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, n_batches=args.n_batches,
                   max_steps=args.max_steps, device=args.device)

    # Eval
    world_test = MiniGridWorld(env_name=args.env, tokenization=args.tokenization,
                               seed=args.seed + 1000)
    match_acc, success = evaluate(model, world_test, n_trials=100,
                                  max_steps=args.max_steps, device=args.device)
    print(f"Held-out match-acc: {match_acc:.3f} | rollout-success: {success:.3f}")

    ckpt_path = out / f"{args.variant}_dkbc.pt"
    torch.save({"model_state_dict": model.state_dict(), "losses": losses,
                "variant": args.variant, "seed": args.seed,
                "match_acc": match_acc, "success_rate": success,
                "config": {"vocab_size": world.unified_vocab_size,
                           "d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "grid_size": world.size,
                           "env": args.env, "tokenization": args.tokenization}}, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
