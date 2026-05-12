"""Successor representation (SR) readout from frozen prediction-trained models.

The strongest cognitive-map content claim: take a model that was ONLY trained
on prediction (no goal-directed supervision), freeze it, train a linear head
to predict discounted future cell-occupancy (the SR). Then at test time,
plant any reward function r at any cell, compute V = SR · r, derive a greedy
policy, roll out closed-loop.

If the prediction-trained representation supports SR readout, the cognitive
map is INTRINSICALLY navigable for arbitrary reward functions — no
goal-directed training of the backbone needed. This is the cleanest
"cognitive map enables flexible navigation" claim.

Reference: Dayan 1993 (SR), Stachenfeld 2017 (hippocampal place cells as SR).

Pipeline per variant:
  1. Load frozen prediction-trained lm200 checkpoint
  2. Generate N_TRAIN trajectories on a TRAIN env (random walks, record true positions)
  3. Capture hidden state at each step; compute Monte-Carlo SR targets per position
  4. Train linear Head: hidden -> SR target  (4096-dim output for 64×64 grid)
  5. Evaluate closed-loop: plant reward at random landmark, SR-greedy planning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_gsf_nodrop import MapFormerWM_Level15GSF_NoDrop
from mapformer.model_tem_faithful import TEMFaithful
from mapformer.model_baseline_rope import MapFormerWM_RoPE


VARIANT_CLS = {
    "RoPE":               MapFormerWM_RoPE,
    "Vanilla":            MapFormerWM,
    "Level15":            MapFormerWM_Level15InEKF,
    "Level15GSF_NoDrop":  MapFormerWM_Level15GSF_NoDrop,
    "TEMFaithful":        TEMFaithful,
}


class _HiddenCapture:
    def __init__(self): self.h = None
    def __call__(self, mod, inp, out): self.h = inp[0].detach()


def get_hidden(model, tokens):
    """Returns hidden state of shape (B, L, d_eff) for the standard MapFormer-WM
    family. For Level15GSF variants, takes mode 0 of the K-mode hidden.
    Returns None for variants we don't support here (e.g., TEMFaithful where
    out_proj is called per-step inside a Python loop)."""
    cap = _HiddenCapture()
    handle = model.out_proj.register_forward_hook(cap)
    with torch.no_grad():
        _ = model(tokens)
    handle.remove()
    h = cap.h
    if h is None:
        return None
    # GSF returns (B, K, L, d_model); take mode 0
    if h.dim() == 4:
        h = h[:, 0]
    return h


def supports_hidden(variant: str) -> bool:
    """Variants we can extract a clean per-position hidden state for via
    forward hook on out_proj. TEMFaithful calls out_proj inside a per-step
    loop, so the hook only captures the last step. Skip it for now."""
    return variant != "TEMFaithful"


def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg["grid_size"])
    if variant == "Level15GSF_NoDrop": kw["n_modes"] = 8
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval(), cfg


def compute_sr_target(env, x0, y0, gamma=0.9, n_samples=30, T_future=20, rng=None):
    """Monte-Carlo estimate the SR row from (x0, y0): discounted future-cell occupancy."""
    if rng is None: rng = np.random
    size = env.size
    sr = np.zeros(size * size, dtype=np.float32)
    for _ in range(n_samples):
        x, y = x0, y0
        for t in range(T_future):
            a = int(rng.randint(0, env.N_ACTIONS))
            dx, dy = env.ACTION_DELTAS[a]
            x = (x + dx) % size; y = (y + dy) % size
            sr[x * size + y] += (gamma ** t)
    return sr / n_samples


def collect_train_data(model, env, n_trajectories=200, T=128, device="cuda",
                       seed=0):
    """Generate trajectories on TRAIN env, capture (hidden, position) pairs."""
    rng = np.random.RandomState(seed)
    hiddens = []
    positions = []
    for traj in range(n_trajectories):
        tokens, _, _ = env.generate_trajectory(T)
        traj_positions = list(env.visited_locations)
        tt = tokens.unsqueeze(0).to(device)
        h = get_hidden(model, tt)  # (1, 2T-1 ish, d_model)
        # Hidden state at obs positions = positions 1, 3, 5, ...; agent is at traj_positions[t]
        # We want hidden right AFTER an obs token (so model has seen position t)
        # Token sequence: (a1, o1, a2, o2, ...). h has 2*T entries (one per token).
        # h[0, 2t+1] = hidden after observing o_{t+1} ≈ after being at position t.
        # Actually let me be careful: at the obs position 2t+1 in the input, the model
        # sees tokens 0..2t+1; the hidden at that position represents the encoded state
        # after seeing the obs. The position the agent is at is traj_positions[t].
        for t in range(min(T, len(traj_positions))):
            obs_pos_in_seq = 2 * t + 1
            if obs_pos_in_seq < h.shape[1]:
                hiddens.append(h[0, obs_pos_in_seq].cpu())
                positions.append(traj_positions[t])
    hiddens = torch.stack(hiddens)  # (N, d_model)
    return hiddens, positions


def train_sr_head(hiddens, sr_targets, n_epochs=200, lr=1e-2, device="cuda"):
    """Train Linear(d_model -> size*size) to predict SR from hidden state."""
    d = hiddens.shape[1]
    n_cells = sr_targets.shape[1]
    head = nn.Linear(d, n_cells).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    H = hiddens.to(device)
    T = sr_targets.to(device)
    for ep in range(n_epochs):
        logits = head(H)
        loss = F.mse_loss(logits, T)
        opt.zero_grad(); loss.backward(); opt.step()
    final_loss = loss.item()
    return head, final_loss


def sr_closed_loop_episode(model, env, sr_head, T_explore, T_navigate,
                           device, rng):
    """One SR-planning episode: random walk explore, then SR-greedy navigate."""
    size = env.size
    n_cells = size * size

    # Random start
    x = int(rng.randint(0, size)); y = int(rng.randint(0, size))

    # Random reward: one-hot at random cell
    rx = int(rng.randint(0, size)); ry = int(rng.randint(0, size))
    r = torch.zeros(n_cells, device=device)
    r[rx * size + ry] = 1.0

    context = []  # interleaved (a, o) tokens

    # Explore phase: random walks
    for _ in range(T_explore):
        a = int(rng.randint(0, env.N_ACTIONS))
        dx, dy = env.ACTION_DELTAS[a]
        x = (x + dx) % size; y = (y + dy) % size
        context.append(a + env.action_offset)
        context.append(int(env.obs_map[x, y].item()) + env.obs_offset)

    # Navigate phase: SR-greedy
    reached = False
    steps = T_navigate
    for t in range(T_navigate):
        values = []
        for a in range(env.N_ACTIONS):
            dx, dy = env.ACTION_DELTAS[a]
            nx = (x + dx) % size; ny = (y + dy) % size
            n_obs = int(env.obs_map[nx, ny].item())
            cand_context = context + [a + env.action_offset, n_obs + env.obs_offset]
            inp = torch.tensor(cand_context, device=device).unsqueeze(0)
            hidden = get_hidden(model, inp)[0, -1]
            sr = sr_head(hidden)
            v = (sr * r).sum().item()
            values.append(v)
        a_star = int(np.argmax(values))
        dx, dy = env.ACTION_DELTAS[a_star]
        x = (x + dx) % size; y = (y + dy) % size
        if (x, y) == (rx, ry):
            reached = True; steps = t + 1; break
        context.append(a_star + env.action_offset)
        context.append(int(env.obs_map[x, y].item()) + env.obs_offset)

    return reached, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="SR_PROBE_RESULTS.md")
    ap.add_argument("--n-train-trajs", type=int, default=100)
    ap.add_argument("--n-eval-episodes", type=int, default=100)
    ap.add_argument("--T-explore", type=int, default=64)
    ap.add_argument("--T-navigate", type=int, default=128)  # generous to allow reach
    ap.add_argument("--sr-samples", type=int, default=20)
    ap.add_argument("--sr-future", type=int, default=20)
    ap.add_argument("--sr-gamma", type=float, default=0.92)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    # Build train env (different from eval env)
    env_train = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                          n_landmarks=200, seed=0)  # train env
    env_test = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                         n_landmarks=200, seed=1000)  # held-out env

    print(f"Train env vocab: {env_train.unified_vocab_size}, grid: {env_train.size}")
    print(f"Test env vocab: {env_test.unified_vocab_size}, grid: {env_test.size}")

    results = {}
    # Skip TEMFaithful: out_proj is called per-step in a Python loop, so a
    # single forward hook can't capture per-position hiddens cleanly. Would
    # need a custom extraction.
    for variant in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop"]:
        if not supports_hidden(variant): continue
        successes = []
        head_losses = []
        for s in args.seeds:
            ckpt = Path(f"mapformer/runs/{variant}_lm200/seed{s}/{variant}.pt")
            if not ckpt.exists():
                print(f"  skip {variant} s{s}: no ckpt")
                continue
            try:
                model, cfg = build(variant, ckpt)
            except Exception as e:
                print(f"  skip {variant} s{s}: {e}")
                continue

            # 1. Collect (hidden, position) pairs from TRAIN env
            print(f"\n=== {variant} seed {s}: collecting training data ===")
            hiddens, positions = collect_train_data(
                model, env_train,
                n_trajectories=args.n_train_trajs, T=128,
                device="cuda", seed=s,
            )
            print(f"  collected {hiddens.shape[0]} (hidden, position) pairs")

            # 2. Compute SR targets via Monte Carlo
            print(f"  computing SR targets ({args.sr_samples} samples × {args.sr_future} steps)...")
            sr_rng = np.random.RandomState(1000 + s)
            sr_targets = np.array([
                compute_sr_target(env_train, p[0], p[1],
                                  gamma=args.sr_gamma,
                                  n_samples=args.sr_samples,
                                  T_future=args.sr_future, rng=sr_rng)
                for p in positions
            ])
            sr_targets = torch.from_numpy(sr_targets)
            print(f"  SR target shape: {sr_targets.shape}, mean magnitude: {sr_targets.mean():.4f}")

            # 3. Train linear SR head
            print(f"  training SR head...")
            sr_head, head_loss = train_sr_head(hiddens, sr_targets, n_epochs=300, lr=1e-2)
            head_losses.append(head_loss)
            print(f"  final SR-head MSE: {head_loss:.5f}")

            # 4. Evaluate closed-loop SR-planning on TEST env
            print(f"  evaluating closed-loop SR planning on held-out env...")
            eval_rng = np.random.RandomState(2000 + s)
            n_success = 0
            n_total = 0
            for ep in range(args.n_eval_episodes):
                ok, steps = sr_closed_loop_episode(
                    model, env_test, sr_head,
                    T_explore=args.T_explore, T_navigate=args.T_navigate,
                    device="cuda", rng=eval_rng,
                )
                n_success += int(ok)
                n_total += 1
            sr = n_success / max(1, n_total)
            successes.append(sr)
            print(f"  closed-loop SR planning success: {sr:.3f}")
            del model, sr_head; torch.cuda.empty_cache()

        if successes:
            results[variant] = {
                "success_mean": float(np.mean(successes)),
                "success_std": float(np.std(successes)),
                "head_loss": float(np.mean(head_losses)) if head_losses else None,
                "n_seeds": len(successes),
            }

    # Markdown report
    lines = []
    lines.append("# Successor representation (SR) readout — reward-conditional planning from frozen models\n")
    lines.append("Take prediction-trained lm200 checkpoints. Freeze the backbone. Train a")
    lines.append("linear head to predict discounted future-cell occupancy (the SR) from the")
    lines.append("model's hidden state. At test time, plant a one-hot reward at a random cell,")
    lines.append("compute V = SR · r, derive greedy policy via one-step lookahead, roll out.\n")
    lines.append("**This is the strongest cognitive-map content claim possible:** the prediction-")
    lines.append("trained representation supports navigation to ARBITRARY rewards without any")
    lines.append("goal-directed training of the backbone.\n")
    lines.append(f"Settings: γ={args.sr_gamma}, future={args.sr_future} steps, {args.sr_samples} MC samples per state.")
    lines.append(f"Eval: {args.n_eval_episodes} episodes, T_explore={args.T_explore}, T_navigate={args.T_navigate}.")
    lines.append(f"Chance success ≈ {1 / (64 * 64):.4f} (random cell, random walk → goal).\n")

    lines.append("## Closed-loop SR-planning success rate\n")
    lines.append(f"| Variant | success rate (mean ± std) | head-train MSE | n_seeds |")
    lines.append("|---|---|---|---|")
    for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop"]:
        if v not in results:
            lines.append(f"| {v} | — | — | 0 |"); continue
        r = results[v]
        sd = r["success_std"]
        loss = r["head_loss"]
        lines.append(f"| **{v}** | {r['success_mean']:.3f} ± {sd:.3f} | {loss:.5f} | {r['n_seeds']} |")
    lines.append("")
    lines.append("(TEMFaithful skipped: out_proj is called per-step inside its forward loop,")
    lines.append("so a single forward hook can't capture per-position hiddens cleanly. Would")
    lines.append("require a custom extraction. Defer for now.)\n")

    lines.append("## Interpretation\n")
    lines.append("- Chance ≈ 0 (random walks rarely hit a specific cell within T_navigate)")
    lines.append("- If RoPE near chance: the representation has no useful position-encoding for SR readout")
    lines.append("- If Vanilla < Level15*: same content gap as in the linear-probe-for-action result")
    lines.append("- If Level15* matches/beats TEMFaithful: cognitive-map content is sufficient for reward-conditional planning\n")
    lines.append("*Auto-generated by probe_sr.py*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n--- Result file written: " + args.output_md)
    print("\n".join(lines[-25:]))


if __name__ == "__main__":
    main()
