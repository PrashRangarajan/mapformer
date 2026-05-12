"""Closed-loop goal-directed evaluation on torus.

Takes goal-directed BC-trained checkpoints (from Cluster D experiment 8) and
evaluates them in closed-loop:

1. Reset env to random start, sample random goal landmark
2. Explore phase: random walks for T_explore steps (env tracks true position)
3. Navigate phase: model picks actions, env executes, agent position updates
4. Success = agent reaches goal cell at any time during navigate

No new training. Same models from Cluster D experiment 8, just evaluated
differently. The metric is end-to-end behavioral success rate, not open-loop
match-acc.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_goal import GoalDirectedGridWorld
from mapformer.train_variant import VARIANT_MAP


def closed_loop_episode(model, env, T_explore, T_navigate, device, rng,
                        action_only=True):
    """Run one closed-loop goal-directed episode.

    Returns:
        success: True if agent reached goal during navigate phase
        steps_to_goal: number of navigate steps before reaching goal (or T_navigate)
    """
    size = env.size
    # Random start
    x = int(rng.randint(0, size))
    y = int(rng.randint(0, size))

    # Random goal landmark
    g = env.landmark_cells[int(rng.randint(0, len(env.landmark_cells)))]
    gx, gy, g_idx = g
    goal_token = env.first_landmark_unified + g_idx

    # Phase 1: explore (random walks). Build context.
    context = [goal_token]
    for _ in range(T_explore):
        a = int(rng.randint(0, env.N_ACTIONS))
        dx, dy = env.ACTION_DELTAS[a]
        x = (x + dx) % size; y = (y + dy) % size
        context.append(a + env.action_offset)
        context.append(int(env.obs_map[x, y].item()) + env.obs_offset)

    # Phase 2: navigate (model picks actions, env executes)
    reached = False
    steps = T_navigate
    with torch.no_grad():
        for t in range(T_navigate):
            inp = torch.tensor(context, device=device).unsqueeze(0)
            logits = model(inp)
            if action_only:
                # Only consider action tokens (0..N_ACTIONS-1) at this position
                action_logits = logits[0, -1, :env.N_ACTIONS]
                a = int(action_logits.argmax().item())
            else:
                a_token = int(logits[0, -1].argmax().item())
                a = a_token if a_token < env.N_ACTIONS else 0
            dx, dy = env.ACTION_DELTAS[a]
            x = (x + dx) % size; y = (y + dy) % size
            if (x, y) == (gx, gy):
                reached = True
                steps = t + 1
                break
            context.append(a + env.action_offset)
            context.append(int(env.obs_map[x, y].item()) + env.obs_offset)

    return reached, steps


def closed_loop_eval(model, env, T_explore, T_navigate, n_episodes,
                    device="cuda", seed=2000):
    rng = np.random.RandomState(seed)
    successes = 0
    total_steps = 0
    for _ in range(n_episodes):
        ok, steps = closed_loop_episode(model, env, T_explore, T_navigate,
                                         device, rng)
        successes += int(ok)
        total_steps += steps
    return successes / n_episodes, total_steps / n_episodes


def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_MAP[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"])
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="GOAL_CLOSEDLOOP_RESULTS.md")
    ap.add_argument("--n-episodes", type=int, default=200)
    args = ap.parse_args()

    # 4 variants from Cluster D exp 8 + 1 absent baseline (RoPE)
    VARIANTS = ["Vanilla", "Level15", "Level15EM", "Level15NoDrop"]

    # Eval env (held-out obs_map and landmark positions)
    env_test = GoalDirectedGridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                      n_landmarks=200, seed=1000)

    # Three difficulty settings
    SETTINGS = [
        ("T_exp=32, T_nav=32", 32, 32),
        ("T_exp=64, T_nav=64 (train)", 64, 64),
        ("T_exp=128, T_nav=64 (long-explore OOD)", 128, 64),
    ]

    results = defaultdict(dict)  # results[variant][setting_label] = (success_rate, avg_steps)
    for variant in VARIANTS:
        ckpt = Path(f"mapformer/runs/{variant}_goal_lm200/seed0/{variant}_goal.pt")
        if not ckpt.exists():
            print(f"  skip {variant}: no ckpt at {ckpt}")
            continue
        try:
            m = build(variant, ckpt)
        except Exception as e:
            print(f"  skip {variant}: build failed: {e}")
            continue
        for label, T_exp, T_nav in SETTINGS:
            sr, avg_steps = closed_loop_eval(
                m, env_test, T_explore=T_exp, T_navigate=T_nav,
                n_episodes=args.n_episodes, device="cuda", seed=2000,
            )
            results[variant][label] = (sr, avg_steps)
            print(f"  {variant:20s} {label:42s} success={sr:.3f} avg_steps={avg_steps:.1f}")
        del m; torch.cuda.empty_cache()

    # Markdown
    lines = []
    lines.append("# Closed-loop goal-directed navigation — does the cognitive map enable behavior?\n")
    lines.append("Same models as Cluster D experiment 8 (goal-directed BC on lm200). NO new")
    lines.append("training. Evaluated in CLOSED LOOP: model picks action → env executes →")
    lines.append("agent position updates → next obs feeds back to model. Success = agent")
    lines.append("reaches goal cell during navigate phase.\n")
    lines.append("Eval env: held-out obs_map (env seed=1000), random start, random goal")
    lines.append(f"landmark per episode. {args.n_episodes} episodes per (variant, setting).\n")

    lines.append("## Closed-loop success rate")
    lines.append("")
    header = "| Variant | " + " | ".join(s[0] for s in SETTINGS) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(SETTINGS) + 1)) + "|")
    for v in VARIANTS:
        if v not in results:
            lines.append(f"| {v} | — | — | — |")
            continue
        row = f"| **{v}** | "
        cells = []
        for label, _, _ in SETTINGS:
            if label not in results[v]: cells.append("—"); continue
            sr, steps = results[v][label]
            cells.append(f"{sr:.3f} ({steps:.1f}st)")
        row += " | ".join(cells) + " |"
        lines.append(row)
    lines.append("")

    lines.append("## Comparison with open-loop match-acc (from Cluster D, exp 8)\n")
    lines.append("| Variant | T_exp=32 (match / closed) | T_exp=64 (match / closed) | T_exp=128 (match / closed) |")
    lines.append("|---|---|---|---|")
    OPEN_LOOP = {
        "Vanilla":       [0.628, 0.950, 0.766],
        "Level15":       [0.939, 0.947, 0.950],
        "Level15EM":     [0.936, 0.949, 0.948],
        "Level15NoDrop": [0.939, 0.946, 0.949],
    }
    for v in VARIANTS:
        if v not in results or v not in OPEN_LOOP:
            lines.append(f"| {v} | — | — | — |")
            continue
        oms = OPEN_LOOP[v]
        cells = []
        for i, (label, _, _) in enumerate(SETTINGS):
            sr = results[v][label][0] if label in results[v] else None
            sr_str = f"{sr:.3f}" if sr is not None else "—"
            cells.append(f"{oms[i]:.3f} / {sr_str}")
        lines.append(f"| **{v}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- **Match-acc** is open-loop: feed the expert trajectory, ask what action the model would pick. Doesn't compound errors.")
    lines.append("- **Closed-loop success** is end-to-end: model picks actions, env executes, agent must actually reach goal. Compounds errors.")
    lines.append("- If closed-loop ≈ match-acc: cognitive map supports re-planning on the fly. Model can recover from its own mistakes mid-trajectory.")
    lines.append("- If closed-loop ≪ match-acc: BC distribution-shift; per-step errors compound and agent gets lost.\n")
    lines.append("*Auto-generated by eval_goal_closedloop.py*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-30:]))


if __name__ == "__main__":
    main()
