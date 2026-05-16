"""Active-inference one-step probe: can the cognitive map drive goal-directed action?

For each candidate action, ask the trained forward model "what observation
do you expect to see next?" Score actions by predicted log-probability of
the goal token; pick argmax. This is one-step active inference under a
delta-on-goal target distribution (= Friston's expected free energy
reduced to information gain about the goal observation).

Critically:
  - The backbones being probed were trained on RANDOM WALKS only. No goal
    conditioning. No policy training. We're using the model purely as a
    learned forward model and asking whether goal-directed planning emerges
    from a simple decision rule at deployment time.
  - Eval is closed-loop: model picks action → env executes → next obs feeds
    back. So this is the behavioural test, sidestepping the BC distribution-
    shift problem that killed `eval_goal_closedloop.py` (no policy training
    → no compounding-error blow-up).

Pass criterion (rough): if Level1.5 / TEM achieve >> random-walk success
rate on held-out goals while RoPE/Vanilla don't, the cognitive map is
*usable* for planning even though it was never trained to plan.

Result lands in `ACTIVE_INFERENCE_RESULTS.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.environment_goal import bfs_torus
from mapformer.train_variant import VARIANT_MAP


def build(variant, ckpt_path, device="cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c["config"]; cls = VARIANT_MAP[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg.get("grid_size", 64))
    if "n_modes" in cfg: kw["n_modes"] = cfg["n_modes"]
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def _score_action_rollout(model, env, base_tokens, a_first, goal_token, horizon, rng, device):
    """Score `a_first` by max log p(goal_token) over a `horizon`-step random
    rollout of subsequent actions. horizon=1 collapses to immediate next-obs."""
    seq = list(base_tokens) + [int(a_first + env.action_offset)]
    score = float("-inf")
    for step in range(horizon):
        tt = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(tt)
        log_p = F.log_softmax(logits[0, -1, :], dim=-1)
        score = max(score, float(log_p[goal_token].item()))
        # roll forward: sample obs by argmax (mode of predicted obs), then add
        # a random action to continue rollout
        if step < horizon - 1:
            pred_obs = int(log_p.argmax().item())
            seq.append(pred_obs)
            seq.append(int(rng.randint(0, env.N_ACTIONS)) + env.action_offset)
    return score


@torch.no_grad()
def active_inference_episode(
    model, env: GridWorld, T_explore: int, T_navigate: int,
    rng: np.random.RandomState, device: str = "cuda",
    horizon: int = 1,
):
    """Run one goal-directed episode using active-inference action selection.

    The model never sees a goal token in its input — the goal is only used
    by the decision rule (score actions by predicted log p(goal_token|...)).
    """
    N = env.size
    # Sample goal
    gx, gy, g_idx = env.landmark_cells[int(rng.randint(0, len(env.landmark_cells)))]
    goal_token = env.first_landmark_unified + g_idx

    # Sample start
    x = int(rng.randint(0, N)); y = int(rng.randint(0, N))
    start_xy = (x, y)

    # Explore: random walk (builds the cognitive map)
    tokens: list[int] = []
    for _ in range(T_explore):
        a = int(rng.randint(0, env.N_ACTIONS))
        dx, dy = env.ACTION_DELTAS[a]
        x = (x + dx) % N; y = (y + dy) % N
        tokens.append(int(a + env.action_offset))
        tokens.append(int(env.obs_map[x, y].item()) + env.obs_offset)

    # Optimal BFS reference
    bfs_path = bfs_torus((x, y), (gx, gy), N)
    optimal_steps = len(bfs_path) if bfs_path else None

    # Active-inference navigate
    steps_to_goal = None
    for t in range(T_navigate):
        if (x, y) == (gx, gy):
            steps_to_goal = t
            break
        # For each candidate first action, score by max log p(goal_token)
        # over a horizon-step rollout (horizon=1 is one-step active inference;
        # horizon>1 is Dreamer-style lookahead).
        scores = torch.full((env.N_ACTIONS,), float("-inf"), device=device)
        for a in range(env.N_ACTIONS):
            scores[a] = _score_action_rollout(
                model, env, tokens, a, goal_token, horizon, rng, device,
            )
        a = int(scores.argmax().item())
        # Execute
        dx, dy = env.ACTION_DELTAS[a]
        x = (x + dx) % N; y = (y + dy) % N
        tokens.append(int(a + env.action_offset))
        tokens.append(int(env.obs_map[x, y].item()) + env.obs_offset)
    if (x, y) == (gx, gy) and steps_to_goal is None:
        steps_to_goal = T_navigate

    return {
        "success": steps_to_goal is not None,
        "steps_to_goal": steps_to_goal,
        "optimal_steps": optimal_steps,
        "ratio_to_optimal": (steps_to_goal / optimal_steps)
            if (steps_to_goal is not None and optimal_steps) else None,
        "start": list(start_xy), "goal": [int(gx), int(gy), int(g_idx)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--T-explore", type=int, default=128)
    ap.add_argument("--T-navigate", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=1,
                    help="Active-inference lookahead. 1 = one-step (structurally "
                         "limited to last-step adjacency); K>1 = Dreamer-style "
                         "rollout, K times more compute per decision.")
    ap.add_argument("--env-seed", type=int, default=0,
                    help="Env seed. 0 = same as training (lm200 single-env).")
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    print(f"Loading {args.variant} from {args.checkpoint}")
    model, cfg = build(args.variant, Path(args.checkpoint), device=args.device)
    env = GridWorld(size=cfg.get("grid_size", 64),
                    n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5),
                    n_landmarks=cfg.get("n_landmarks", 200),
                    seed=args.env_seed)
    print(f"  env: size={env.size}, n_landmarks={len(env.landmark_cells)}")
    rng = np.random.RandomState(args.rng_seed)
    results = []
    for ep in range(args.n_episodes):
        r = active_inference_episode(
            model, env, args.T_explore, args.T_navigate, rng,
            device=args.device, horizon=args.horizon,
        )
        results.append(r)
        if (ep + 1) % 20 == 0:
            sr = np.mean([x["success"] for x in results])
            print(f"  ep {ep+1}/{args.n_episodes}: success_rate so far {sr:.3f}")

    successes = [r["success"] for r in results]
    succ = [r for r in results if r["success"] and r["ratio_to_optimal"] is not None]
    out = {
        "variant": args.variant, "ckpt": args.checkpoint,
        "n_episodes": len(results),
        "horizon": args.horizon,
        "success_rate": float(np.mean(successes)),
        "mean_steps_to_goal": float(np.mean([r["steps_to_goal"] for r in succ])) if succ else None,
        "mean_ratio_to_optimal": float(np.mean([r["ratio_to_optimal"] for r in succ])) if succ else None,
        "T_explore": args.T_explore, "T_navigate": args.T_navigate,
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({**out, "raw": results}, indent=2))
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
