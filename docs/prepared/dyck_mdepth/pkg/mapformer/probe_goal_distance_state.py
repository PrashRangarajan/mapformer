"""Goal-distance probe over EXPLICIT spatial state (not the readout hidden state).

The first goal-distance probe (probe_goal_distance.py) read from the input to
``out_proj``, which for MapFormer family is the post-LayerNorm last-layer
hidden state — a mix of position and content. All variants came in at or
below the constant-predictor baseline; only TEMFaithful had a weak signal
(Spearman 0.27).

This probe reads the model's EXPLICIT spatial state instead:

  - Level15-family: ``model.last_theta_hat`` (B, L, H, NB) — InEKF-corrected
    angles per head/block. Flattened to (H*NB,) per timestep.
  - Vanilla: ``theta_path`` computed on the fly (the uncorrected cumulative-
    sum angle). Same shape.
  - TEMFaithful: ``g`` (B, d_g) — the structural code, rolled out manually
    step by step (same logic as in make_place_cell_figure.py).

The head is the same 2-layer MLP from probe_goal_distance.py, with input
dim = (spatial_state_dim + goal_embedding_dim). If theta_hat / theta_path /
g support goal-relative-distance prediction much better than the mixed
hidden state, we know the cognitive map IS there, just not in the readout
representation — which is the active-inference / world-model-planning
relevant claim.

For theta values, we feed (cos(theta), sin(theta)) instead of raw theta,
since angles wrap modulo 2π and raw theta is unbounded under path integration
at long T. This is a faithful representation; the head can't sneak around it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from mapformer.environment import GridWorld
from mapformer.environment_goal import bfs_torus
from mapformer.train_variant import VARIANT_MAP
from mapformer.probe_goal_distance import DistanceHead, train_head, eval_head, build


def state_dim_for(variant: str, model) -> int:
    """Dim of the explicit spatial state per obs position."""
    if variant in ("Level15", "Level15NoDrop", "Level15GSF_NoDrop",
                   "Level15GSF_NoDrop_K16", "Vanilla", "VanillaNoDrop"):
        # (H*NB) angles → 2 * H*NB after cos/sin
        return 2 * model.n_heads * model.n_blocks
    if variant == "TEMFaithful":
        return model.d_g
    raise ValueError(f"unsupported variant: {variant}")


@torch.no_grad()
def collect_state_data(
    model, variant: str, env: GridWorld,
    n_trajectories: int, T: int, goal_pool: list,
    rng: np.random.RandomState, device: str,
):
    """Same return as collect_env_data, but H now stores explicit spatial
    state instead of hidden-layer output."""
    H_all, G_all, D_all = [], [], []
    for _ in range(n_trajectories):
        gx, gy, g_idx = goal_pool[int(rng.randint(0, len(goal_pool)))]
        goal_token = env.first_landmark_unified + g_idx

        tokens, _, _ = env.generate_trajectory(T)
        positions = list(env.visited_locations)
        tt = tokens.unsqueeze(0).to(device)

        if variant == "TEMFaithful":
            # Manual rollout of TEM's g state at obs positions
            W_a_all = model._orthogonal_W()
            g = model.g_init.unsqueeze(0).clone()
            g_at_obs = []
            t_obs = 0
            for t in range(tt.shape[1]):
                tok_id = int(tt[0, t].item())
                if tok_id < model.n_actions:
                    Wb = W_a_all[tok_id]
                    g = (Wb @ g.unsqueeze(-1)).squeeze(-1)
                else:
                    g_at_obs.append(g[0].cpu().numpy())
                    t_obs += 1
            state_seq = np.stack(g_at_obs) if g_at_obs else None  # (n_obs, d_g)
        else:
            # MapFormer family: forward once, then grab theta state
            _ = model(tt)
            if hasattr(model, "last_theta_hat"):
                theta = model.last_theta_hat                       # (1, L, H, NB)
            else:
                # Vanilla: recompute theta_path on the fly
                x = model.token_emb(tt)
                delta = model.action_to_lie(x)
                cum = torch.cumsum(delta, dim=1)
                theta = cum * model.path_integrator.omega.unsqueeze(0).unsqueeze(0)
            B, L, H, NB = theta.shape
            theta_obs = theta[0, 1::2, :, :]                       # (n_obs, H, NB)
            cs = torch.cat([torch.cos(theta_obs), torch.sin(theta_obs)], dim=-1)
            state_seq = cs.reshape(theta_obs.shape[0], -1).cpu().numpy()

        for t, (x, y) in enumerate(positions):
            if t >= state_seq.shape[0]: break
            d = len(bfs_torus((int(x), int(y)), (int(gx), int(gy)), env.size))
            if d == 0 and (x, y) != (gx, gy): continue
            H_all.append(state_seq[t])
            G_all.append(int(goal_token))
            D_all.append(int(d))
    return np.stack(H_all), np.array(G_all), np.array(D_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--variant", required=True, type=str)
    ap.add_argument("--n-trajectories", type=int, default=200,
                    help="Total trajectories on the single env (split into train/eval).")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--n-landmarks", type=int, default=200)
    ap.add_argument("--train-goal-frac", type=float, default=0.75)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.rng_seed)
    print(f"Loading {args.variant} from {args.checkpoint}")
    model, cfg = build(args.variant, Path(args.checkpoint), device=args.device)
    state_d = state_dim_for(args.variant, model)
    model_emb = model.token_emb if hasattr(model, "token_emb") else model.content_emb
    d_goal = model_emb.embedding_dim
    print(f"  spatial state dim: {state_d}, goal emb dim: {d_goal}")

    env = GridWorld(size=cfg.get("grid_size", 64),
                    n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5),
                    n_landmarks=args.n_landmarks,
                    seed=args.env_seed)
    n_train_g = int(args.n_landmarks * args.train_goal_frac)
    rng_g = np.random.RandomState(args.rng_seed)
    idxs = np.arange(len(env.landmark_cells)); rng_g.shuffle(idxs)
    train_g_set = set(idxs[:n_train_g].tolist())
    train_goals = [env.landmark_cells[i] for i in range(len(env.landmark_cells))
                   if i in train_g_set]
    heldout_goals = [env.landmark_cells[i] for i in range(len(env.landmark_cells))
                     if i not in train_g_set]

    n_tr = int(args.n_trajectories * 0.75)
    n_ev = args.n_trajectories - n_tr
    print(f"Collecting {n_tr} probe-train trajectories...")
    Htr, Gtr, Dtr = collect_state_data(
        model, args.variant, env, n_tr, args.T, train_goals,
        rng, args.device,
    )
    print(f"  N train points: {Htr.shape[0]}")

    print(f"Collecting probe eval data (train env + train goals)...")
    Htt, Gtt, Dtt = collect_state_data(
        model, args.variant, env, n_ev // 2, args.T, train_goals,
        rng, args.device,
    )
    print(f"Collecting probe eval data (train env + HELDOUT goals)...")
    Hth, Gth, Dth = collect_state_data(
        model, args.variant, env, n_ev, args.T, heldout_goals,
        rng, args.device,
    )

    head = DistanceHead(d_hidden=state_d, d_goal=d_goal,
                        hidden=args.head_hidden).to(args.device)
    train_head(head, model_emb, Htr, Gtr, Dtr,
               n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
               weight_decay=args.weight_decay, device=args.device)

    print("\nEval results:")
    results = {}
    for name, H, G, D in [
        ("train_goals", Htt, Gtt, Dtt),
        ("heldout_goals", Hth, Gth, Dth),
    ]:
        r = eval_head(head, model_emb, H, G, D,
                      batch_size=args.batch_size, device=args.device)
        results[name] = r
        print(f"  {name}: MAE={r['mae']:.3f} (vs const {r['mae_const_baseline']:.3f}), "
              f"spearman={r['spearman']:.3f}, n={r['n']}")

    out = {
        "variant": args.variant, "checkpoint": args.checkpoint,
        "state_dim": state_d, "goal_emb_dim": d_goal,
        "results": results,
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
