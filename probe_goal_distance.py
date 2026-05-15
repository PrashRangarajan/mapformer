"""Probe: does the frozen cognitive map encode goal-relative distance?

The Tolman / vector-navigation operational definition of a cognitive map is
that the internal state at any point in time, combined with the identity of
a goal, supports an estimate of how to reach that goal. The cleanest
representational test is: given a FROZEN hidden state and a separately
provided goal token, can a small MLP head predict the BFS distance to the
goal cell on a HELD-OUT environment?

Key design points (and why):

- Backbones are trained WITHOUT goal conditioning. Random-walk trajectories
  only. This isolates "the hidden state contains where I am" from "the
  model is given the goal." Tolman's latent-learning claim is exactly
  that: the cognitive map is built without task pressure.

- The head sees (hidden_state, model.token_emb(goal_token)). The goal
  embedding is the model's OWN representation of that landmark — we don't
  give the head a side-channel of "what does this landmark mean." The
  head can only combine pre-existing representations.

- Train probe on TRAIN envs (the multi-env training set the backbone saw).
  Eval probe on HELD-OUT envs (the same held-out pool the backbone was
  evaluated on at prediction). Two cuts:
    1. train env + train goals (sanity)
    2. held-out env + held-out goals (the real test of representational
       cognitive map)

- Metrics: MAE on BFS distance, Spearman rank correlation, and a coarse
  "is argmin-direction-action correct" check that uses ground-truth
  one-step-ahead positions (no model rollout — keeps the test purely
  representational).

If Level1.5 / TEM beat Vanilla / RoPE on the held-out cut, the cognitive
map exists as a *representation* even before any planning machinery is
attached. That's the headline workshop claim, and it sidesteps the
closed-loop policy distribution-shift problem entirely.
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


# -------- hidden-state capture (same trick as probe_hex_emergence.py) --------
class _HiddenCapture:
    """Captures input to out_proj (= LayerNormed last-layer hidden state for
    MapFormer family; LN(x_hat) for TEMFaithful — we still get a per-position
    embedding either way)."""
    def __init__(self): self.hiddens = []
    def reset(self): self.hiddens = []
    def __call__(self, mod, inp, out):
        h = inp[0].detach()
        if h.dim() == 4: h = h[:, 0]                 # GSF: take mode 0
        self.hiddens.append(h)


def build(variant: str, ckpt_path: Path, device: str = "cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c["config"]; cls = VARIANT_MAP[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg.get("grid_size", 64))
    if "n_modes" in cfg: kw["n_modes"] = cfg["n_modes"]
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg


# -------- data collection --------------------------------------------------
@torch.no_grad()
def collect_env_data(
    model, env: GridWorld, n_trajectories: int, T: int,
    goal_pool: list,                # list of (gx, gy, g_idx)
    rng: np.random.RandomState,
    device: str,
    d_model: int,
):
    """Generate random-walk trajectories on `env`, for each trajectory pick
    one goal from `goal_pool`, record (hidden_state, goal_token_id, distance)
    at every obs position.

    Returns: H (N, d_model), G (N,), D (N,) arrays.
    """
    cap = _HiddenCapture()
    handle = model.out_proj.register_forward_hook(cap)

    H_all, G_all, D_all = [], [], []
    for _ in range(n_trajectories):
        gx, gy, g_idx = goal_pool[int(rng.randint(0, len(goal_pool)))]
        goal_token = env.first_landmark_unified + g_idx

        cap.reset()
        tokens, _, _ = env.generate_trajectory(T)
        positions = list(env.visited_locations)
        tt = tokens.unsqueeze(0).to(device)
        _ = model(tt)

        # Unify the two capture modes
        if len(cap.hiddens) == 1 and cap.hiddens[0].dim() == 3:
            h_full = cap.hiddens[0][0]                # (L, d)
            seq_len = h_full.shape[0]
        else:
            # TEMFaithful: many per-step (1, d) captures
            h_full = torch.stack([h.squeeze(0) if h.dim() == 2 else h
                                  for h in cap.hiddens], dim=0)
            seq_len = h_full.shape[0]

        for t, (x, y) in enumerate(positions):
            obs_idx = 2 * t + 1
            if obs_idx >= seq_len: break
            d = len(bfs_torus((int(x), int(y)), (int(gx), int(gy)), env.size))
            if d == 0 and (x, y) != (gx, gy): continue   # unreachable
            H_all.append(h_full[obs_idx].cpu().numpy())
            G_all.append(int(goal_token))
            D_all.append(int(d))
    handle.remove()
    return np.stack(H_all), np.array(G_all), np.array(D_all)


# -------- probe head -------------------------------------------------------
class DistanceHead(nn.Module):
    """(hidden, goal_emb) -> predicted BFS distance.

    Two-layer MLP, post-concat. Goal embedding is the FROZEN model.token_emb
    of the goal token id — the head can't learn what a landmark "means," only
    how to combine its existing embedding with the hidden state.
    """
    def __init__(self, d_hidden: int, d_goal: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden + d_goal, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, h, goal_emb):
        x = torch.cat([h, goal_emb], dim=-1)
        return self.net(x).squeeze(-1)


def train_head(
    head: DistanceHead, model_token_emb: nn.Embedding,
    H: np.ndarray, G: np.ndarray, D: np.ndarray,
    n_epochs: int, batch_size: int, lr: float, weight_decay: float,
    device: str,
):
    H_t = torch.from_numpy(H).float().to(device)
    G_t = torch.from_numpy(G).long().to(device)
    D_t = torch.from_numpy(D).float().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    N = H_t.shape[0]
    for ep in range(n_epochs):
        perm = torch.randperm(N, device=device)
        losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            h = H_t[idx]; g = G_t[idx]; d = D_t[idx]
            with torch.no_grad():
                g_emb = model_token_emb(g)
            pred = head(h, g_emb)
            loss = F.mse_loss(pred, d)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{n_epochs} train MSE {np.mean(losses):.3f}")


@torch.no_grad()
def eval_head(
    head: DistanceHead, model_token_emb: nn.Embedding,
    H: np.ndarray, G: np.ndarray, D: np.ndarray,
    batch_size: int, device: str,
):
    H_t = torch.from_numpy(H).float().to(device)
    G_t = torch.from_numpy(G).long().to(device)
    D_t = torch.from_numpy(D).float().to(device)
    preds = []
    for i in range(0, H_t.shape[0], batch_size):
        h = H_t[i:i + batch_size]; g = G_t[i:i + batch_size]
        g_emb = model_token_emb(g)
        preds.append(head(h, g_emb).cpu().numpy())
    preds = np.concatenate(preds)
    mae = float(np.mean(np.abs(preds - D)))
    rho, _ = spearmanr(preds, D)
    # Random-baseline MAE: predict the mean distance everywhere
    mae_const = float(np.mean(np.abs(D - D.mean())))
    return {"mae": mae, "spearman": float(rho), "mae_const_baseline": mae_const,
            "n": int(H_t.shape[0])}


# -------- main pipeline ----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--variant", required=True, type=str)
    ap.add_argument("--n-train-envs", type=int, default=20,
                    help="Number of train envs (same family as the backbone "
                         "was trained on). Probe data collected from these.")
    ap.add_argument("--n-test-envs", type=int, default=20,
                    help="Held-out envs (different obs_map). Probe evaluated here.")
    ap.add_argument("--n-trajectories-per-env", type=int, default=50)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--n-landmarks", type=int, default=200)
    ap.add_argument("--p-empty", type=float, default=0.5)
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--train-env-seed-base", type=int, default=0,
                    help="env seed for the first train env (others incremented).")
    ap.add_argument("--test-env-seed-base", type=int, default=10000)
    ap.add_argument("--train-goal-frac", type=float, default=0.75,
                    help="Fraction of landmarks used as train goals; the rest are held-out.")
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
    d_model = cfg["d_model"]
    print(f"  d_model={d_model}, vocab={cfg['vocab_size']}")

    # ---- Build env pools ----
    def make_envs(n_envs, base_seed):
        return [
            GridWorld(size=args.size, n_obs_types=args.n_obs_types,
                      p_empty=args.p_empty, n_landmarks=args.n_landmarks,
                      seed=base_seed + i)
            for i in range(n_envs)
        ]
    train_envs = make_envs(args.n_train_envs, args.train_env_seed_base)
    test_envs = make_envs(args.n_test_envs, args.test_env_seed_base) if args.n_test_envs > 0 else []
    print(f"  built {len(train_envs)} train envs + {len(test_envs)} test envs")

    # ---- Split landmarks per env into train_goals / test_goals ----
    n_train_g = int(args.n_landmarks * args.train_goal_frac)

    def split_goals(env, rng_local):
        idxs = np.arange(len(env.landmark_cells))
        rng_local.shuffle(idxs)
        train_idxs = set(idxs[:n_train_g].tolist())
        test_idxs = set(idxs[n_train_g:].tolist())
        train_g = [env.landmark_cells[i] for i in range(len(env.landmark_cells))
                   if i in train_idxs]
        test_g = [env.landmark_cells[i] for i in range(len(env.landmark_cells))
                  if i in test_idxs]
        return train_g, test_g

    # ---- Collect data ----
    print("Collecting probe TRAINING data (train envs, train goals)...")
    Htr, Gtr, Dtr = [], [], []
    for env in train_envs:
        tg, _ = split_goals(env, np.random.RandomState(args.rng_seed))
        H_, G_, D_ = collect_env_data(
            model, env, args.n_trajectories_per_env, args.T,
            tg, rng, args.device, d_model,
        )
        Htr.append(H_); Gtr.append(G_); Dtr.append(D_)
    Htr = np.concatenate(Htr); Gtr = np.concatenate(Gtr); Dtr = np.concatenate(Dtr)
    print(f"  N train points: {Htr.shape[0]}, dist range [{Dtr.min()}, {Dtr.max()}]")

    print("Collecting probe EVAL data: train env + train goals (sanity)...")
    Hs_tt, Gs_tt, Ds_tt = [], [], []
    for env in train_envs[:5]:
        tg, _ = split_goals(env, np.random.RandomState(args.rng_seed))
        H_, G_, D_ = collect_env_data(
            model, env, max(args.n_trajectories_per_env // 5, 5), args.T,
            tg, rng, args.device, d_model,
        )
        Hs_tt.append(H_); Gs_tt.append(G_); Ds_tt.append(D_)
    Hs_tt = np.concatenate(Hs_tt); Gs_tt = np.concatenate(Gs_tt); Ds_tt = np.concatenate(Ds_tt)

    print("Collecting probe EVAL data: train env + HELD-OUT goals...")
    Hs_th, Gs_th, Ds_th = [], [], []
    for env in train_envs[:5]:
        _, hg = split_goals(env, np.random.RandomState(args.rng_seed))
        if not hg: continue
        H_, G_, D_ = collect_env_data(
            model, env, max(args.n_trajectories_per_env // 5, 5), args.T,
            hg, rng, args.device, d_model,
        )
        Hs_th.append(H_); Gs_th.append(G_); Ds_th.append(D_)
    Hs_th = np.concatenate(Hs_th); Gs_th = np.concatenate(Gs_th); Ds_th = np.concatenate(Ds_th)

    if test_envs:
        print("Collecting probe EVAL data: HELD-OUT env + HELD-OUT goals (the real test)...")
        Hs_hh, Gs_hh, Ds_hh = [], [], []
        for env in test_envs:
            _, hg = split_goals(env, np.random.RandomState(args.rng_seed))
            if not hg: continue
            H_, G_, D_ = collect_env_data(
                model, env, max(args.n_trajectories_per_env // 2, 10), args.T,
                hg, rng, args.device, d_model,
            )
            Hs_hh.append(H_); Gs_hh.append(G_); Ds_hh.append(D_)
        Hs_hh = np.concatenate(Hs_hh); Gs_hh = np.concatenate(Gs_hh); Ds_hh = np.concatenate(Ds_hh)
    else:
        Hs_hh = Gs_hh = Ds_hh = None

    # ---- Train head ----
    print("Training distance head...")
    head = DistanceHead(d_hidden=Htr.shape[-1], d_goal=d_model,
                        hidden=args.head_hidden).to(args.device)
    model_emb = model.token_emb if hasattr(model, "token_emb") else model.content_emb
    train_head(
        head, model_emb, Htr, Gtr, Dtr,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, device=args.device,
    )

    # ---- Eval ----
    print("\nEval results:")
    results = {}
    eval_cuts = [
        ("train_env_train_goals", Hs_tt, Gs_tt, Ds_tt),
        ("train_env_heldout_goals", Hs_th, Gs_th, Ds_th),
    ]
    if Hs_hh is not None:
        eval_cuts.append(("heldout_env_heldout_goals", Hs_hh, Gs_hh, Ds_hh))
    for name, H, G, D in eval_cuts:
        r = eval_head(head, model_emb, H, G, D,
                       batch_size=args.batch_size, device=args.device)
        results[name] = r
        print(f"  {name}: MAE={r['mae']:.3f} (vs const-baseline {r['mae_const_baseline']:.3f}),"
              f" spearman={r['spearman']:.3f}, n={r['n']}")

    out = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "config": {k: getattr(args, k) for k in [
            "n_train_envs", "n_test_envs", "n_trajectories_per_env", "T",
            "n_landmarks", "train_goal_frac", "epochs", "lr",
        ]},
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
