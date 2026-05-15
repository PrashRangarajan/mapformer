"""Position-decoding probe: can the explicit spatial state recover agent (x, y)?

Cheaper sanity-check for the goal-distance probe. If the spatial state
(``theta_hat`` for Level15, ``theta_path`` for Vanilla, ``g`` for TEM)
doesn't even encode the agent's current cell linearly, the goal-distance
probe couldn't possibly work.

Target: (x, y) on a torus. Encoded as (cos(2π x/N), sin(2π x/N), cos(2π y/N),
sin(2π y/N)) so the target is bounded and wraps correctly. MAE measured in
the embedded space, plus a derived "average cell-distance error" estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.probe_goal_distance import build


def collect_state_and_xy(model, variant: str, env: GridWorld,
                         n_trajectories: int, T: int, device: str):
    """Returns (H, dx, dy) where dx/dy are DISPLACEMENT from the trajectory's
    starting cell, taken mod N. The model's theta encodes displacement from
    its (unknown) start, so absolute (x, y) is unrecoverable — but
    displacement should be."""
    H, X, Y = [], [], []
    for _ in range(n_trajectories):
        tokens, _, _ = env.generate_trajectory(T)
        positions = list(env.visited_locations)
        x0, y0 = positions[0]                 # start cell (used as origin)
        tt = tokens.unsqueeze(0).to(device)

        if variant == "TEMFaithful":
            W_a_all = model._orthogonal_W()
            g = model.g_init.unsqueeze(0).clone()
            g_at_obs = []
            for t in range(tt.shape[1]):
                tok_id = int(tt[0, t].item())
                if tok_id < model.n_actions:
                    Wb = W_a_all[tok_id]
                    g = (Wb @ g.unsqueeze(-1)).squeeze(-1)
                else:
                    g_at_obs.append(g[0].cpu().numpy())
            state_seq = np.stack(g_at_obs) if g_at_obs else None
        else:
            with torch.no_grad():
                _ = model(tt)
                if hasattr(model, "last_theta_hat"):
                    theta = model.last_theta_hat
                else:
                    x = model.token_emb(tt)
                    delta = model.action_to_lie(x)
                    cum = torch.cumsum(delta, dim=1)
                    theta = cum * model.path_integrator.omega.unsqueeze(0).unsqueeze(0)
            theta_obs = theta[0, 1::2, :, :]
            cs = torch.cat([torch.cos(theta_obs), torch.sin(theta_obs)], dim=-1)
            state_seq = cs.reshape(theta_obs.shape[0], -1).cpu().numpy()

        N = env.size
        for t, (x, y) in enumerate(positions):
            if t >= state_seq.shape[0]: break
            H.append(state_seq[t])
            X.append(int((x - x0) % N))
            Y.append(int((y - y0) % N))
    return np.stack(H), np.array(X), np.array(Y)


def encode_xy(x: np.ndarray, y: np.ndarray, N: int):
    th_x = 2 * np.pi * x / N
    th_y = 2 * np.pi * y / N
    return np.stack([np.cos(th_x), np.sin(th_x),
                      np.cos(th_y), np.sin(th_y)], axis=-1)


def decode_xy(emb: np.ndarray, N: int):
    th_x = np.arctan2(emb[..., 1], emb[..., 0])
    th_y = np.arctan2(emb[..., 3], emb[..., 2])
    x = (th_x * N / (2 * np.pi)) % N
    y = (th_y * N / (2 * np.pi)) % N
    return x, y


def torus_distance(x1, y1, x2, y2, N):
    dx = np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))
    dy = np.minimum(np.abs(y1 - y2), N - np.abs(y1 - y2))
    return dx + dy


class XYHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
    def forward(self, h):
        return self.net(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--variant", required=True, type=str)
    ap.add_argument("--n-trajectories", type=int, default=200)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--n-landmarks", type=int, default=200)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    print(f"Loading {args.variant} from {args.checkpoint}")
    model, cfg = build(args.variant, Path(args.checkpoint), device=args.device)
    N = cfg.get("grid_size", 64)
    env = GridWorld(size=N, n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5),
                    n_landmarks=args.n_landmarks, seed=args.env_seed)

    n_tr = int(args.n_trajectories * 0.75)
    n_ev = args.n_trajectories - n_tr
    print(f"Collecting train ({n_tr}) + eval ({n_ev}) trajectories...")
    Htr, Xtr, Ytr = collect_state_and_xy(model, args.variant, env, n_tr, args.T, args.device)
    Hev, Xev, Yev = collect_state_and_xy(model, args.variant, env, n_ev, args.T, args.device)
    Etr = encode_xy(Xtr, Ytr, N)
    Eev = encode_xy(Xev, Yev, N)
    print(f"  N train: {Htr.shape[0]}, N eval: {Hev.shape[0]}, state_dim: {Htr.shape[-1]}")

    head = XYHead(d_in=Htr.shape[-1], hidden=args.head_hidden).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    Htr_t = torch.from_numpy(Htr).float().to(args.device)
    Etr_t = torch.from_numpy(Etr).float().to(args.device)
    for ep in range(args.epochs):
        perm = torch.randperm(Htr_t.shape[0], device=args.device)
        losses = []
        for i in range(0, Htr_t.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = head(Htr_t[idx])
            loss = F.mse_loss(pred, Etr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{args.epochs} train MSE {np.mean(losses):.5f}")

    head.eval()
    with torch.no_grad():
        Hev_t = torch.from_numpy(Hev).float().to(args.device)
        pred = head(Hev_t).cpu().numpy()
    x_hat, y_hat = decode_xy(pred, N)
    d_err = torus_distance(x_hat, y_hat, Xev, Yev, N)
    mae_cells = float(np.mean(d_err))
    median_cells = float(np.median(d_err))
    chance = float(np.mean(torus_distance(
        np.random.randint(0, N, size=len(Xev)), np.random.randint(0, N, size=len(Xev)),
        Xev, Yev, N)))

    out = {
        "variant": args.variant, "checkpoint": args.checkpoint,
        "state_dim": int(Htr.shape[-1]), "N": int(N),
        "n_eval": int(len(Xev)),
        "mae_cells": mae_cells,
        "median_cells": median_cells,
        "chance_baseline_mae_cells": chance,
    }
    print(f"\nResults:")
    print(f"  Decoded MAE: {mae_cells:.2f} cells (median {median_cells:.2f})")
    print(f"  Random-position baseline: {chance:.2f} cells")
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
