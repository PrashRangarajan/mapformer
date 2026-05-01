"""Evaluate trained continuous-nav models on multiple sequence lengths and
noise levels. Reports MSE on DoG predictions and decoded-position error.

Decoded position from a predicted DoG vector p̂:
    The model's prediction is supposed to be `DoG(true_position)`. Decode by
    finding the place-cell centre c_j with maximum predicted firing — the
    population-vector decoder. Distance(decoded_centre, true_position) on
    the torus is the spatial error.

Usage:
    python3 -m mapformer.eval_continuous \
        --checkpoints runs/cnav/Vanilla/seed0/Vanilla.pt \
                       runs/cnav/Level15/seed0/Level15.pt \
        --T-list 128 256 512 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .continuous_nav import ContinuousNav2D
from .model_continuous import (
    MapFormerWM_Continuous, MapFormerWM_Continuous_Level15,
)

VARIANT_CLS = {
    "Vanilla": MapFormerWM_Continuous,
    "Level15": MapFormerWM_Continuous_Level15,
}


def _build(ckpt_path: str, device: str = "cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c.get("config", {})
    variant = c.get("variant", "Level15")
    cls = VARIANT_CLS[variant]
    extra = {}
    if variant == "Level15":
        extra["log_R_init_bias"] = 0.0
    m = cls(
        action_dim=2,
        obs_dim=cfg.get("n_place_cells", 256),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 2),
        n_layers=cfg.get("n_layers", 1),
        grid_size=int(cfg.get("size", 64)),
        n_grid_units=cfg.get("n_grid_units", 0),
        **extra,
    )
    m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg, variant


def torus_dist(p1: np.ndarray, p2: np.ndarray, size: float) -> np.ndarray:
    """L2 distance on a torus."""
    d = p1 - p2
    d = (d + size / 2) % size - size / 2
    return np.sqrt((d * d).sum(-1))


def eval_one(model, env, T: int, n_traj: int, eval_v_noise: float,
             eval_omega_noise: float, device: str):
    """Returns (mse, mean_position_error, percentile_50, percentile_90)."""
    # Override env noise for this eval (fresh env, not the training env)
    env.v_noise_std = eval_v_noise
    env.omega_noise_std = eval_omega_noise

    rng = np.random.RandomState(42)
    mses = []
    errs = []

    with torch.no_grad():
        for _ in range(n_traj):
            actions, _, positions, _, obs = env.generate_trajectory(T, rng=rng)
            A = torch.from_numpy(actions[None]).to(device)
            O = torch.from_numpy(obs    [None]).to(device)
            preds = model(A, O)                  # (1, 2T, n_pc)
            preds_at_actions = preds[0, 0::2].cpu().numpy()  # (T, n_pc)

            mses.append(((preds_at_actions - obs) ** 2).mean())

            # Population-vector decoder: argmax place cell
            decoded_idx = preds_at_actions.argmax(axis=1)
            decoded_pos = env.place_centers[decoded_idx]
            err = torus_dist(decoded_pos, positions, env.size)
            errs.append(err)

    return (
        float(np.mean(mses)),
        float(np.concatenate(errs).mean()),
        float(np.percentile(np.concatenate(errs), 50)),
        float(np.percentile(np.concatenate(errs), 90)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--T-list", type=int, nargs="+",
                        default=[128, 256, 512, 1024])
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2],
                        help="(v_noise, omega_noise) pairs sweep — use same "
                             "value for both. 0.0 = no eval-time noise.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("# Continuous-nav evaluation\n")
    print("Position decoder: argmax over place cells (population vector).")
    print("Spatial error reported in cells (torus distance from decoded centre to true position).\n")

    rows = []
    for ckpt_path in args.checkpoints:
        print(f"Loading {ckpt_path}...")
        model, cfg, variant = _build(ckpt_path, device=args.device)
        env = ContinuousNav2D(
            size=cfg.get("size", 64.0),
            n_place_cells=cfg.get("n_place_cells", 256),
            sigma_E=cfg.get("sigma_E", 1.5),
            sigma_I=cfg.get("sigma_I", 3.0),
            v_mean=cfg.get("v_mean", 0.7),
            v_std=cfg.get("v_std", 0.3),
            omega_std=cfg.get("omega_std", 0.5),
            seed=999,
        )
        for T in args.T_list:
            for noise in args.noise_levels:
                mse, mean_err, p50, p90 = eval_one(
                    model, env, T, args.n_traj,
                    eval_v_noise=noise, eval_omega_noise=noise,
                    device=args.device,
                )
                rows.append({
                    "ckpt": Path(ckpt_path).stem, "variant": variant,
                    "T": T, "noise": noise,
                    "mse": mse, "mean_err": mean_err, "p50": p50, "p90": p90,
                })

    # Print as table
    print("\n## Results table\n")
    print("| Ckpt | T | eval-noise | MSE | mean err (cells) | p50 err | p90 err |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['ckpt']} | {r['T']} | {r['noise']:.2f} | "
              f"{r['mse']:.5f} | {r['mean_err']:.2f} | "
              f"{r['p50']:.2f} | {r['p90']:.2f} |")


if __name__ == "__main__":
    main()
