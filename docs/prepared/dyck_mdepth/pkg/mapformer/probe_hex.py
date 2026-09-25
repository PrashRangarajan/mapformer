"""Probe a trained model's grid layer for hex-grid emergence.

Workflow:
  1. Load Level15_DoG checkpoint.
  2. Run N trajectories of length T; record (position, grid_activations)
     at obs positions in `model.last_grid_activations`.
  3. For each unit j, build a rate map of shape (gs, gs) by averaging
     activations across visits to each cell.
  4. Compute the spatial autocorrelogram (SAC) per unit and a Sargolini-
     style grid score.
  5. Print summary: max grid score, fraction of units > 0.3, distribution.

Usage:
  python3 -m mapformer.probe_hex --checkpoint runs/Level15_DoG_clean/seed0/Level15_DoG.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .environment import GridWorld
from .model_level15_dog import MapFormerWM_Level15_DoG


def _build(ckpt_path: str, device: str = "cuda") -> MapFormerWM_Level15_DoG:
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c.get("config", {})
    m = MapFormerWM_Level15_DoG(
        vocab_size=cfg["vocab_size"],
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 2),
        n_layers=cfg.get("n_layers", 1),
        grid_size=cfg.get("grid_size", 64),
    )
    m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval()


def collect_rate_maps(model, env, T: int, n_traj: int, device: str = "cuda"):
    """Build (n_grid, gs, gs) rate-map tensor by averaging unit activations
    at each visited cell across many trajectories."""
    gs = env.size
    n_grid = model.n_grid_units
    sums = np.zeros((n_grid, gs, gs), dtype=np.float64)
    counts = np.zeros((gs, gs), dtype=np.float64)

    with torch.no_grad():
        for _ in range(n_traj):
            tokens, _, _ = env.generate_trajectory(T)
            locs = list(env.visited_locations)
            tt = tokens.unsqueeze(0).to(device)
            _ = model(tt[:, :-1])
            g = model.last_grid_activations[0].cpu().numpy()  # (L_in, n_grid)

            for t, (px, py) in enumerate(locs):
                idx = 2 * t + 1
                if idx >= g.shape[0]:
                    break
                sums[:, px, py] += g[idx]
                counts[px, py] += 1.0

    counts_safe = np.maximum(counts, 1.0)
    rate = sums / counts_safe[None]
    return rate, counts


def spatial_autocorrelogram(rate_map: np.ndarray) -> np.ndarray:
    """2D autocorrelogram of a rate map (no normalization needed for grid score
    which uses Pearson correlation over annular region)."""
    r = rate_map - rate_map.mean()
    F = np.fft.fft2(r)
    sac = np.fft.ifft2(F * np.conj(F)).real
    sac = np.fft.fftshift(sac)
    return sac


def grid_score(sac: np.ndarray) -> float:
    """Sargolini-style grid score from SAC.

    Construct an annular region around the center (excluding the central peak),
    compute Pearson correlation of that annulus with rotations at
    30/60/90/120/150 degrees, then:
        score = min(corr60, corr120) - max(corr30, corr90, corr150)
    """
    from scipy.ndimage import rotate

    H, W = sac.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices(sac.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Annular bounds — empirical: skip central peak, take ring up to ~half the map
    r_inner = max(2.0, 0.05 * min(H, W))
    r_outer = 0.45 * min(H, W)
    mask = (r >= r_inner) & (r <= r_outer)
    if mask.sum() < 50:
        return float("nan")

    base = sac.copy()
    base_vals = base[mask]
    base_mean = base_vals.mean()
    base_std = base_vals.std() + 1e-12

    def corr(angle_deg):
        rot = rotate(base, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
        rv = rot[mask]
        return float(((base_vals - base_mean) * (rv - rv.mean())).mean()
                     / (base_std * (rv.std() + 1e-12)))

    c30 = corr(30); c60 = corr(60); c90 = corr(90); c120 = corr(120); c150 = corr(150)
    return min(c60, c120) - max(c30, c90, c150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-traj", type=int, default=200)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--env-seed", type=int, default=12345)
    parser.add_argument("--n-landmarks", type=int, default=0)
    parser.add_argument("--save-rate-maps", default=None,
                        help="Optional .npz path to save rate maps + scores.")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)

    print(f"Loading {args.checkpoint}...")
    model = _build(args.checkpoint, device=args.device)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                    n_landmarks=args.n_landmarks, seed=args.env_seed)

    print(f"Collecting rate maps over {args.n_traj} trajectories of T={args.T}...")
    rate, counts = collect_rate_maps(model, env, args.T, args.n_traj, device=args.device)
    coverage = (counts > 0).mean()
    print(f"  cell coverage: {coverage:.1%}, "
          f"min/median/max visits per visited cell: "
          f"{int(counts[counts > 0].min())}/{int(np.median(counts[counts > 0]))}/"
          f"{int(counts.max())}")

    print(f"Computing grid scores for {model.n_grid_units} units...")
    scores = []
    for j in range(model.n_grid_units):
        rj = rate[j]
        if rj.std() < 1e-6:
            scores.append(float("nan"))
            continue
        sac = spatial_autocorrelogram(rj)
        scores.append(grid_score(sac))
    scores = np.array(scores)

    valid = scores[~np.isnan(scores)]
    print()
    print(f"=== Grid score summary (n={len(valid)} valid units) ===")
    print(f"  max:        {valid.max():.3f}")
    print(f"  99th pct:   {np.percentile(valid, 99):.3f}")
    print(f"  95th pct:   {np.percentile(valid, 95):.3f}")
    print(f"  median:     {np.median(valid):.3f}")
    print(f"  frac>0.0:   {(valid > 0.0).mean():.2%}")
    print(f"  frac>0.3:   {(valid > 0.3).mean():.2%}")
    print(f"  frac>0.5:   {(valid > 0.5).mean():.2%}")

    if args.save_rate_maps:
        np.savez(args.save_rate_maps, rate=rate, counts=counts, scores=scores)
        print(f"Saved: {args.save_rate_maps}")


if __name__ == "__main__":
    main()
