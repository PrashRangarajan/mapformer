"""Probe a trained continuous-nav model's grid layer for hex emergence.

Adapted from probe_hex.py to handle continuous (x, y) positions instead
of discrete cells. Strategy: bin continuous positions into a fine grid,
average the ReLU bottleneck activations per bin, compute Sargolini grid
score per unit on the resulting rate maps.

Usage:
    python3 -m mapformer.probe_hex_continuous \
        --checkpoint runs/cnav/Level15/seed0/Level15.pt \
        --device cuda --n-traj 200 --T 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .continuous_nav import ContinuousNav2D
from .model_continuous import (
    MapFormerWM_Continuous, MapFormerWM_Continuous_Level15,
    MapFormerEM_Continuous, MapFormerEM_Continuous_Level15,
)

VARIANT_CLS = {
    "Vanilla":   MapFormerWM_Continuous,
    "Level15":   MapFormerWM_Continuous_Level15,
    "VanillaEM": MapFormerEM_Continuous,
    "Level15EM": MapFormerEM_Continuous_Level15,
}


def _build(ckpt_path: str, device: str = "cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c.get("config", {})
    variant = c.get("variant", "Level15")
    cls = VARIANT_CLS[variant]
    extra = {}
    if variant == "Level15":
        extra["log_R_init_bias"] = 0.0
    elif variant == "Level15EM":
        extra["log_R_init_bias"] = 3.0
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
    return m.to(device).eval(), cfg


def collect_rate_maps(model, env, T: int, n_traj: int, n_bins: int,
                      device: str = "cuda"):
    """Bin continuous positions into n_bins x n_bins, average activations.

    Returns:
        rate:   (n_grid, n_bins, n_bins) per-unit rate map
        counts: (n_bins, n_bins) visits per bin
    """
    if model.n_grid_units == 0:
        raise ValueError("Model has no grid bottleneck — train with --n-grid-units > 0")

    n_grid = model.n_grid_units
    bin_size = env.size / n_bins
    sums = np.zeros((n_grid, n_bins, n_bins), dtype=np.float64)
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    rng = np.random.RandomState(0)

    with torch.no_grad():
        for _ in range(n_traj):
            actions, _, positions, _, obs = env.generate_trajectory(T, rng=rng)
            A = torch.from_numpy(actions[None]).to(device)
            O = torch.from_numpy(obs    [None]).to(device)
            _ = model(A, O)
            g = model.last_grid_activations[0].cpu().numpy()  # (2T, n_grid)

            # Activations at action positions (predict next obs at p_t)
            for t in range(T):
                px, py = positions[t]
                bx = int(min(n_bins - 1, max(0, px / bin_size)))
                by = int(min(n_bins - 1, max(0, py / bin_size)))
                sums[:, bx, by] += g[2 * t]   # action position 2t
                counts[bx, by] += 1.0

    counts_safe = np.maximum(counts, 1.0)
    rate = sums / counts_safe[None]
    return rate, counts


def spatial_autocorrelogram(rate_map: np.ndarray) -> np.ndarray:
    r = rate_map - rate_map.mean()
    F = np.fft.fft2(r)
    sac = np.fft.ifft2(F * np.conj(F)).real
    return np.fft.fftshift(sac)


def grid_score(sac: np.ndarray) -> float:
    from scipy.ndimage import rotate

    H, W = sac.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices(sac.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

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
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--n-bins", type=int, default=64,
                        help="Number of spatial bins per side for rate map")
    parser.add_argument("--save-rate-maps", default=None)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)

    print(f"Loading {args.checkpoint}...")
    model, cfg = _build(args.checkpoint, device=args.device)
    print(f"  variant={cfg.get('variant', '?')}, n_grid_units={model.n_grid_units}, "
          f"size={cfg.get('size', '?')}, npc={cfg.get('n_place_cells', '?')}")

    # Eval env: same params as training (so the spatial scale is consistent)
    env = ContinuousNav2D(
        size=cfg.get("size", 64.0),
        n_place_cells=cfg.get("n_place_cells", 256),
        sigma_E=cfg.get("sigma_E", 1.5),
        sigma_I=cfg.get("sigma_I", 3.0),
        v_mean=cfg.get("v_mean", 0.7),
        v_std=cfg.get("v_std", 0.3),
        omega_std=cfg.get("omega_std", 0.5),
        v_noise_std=cfg.get("v_noise_std", 0.0),
        omega_noise_std=cfg.get("omega_noise_std", 0.0),
        seed=42,  # different from training
    )

    print(f"Collecting rate maps over {args.n_traj} trajectories of T={args.T}...")
    rate, counts = collect_rate_maps(model, env, args.T, args.n_traj,
                                     n_bins=args.n_bins, device=args.device)
    coverage = (counts > 0).mean()
    print(f"  bin coverage: {coverage:.1%}")

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
