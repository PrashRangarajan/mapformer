#!/usr/bin/env python3
"""
Clone-structure analysis for MapFormer variants.

Tests whether the model has learned CSCG-style "clones" — distinct internal
representations for the same aliased observation at different grid cells.

For each observation type:
  1. Collect (true_location, model_features) pairs across many trajectories.
  2. Measure how well the true location can be decoded from features.
  3. Visualize per-type clustering (PCA colored by location).

Three representations analyzed per token at observation positions:
  A) θ̂ (post-correction rotation state) — baseline, trivially decodable if
     path integration works (θ̂ literally encodes position-from-start).
  B) Last hidden state before the output projection — carries content + position.
  C) (cos θ̂, sin θ̂) — the SO(2) group elements actually fed to RoPE.

Representation B is the most interesting for clone analysis: features there
should separate into per-cell clusters if the model has learned clone-like
structure. A is a sanity check.

Run:
  python3 -m mapformer.clone_analysis \
      --checkpoint figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt \
      --device cuda
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_kalman import MapFormerWM_InEKF
from mapformer.model_inekf_proper import MapFormerWM_ProperInEKF
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF


def build_model_from_config(config, cls):
    return cls(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        grid_size=config["grid_size"],
    )


def pick_model_class(name):
    if "Level2InEKF" in name:
        return MapFormerWM_Level2InEKF
    if "PredictiveCoding" in name:
        return MapFormerWM_PredictiveCoding
    if "ParallelInEKF" in name:
        return MapFormerWM_ParallelInEKF
    if "ProperInEKF" in name:
        return MapFormerWM_ProperInEKF
    if "InEKF" in name:
        return MapFormerWM_InEKF
    if "WM" in name:
        return MapFormerWM
    return MapFormerEM


# ------------------------------------------------------------
# Instrument MapFormerWM-style models to capture hidden features
# ------------------------------------------------------------

def capture_features_hook(model):
    """Attach a forward hook that captures the output of out_norm (the last
    hidden state just before out_proj). Works for any MapFormerWM-derived model.

    Returns a closure that retrieves the last captured activation.
    """
    captured = {}

    def hook(module, inp, out):
        captured["hidden"] = out.detach()

    handle = model.out_norm.register_forward_hook(hook)

    def get():
        return captured.get("hidden")

    return get, handle


# ------------------------------------------------------------
# Collect trajectories and associated model features
# ------------------------------------------------------------

def collect_data(model, env, n_trajectories, n_steps, device, fixed_start=None):
    """Run trajectories, record at each observation position:
        (obs_type_idx, x, y, theta_hat, hidden).

    fixed_start: (x, y) or None for random starts. Fixing the start makes
    θ̂ comparable across trajectories (same reference frame).
    """
    model.eval()
    get_hidden, handle = capture_features_hook(model)

    obs_types = []
    positions = []
    thetas = []
    hiddens = []

    with torch.no_grad():
        for _ in range(n_trajectories):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(
                n_steps, start=fixed_start
            )
            locs = list(env.visited_locations)  # list of (x,y) per step (length = n_steps)
            tokens_t = tokens.unsqueeze(0).to(device)

            _ = model(tokens_t[:, :-1])  # forward pass, may populate model.last_theta_hat

            # Extract theta_hat (corrected angles if the model has it, else theta_path)
            if hasattr(model, "last_theta_hat"):
                th = model.last_theta_hat[0]  # (L-1, H, NB)
            else:
                # Fall back to path integration (vanilla case)
                x = model.token_emb(tokens_t[:, :-1])
                delta = model.action_to_lie(x)
                cum = torch.cumsum(delta, dim=1)
                th_full = cum * model.path_integrator.omega.unsqueeze(0).unsqueeze(0)
                th = th_full[0]  # (L-1, H, NB)

            hidden = get_hidden()[0]  # (L-1, d_model)

            # Interleaved stream: observation positions are odd indices in
            # the FULL token sequence. But we fed tokens[:-1], so indices are
            # 0..L-2. In the FULL sequence, obs are at odd indices (1, 3, 5, ...).
            # After dropping the last, the observation-position indices are still
            # 1, 3, 5, ... up to L-2.
            L_minus1 = th.shape[0]
            # Step t corresponds to action at position 2t, obs at position 2t+1.
            # We record the state AT the observation position (after the obs token).
            for step_idx, (x_loc, y_loc) in enumerate(locs):
                obs_pos = 2 * step_idx + 1
                if obs_pos >= L_minus1:
                    break
                obs_token_val = tokens[obs_pos].item()
                # Convert unified obs token -> obs type index in [0, K] (K = blank)
                obs_type = obs_token_val - env.obs_offset
                if obs_type < 0 or obs_type >= env.obs_vocab_size:
                    continue  # malformed — skip

                obs_types.append(obs_type)
                positions.append((x_loc, y_loc))
                thetas.append(th[obs_pos].cpu().numpy().flatten())  # (H*NB,)
                hiddens.append(hidden[obs_pos].cpu().numpy())       # (d_model,)

    handle.remove()
    return (
        np.array(obs_types),
        np.array(positions),
        np.array(thetas),
        np.array(hiddens),
    )


# ------------------------------------------------------------
# Analyses
# ------------------------------------------------------------

def per_type_decodability(obs_types, positions, features, env, min_per_type=50):
    """For each obs type, train a linear regressor to predict (x, y) from
    model features. Report R² (1.0 = perfect recovery, 0.0 = no signal).

    Regression rather than classification: much faster than multinomial with
    hundreds of classes, and R² is a cleaner "how much position info is here"
    metric. A random baseline gives R² ≈ 0.
    """
    per_type_r2 = {}
    overall_r2_weighted = 0.0
    overall_n = 0

    for t in range(env.obs_vocab_size):
        mask = obs_types == t
        if mask.sum() < min_per_type:
            continue
        X = features[mask]
        y_xy = positions[mask].astype(np.float32)  # (n, 2)

        unique_cells = np.unique(positions[mask] @ np.array([env.size, 1]))
        if len(unique_cells) < 2:
            continue

        n = len(X)
        idx = np.random.RandomState(0).permutation(n)
        split = int(0.8 * n)
        tr_idx, te_idx = idx[:split], idx[split:]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_xy[tr_idx])
        pred = reg.predict(X_te)
        r2 = max(0.0, r2_score(y_xy[te_idx], pred, multioutput="uniform_average"))

        per_type_r2[t] = (r2, int(mask.sum()), len(unique_cells))
        overall_r2_weighted += r2 * len(te_idx)
        overall_n += len(te_idx)

    mean_r2 = overall_r2_weighted / max(overall_n, 1)
    return mean_r2, per_type_r2


def clone_separation_score(obs_types, positions, features, env, min_per_type=50):
    """Silhouette-like score: for each obs type, how well do features
    separate by true cell? Measured as:
        mean_within = avg feature distance within-same-cell
        mean_between = avg feature distance across-different-cells
    Return (mean_between - mean_within) / mean_between  ∈ (-inf, 1].
    Higher = more clone-like separation.
    """
    grid_n = env.size
    scores = []

    rng = np.random.RandomState(0)
    for t in range(env.obs_vocab_size):
        mask = obs_types == t
        if mask.sum() < min_per_type:
            continue
        X = features[mask]
        y_pos = positions[mask]
        y_class = y_pos[:, 0] * grid_n + y_pos[:, 1]

        # Normalize features for cosine-ish comparison
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Xn = X / norms

        # Sample pairs for efficiency
        n = len(Xn)
        idx1 = rng.randint(0, n, size=min(2000, n * n))
        idx2 = rng.randint(0, n, size=min(2000, n * n))
        same = y_class[idx1] == y_class[idx2]
        diff = ~same
        not_self = idx1 != idx2
        same &= not_self
        diff &= not_self

        if same.sum() < 5 or diff.sum() < 5:
            continue

        d_same = 1.0 - np.sum(Xn[idx1[same]] * Xn[idx2[same]], axis=1)
        d_diff = 1.0 - np.sum(Xn[idx1[diff]] * Xn[idx2[diff]], axis=1)
        score = (d_diff.mean() - d_same.mean()) / (d_diff.mean() + 1e-8)
        scores.append((t, score, int(mask.sum())))

    if not scores:
        return 0.0, []
    mean_score = np.mean([s for _, s, _ in scores])
    return mean_score, scores


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-trajectories", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--fixed-start", action="store_true",
                        help="Use fixed start (32,32) for all trajectories.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]
    name = Path(args.checkpoint).stem
    cls = pick_model_class(name)
    model = build_model_from_config(config, cls)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    env = GridWorld(
        size=config["grid_size"],
        n_obs_types=config["n_obs_types"],
        p_empty=config["p_empty"],
        seed=42,
    )
    start = (config["grid_size"] // 2, config["grid_size"] // 2) if args.fixed_start else None

    print(f"Loaded {name}")
    print(f"Running {args.n_trajectories} trajectories, T={args.n_steps}"
          f"{', fixed start' if args.fixed_start else ', random start'}")

    obs_types, positions, thetas, hiddens = collect_data(
        model, env, args.n_trajectories, args.n_steps, args.device, fixed_start=start
    )
    n_total = len(obs_types)
    print(f"Collected {n_total} observation tokens")
    print(f"  θ̂ feature dim: {thetas.shape[1]}")
    print(f"  hidden feature dim: {hiddens.shape[1]}")

    blank_idx = env.blank_token
    print(f"  {(obs_types == blank_idx).sum()} blank tokens ({(obs_types == blank_idx).mean():.1%})")
    print(f"  {(obs_types != blank_idx).sum()} non-blank")

    # ============================================================
    # Per-type decodability
    # ============================================================
    print("\n" + "=" * 60)
    print("CLONE ANALYSIS — can we decode (x,y) from features, per obs type?")
    print("=" * 60)

    for repr_name, feats in [("θ̂ (path/corrected)", thetas), ("hidden (pre-out)", hiddens)]:
        mean_r2, per_type = per_type_decodability(obs_types, positions, feats, env)
        print(f"\n{repr_name}:")
        print(f"  Mean R² across obs types: {mean_r2:.3f}  (1.0 = perfect position recovery)")
        print(f"  Per-type R²:")
        print(f"    {'type':>4} {'R²':>6} {'n_samples':>10} {'n_unique_cells':>16}")
        for t in sorted(per_type.keys()):
            if t == blank_idx:
                continue
            r2, n, n_uniq = per_type[t]
            print(f"    {t:>4} {r2:>6.3f} {n:>10d} {n_uniq:>16d}")
        if blank_idx in per_type:
            r2, n, n_uniq = per_type[blank_idx]
            print(f"    BLANK ({blank_idx}): R²={r2:.3f}  n={n}  unique_cells={n_uniq}")

    # ============================================================
    # Clone separation (silhouette-like)
    # ============================================================
    print("\n" + "=" * 60)
    print("CLONE SEPARATION SCORE")
    print("=" * 60)
    for repr_name, feats in [("θ̂", thetas), ("hidden", hiddens)]:
        score, _ = clone_separation_score(obs_types, positions, feats, env)
        print(f"  {repr_name:>20s} — separation: {score:.4f}"
              f"  (higher is more clone-like; 0 = no separation)")

    # ============================================================
    # PCA visualization (save first non-blank obs type)
    # ============================================================
    print("\n" + "=" * 60)
    print("PCA VISUALIZATION (hidden features)")
    print("=" * 60)
    # Pick the most common non-blank obs type for visualization
    counts = np.bincount(obs_types, minlength=env.obs_vocab_size)
    counts[blank_idx] = 0  # exclude blank
    target_type = int(np.argmax(counts))
    mask = obs_types == target_type
    if mask.sum() >= 50:
        X = hiddens[mask]
        pca = PCA(n_components=2)
        Z = pca.fit_transform(StandardScaler().fit_transform(X))
        pos = positions[mask]
        print(f"  Obs type {target_type}: {mask.sum()} samples, "
              f"{len(np.unique(pos[:, 0] * env.size + pos[:, 1]))} unique cells")
        print(f"  PC1 var: {pca.explained_variance_ratio_[0]:.3f}  "
              f"PC2 var: {pca.explained_variance_ratio_[1]:.3f}")
        # Save PCA data for downstream plotting
        out_dir = Path(args.checkpoint).parent
        np.savez(
            out_dir / f"clone_pca_type{target_type}.npz",
            pca=Z, positions=pos, obs_type=target_type,
            variance_ratio=pca.explained_variance_ratio_,
        )
        print(f"  Saved: {out_dir}/clone_pca_type{target_type}.npz")

    print("\nDone.")


if __name__ == "__main__":
    main()
