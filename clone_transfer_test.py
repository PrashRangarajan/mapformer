#!/usr/bin/env python3
"""Test 3: Clone-separation transfer test.

The original `clone_analysis.py` measures how cleanly cells emitting the
same observation type cluster apart in feature space, on the TRAINING
environment. PC won that metric (0.619 vs Vanilla 0.573 vs Level1.5 0.395).

Question: does PC's clone-structure win transfer to a held-out environment
(fresh obs_map), or is it a feature of memorising the training obs_map?

Hypothesis: if PC's clean clustering reflects a transferable cognitive-map
property, the lead should hold on a fresh environment. If it reflects
memorisation, the lead should collapse.

Test: run clone analysis on each variant against env_seed=10000 (never seen
in training). Compare separation scores to the in-distribution numbers.

Output:
  CLONE_TRANSFER_TEST.md — table of separation scores: in-dist vs OOD env.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_level15_pc import MapFormerWM_Level15PC


VARIANT_CLS = {
    "Vanilla":   MapFormerWM,
    "Level1":    MapFormerWM_ParallelInEKF,
    "Level15":   MapFormerWM_Level15InEKF,
    "PC":        MapFormerWM_PredictiveCoding,
    "Level15PC": MapFormerWM_Level15PC,
}


def build_model(variant, ckpt_path, device="cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(
        vocab_size=cfg.get("vocab_size"),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 2),
        n_layers=cfg.get("n_layers", 1),
        grid_size=cfg.get("grid_size", 64),
    )
    m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def collect_features_by_obs_and_cell(model, env, T=128, n_trials=300, device="cuda"):
    """For each (obs_type, cell) pair, collect feature vectors at obs token positions.

    Returns dict[obs_type][cell] -> list of feature vectors (using θ̂ representation).
    """
    by_obs = defaultdict(lambda: defaultdict(list))

    captured = {}
    def hook(module, inp, out):
        captured["theta_hat"] = (
            module.last_theta_hat if hasattr(module, "last_theta_hat") else None
        )

    for trial in range(n_trials):
        tokens, _, _ = env.generate_trajectory(T)
        visited = env.visited_locations
        tt = tokens.unsqueeze(0).to(device)
        try:
            _ = model(tt[:, :-1])
        except Exception:
            continue

        # Use last_theta_hat if available, else compute θ_path
        if hasattr(model, "last_theta_hat") and model.last_theta_hat is not None:
            theta = model.last_theta_hat[0].cpu().numpy()  # (L, H, NB)
        else:
            x = model.token_emb(tt[:, :-1])
            delta = model.action_to_lie(x)
            cum = torch.cumsum(delta, dim=1)
            theta = (cum * model.path_integrator.omega.unsqueeze(0).unsqueeze(0))[0].cpu().numpy()

        L = theta.shape[0]
        # Iterate over OBS positions (odd indices). Each obs position is
        # at cell visited[t] where 2*t+1 is the position.
        for t in range(L // 2):
            obs_pos = 2 * t + 1
            if obs_pos >= L:
                continue
            cell = visited[t]
            tok = int(tt[0, obs_pos].item())
            obs_id = tok - env.obs_offset
            if obs_id < 0 or obs_id == env.blank_token:
                continue  # skip non-obs and blank
            if obs_id >= env.n_obs_types:
                continue  # skip landmarks (one-shot, no clone structure)
            # use cos/sin of θ̂ as the feature
            feat = np.concatenate([np.cos(theta[obs_pos].flatten()),
                                   np.sin(theta[obs_pos].flatten())])
            by_obs[obs_id][cell].append(feat)
    return by_obs


def cosine_separation(by_obs):
    """For each obs type with ≥2 distinct cells visited, compute the separation
    score: (between-cell − within-cell) / between-cell, using cosine distance.

    Returns mean separation score across obs types.
    """
    scores = []
    for obs_id, cell_dict in by_obs.items():
        cells = [c for c, fs in cell_dict.items() if len(fs) >= 2]
        if len(cells) < 2:
            continue
        # mean feature per cell
        cell_means = {c: np.stack(cell_dict[c]).mean(axis=0) for c in cells}
        # within-cell: average distance from each feature to its cell mean
        within = []
        for c in cells:
            mean = cell_means[c]
            for f in cell_dict[c]:
                a = f / (np.linalg.norm(f) + 1e-8)
                b = mean / (np.linalg.norm(mean) + 1e-8)
                within.append(1 - float(a @ b))
        within = np.mean(within)
        # between-cell: average distance between cell means
        between = []
        for i, ci in enumerate(cells):
            for cj in cells[i+1:]:
                a = cell_means[ci] / (np.linalg.norm(cell_means[ci]) + 1e-8)
                b = cell_means[cj] / (np.linalg.norm(cell_means[cj]) + 1e-8)
                between.append(1 - float(a @ b))
        if not between:
            continue
        between = np.mean(between)
        if between > 0:
            scores.append((between - within) / between)
    return float(np.mean(scores)) if scores else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="lm200")
    p.add_argument("--variants", nargs="+",
                   default=["Vanilla", "Level1", "Level15", "PC", "Level15PC"])
    p.add_argument("--seed", type=int, default=0,
                   help="model seed (training env seed)")
    p.add_argument("--ood-env-seed", type=int, default=10000,
                   help="env seed for the OOD test (fresh obs_map)")
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--output", default="CLONE_TRANSFER_TEST.md")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    n_lm = 200 if args.config == "lm200" else 0

    in_dist = {}
    ood = {}

    for variant in args.variants:
        ckpt = runs / f"{variant}_{args.config}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"[skip] {variant}: no ckpt", file=sys.stderr); continue
        try:
            m, cfg = build_model(variant, ckpt, args.device)
        except Exception as e:
            print(f"[skip] {variant}: {e}", file=sys.stderr); continue

        # In-dist (training env)
        env_indist = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", n_lm),
            seed=args.seed,  # training env seed
        )
        np.random.seed(args.seed * 7919); torch.manual_seed(args.seed * 7919)
        feats_in = collect_features_by_obs_and_cell(
            m, env_indist, args.T, args.n_trials, args.device,
        )
        in_dist[variant] = cosine_separation(feats_in)

        # OOD (fresh env)
        env_ood = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", n_lm),
            seed=args.ood_env_seed,
        )
        np.random.seed(args.ood_env_seed * 7919)
        torch.manual_seed(args.ood_env_seed * 7919)
        feats_ood = collect_features_by_obs_and_cell(
            m, env_ood, args.T, args.n_trials, args.device,
        )
        ood[variant] = cosine_separation(feats_ood)

        del m; torch.cuda.empty_cache()
        print(f"  {variant}: in-dist sep={in_dist[variant]:.3f}, "
              f"OOD sep={ood[variant]:.3f}", file=sys.stderr)

    # Markdown
    md = ["# Test 3: Clone-separation transfer test\n"]
    md.append(f"Config: **{args.config}**, model seed: {args.seed}, "
              f"OOD env seed: {args.ood_env_seed}, T={args.T}, "
              f"trials={args.n_trials}\n")
    md.append("Higher separation score = cleaner per-cell clustering of θ̂ "
              "for the same observation type. Drop from in-dist to OOD = "
              "evidence of memorisation rather than abstract cognitive-map "
              "structure.\n")
    md.append("| Variant | In-dist sep | OOD sep | Δ (drop) |")
    md.append("|---|---|---|---|")
    for v in args.variants:
        if v not in in_dist:
            continue
        a, b = in_dist[v], ood[v]
        delta = a - b
        md.append(f"| {v} | {a:+.3f} | {b:+.3f} | {delta:+.3f} |")
    md.append("\n**Decision rule:**\n")
    md.append("- If PC's lead persists OOD (Δ small for PC): clone-structure "
              "is a real, transferable feature.")
    md.append("- If PC's lead collapses OOD (Δ large for PC): PC's win is "
              "memorisation of the training env, not abstract cognitive-map "
              "structure.")
    md.append("\n*Auto-generated by `clone_transfer_test.py`.*\n")
    Path(args.output).write_text("\n".join(md))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
