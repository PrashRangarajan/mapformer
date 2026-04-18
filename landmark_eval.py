#!/usr/bin/env python3
"""
Landmark-aware evaluation: accuracy + NLL + per-cell-type breakdown.

For each model, evaluate on trajectories and report:
  - Overall revisit accuracy
  - Overall revisit NLL (calibration proxy)
  - Per-cell-type breakdown: landmark / regular-obs / blank
  - Length generalisation (T=128, 512, 2048)

Kalman / PC methods should shine on landmark cells (sharp measurements) and
at long OOD lengths where drift would be catastrophic without corrections.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_kalman import MapFormerWM_InEKF
from mapformer.model_inekf_proper import MapFormerWM_ProperInEKF
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding


def build_model_from_config(config, cls):
    return cls(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        grid_size=config["grid_size"],
    )


def pick_model_class(name):
    if "PredictiveCoding" in name: return MapFormerWM_PredictiveCoding
    if "ParallelInEKF" in name: return MapFormerWM_ParallelInEKF
    if "ProperInEKF" in name: return MapFormerWM_ProperInEKF
    if "InEKF" in name: return MapFormerWM_InEKF
    if "WM" in name: return MapFormerWM
    return MapFormerEM


def classify_target(target_token, env):
    """Return 'landmark', 'blank', or 'regular' given a unified token value."""
    if target_token == env.unified_blank:
        return "blank"
    if target_token >= env.first_landmark_unified:
        return "landmark"
    return "regular"


def eval_checkpoint(model, env, n_steps, n_trials, device, seed=0):
    """Return dict with overall + per-cell-type accuracy and NLL."""
    model.eval()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # accumulators
    cats = ["landmark", "regular", "blank", "overall"]
    correct = {c: 0 for c in cats}
    total = {c: 0 for c in cats}
    nll_sum = {c: 0.0 for c in cats}

    with torch.no_grad():
        for _ in range(n_trials):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)
            tokens_t = tokens.unsqueeze(0).to(device)
            revisit_mask_t = revisit_mask.unsqueeze(0).to(device)

            logits = model(tokens_t[:, :-1])  # (1, 2T-1, V)
            log_probs = F.log_softmax(logits, dim=-1)
            preds = log_probs.argmax(-1)[0]  # (2T-1,)
            targets = tokens_t[0, 1:]        # (2T-1,)
            mask = revisit_mask_t[0, 1:]     # (2T-1,) bool

            if mask.sum() == 0:
                continue

            # Indices of revisit targets
            revisit_idx = mask.nonzero(as_tuple=True)[0]
            for idx in revisit_idx.tolist():
                tgt = targets[idx].item()
                pred = preds[idx].item()
                # NLL contribution of this position
                nll = -log_probs[0, idx, tgt].item()
                cat = classify_target(tgt, env)

                correct[cat] += int(pred == tgt)
                total[cat] += 1
                nll_sum[cat] += nll

                correct["overall"] += int(pred == tgt)
                total["overall"] += 1
                nll_sum["overall"] += nll

    results = {}
    for c in cats:
        if total[c] == 0:
            results[c] = {"acc": float("nan"), "nll": float("nan"), "n": 0}
        else:
            results[c] = {
                "acc": correct[c] / total[c],
                "nll": nll_sum[c] / total[c],
                "n": total[c],
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-steps", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--n-trials", type=int, default=200)
    args = parser.parse_args()

    # Load environment (shared across all checkpoints — they must agree on n_landmarks)
    first_ckpt = torch.load(args.checkpoints[0], map_location=args.device, weights_only=False)
    cfg = first_ckpt["config"]
    n_landmarks = cfg.get("n_landmarks", 0)
    env = GridWorld(
        size=cfg["grid_size"],
        n_obs_types=cfg["n_obs_types"],
        p_empty=cfg["p_empty"],
        n_landmarks=n_landmarks,
        seed=42,
    )
    print(f"Environment: grid={cfg['grid_size']}, K={cfg['n_obs_types']}, "
          f"p_empty={cfg['p_empty']}, n_landmarks={n_landmarks}")
    print(f"Unified vocab size: {env.unified_vocab_size}")
    print()

    # Load all models
    models = {}
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        name = Path(path).stem
        cls = pick_model_class(name)
        m = build_model_from_config(ckpt["config"], cls)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(args.device)
        models[name] = m

    # Evaluate at each T
    for T in args.n_steps:
        print("=" * 88)
        print(f"T = {T}  ({args.n_trials} trials)")
        print("=" * 88)

        for name, m in models.items():
            res = eval_checkpoint(m, env, T, args.n_trials, args.device, seed=0)
            print(f"\n  {name}")
            for cat in ["overall", "landmark", "regular", "blank"]:
                r = res[cat]
                if r["n"] > 0:
                    print(f"    {cat:>9s}  acc={r['acc']:.3f}  NLL={r['nll']:.3f}  n={r['n']}")
                else:
                    print(f"    {cat:>9s}  (no samples)")
        print()


if __name__ == "__main__":
    main()
