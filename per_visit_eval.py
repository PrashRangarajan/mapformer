#!/usr/bin/env python3
"""Per-visit-count accuracy evaluation (the canonical TEM/TEM-T metric).

For each prediction position, we know:
  (a) which (x, y) cell the model is at
  (b) how many times THAT cell has been visited so far in the trajectory

We bin predictions by visit count k and compute per-bin accuracy:

  acc(k) = P[correct | this is the k-th visit to this cell]

A model that learns the cognitive-map structure should jump from chance
at k=1 (never seen this cell, observation looks random) to high accuracy
at k=2 (one prior visit suffices to memorize the obs at that cell). A
model that just averages over revisits would have a flat curve.

This is the test that distinguishes "learns structure" from "memorizes
one specific map". TEM (Whittington et al. 2020, Cell) and TEM-T
(Whittington et al. 2022) report this curve as their headline
generalization metric.

Usage:
  python3 -m mapformer.per_visit_eval \
      --runs-dir runs --config lm200 \
      --variants Vanilla VanillaEM Level15 Level15EM MambaLike LSTM \
      --model-seeds 0 1 2 --n-test-seeds 5 \
      --max-visits 8 \
      --output PER_VISIT_lm200.md
"""

import argparse
import sys
import datetime
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_ablations import ABLATIONS
from mapformer.model_baselines_extra import EXTRA_BASELINES


VARIANT_CLS = {
    "Vanilla":    MapFormerWM,
    "VanillaEM":  MapFormerEM,
    "Level1":     MapFormerWM_ParallelInEKF,
    "Level15":    MapFormerWM_Level15InEKF,
    "Level15EM":  MapFormerEM_Level15InEKF,
    "Level2":     MapFormerWM_Level2InEKF,
    "PC":         MapFormerWM_PredictiveCoding,
    "RoPE":       MapFormerWM_RoPE,
    **ABLATIONS,
    **EXTRA_BASELINES,
}


def build_model(variant: str, ckpt_path: Path, device: str = "cuda"):
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
    m = m.to(device).eval()
    return m, cfg


def eval_per_visit(model, env, T: int, n_trials: int, max_visits: int,
                   device: str = "cuda"):
    """Returns dict[visit_count] -> {"correct": int, "total": int}."""
    by_visit = {k: {"correct": 0, "total": 0} for k in range(1, max_visits + 1)}
    by_visit["over_max"] = {"correct": 0, "total": 0}

    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, _ = env.generate_trajectory(T)
            visited = env.visited_locations  # [(x, y), ...] length T
            tt = tokens.unsqueeze(0).to(device)
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0].cpu().numpy()  # (2T-1,)
            tgts = tt[0, 1:].cpu().numpy()

            visit_count = defaultdict(int)
            # Step t in 0..T-1; obs token is at index 2t+1; so its prediction
            # comes from input index 2t. The "current cell" is visited[t].
            for t in range(T):
                cell = visited[t]
                visit_count[cell] += 1
                k = visit_count[cell]
                # Prediction position in the (2T-1) length output:
                pred_idx = 2 * t  # logit-row that PREDICTS tokens[2t+1]
                if pred_idx >= preds.shape[0]:
                    continue
                bin_key = k if k <= max_visits else "over_max"
                by_visit[bin_key]["total"] += 1
                if preds[pred_idx] == tgts[pred_idx]:
                    by_visit[bin_key]["correct"] += 1
    return by_visit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="lm200", help="clean|noise|lm200")
    p.add_argument("--variants", nargs="+",
                   default=["Vanilla", "VanillaEM", "Level15", "Level15EM",
                            "MambaLike", "LSTM"])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-test-seeds", type=int, default=3,
                   help="fresh obs_map seeds per model")
    p.add_argument("--test-seed-base", type=int, default=10000)
    p.add_argument("--T", type=int, default=512,
                   help="trajectory length (longer => more revisits per cell)")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--max-visits", type=int, default=8,
                   help="bin visit counts above this into 'over_max'")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="PER_VISIT.md")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    train_n_lm = 200 if args.config == "lm200" else 0
    test_seeds = list(range(args.test_seed_base,
                            args.test_seed_base + args.n_test_seeds))

    # Variant -> visit_count -> list of accuracies (one per (model_seed, test_seed))
    results = {v: {k: [] for k in list(range(1, args.max_visits + 1)) + ["over_max"]}
               for v in args.variants}

    for variant in args.variants:
        for ms in args.model_seeds:
            ckpt = runs / f"{variant}_{args.config}" / f"seed{ms}" / f"{variant}.pt"
            if not ckpt.exists():
                continue
            try:
                model, cfg = build_model(variant, ckpt, args.device)
            except Exception as e:
                print(f"[skip] {variant} seed{ms}: {e}", file=sys.stderr)
                continue
            for ts in test_seeds:
                env = GridWorld(
                    size=cfg.get("grid_size", 64),
                    n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5),
                    n_landmarks=cfg.get("n_landmarks", train_n_lm),
                    seed=ts,
                )
                np.random.seed(ts * 7919)
                torch.manual_seed(ts * 7919)
                bv = eval_per_visit(model, env, args.T, args.n_trials,
                                    args.max_visits, device=args.device)
                if bv is None:
                    continue
                for k, d in bv.items():
                    if d["total"] > 0:
                        results[variant][k].append(d["correct"] / d["total"])
            print(f"  done {variant} ms={ms}", file=sys.stderr)

    # Write markdown
    out_lines = [
        f"# Per-Visit-Count Accuracy (TEM-style generalization curve)\n",
        f"Generated: {datetime.datetime.now()}\n",
        f"Config: **{args.config}**, T={args.T}, fresh obs_map seeds: "
        f"{args.n_test_seeds} per model, model seeds: {args.model_seeds}\n",
        f"Variants: {', '.join(args.variants)}\n",
        f"\nEach cell: mean ± std over (model_seeds × test_seeds) "
        f"= {len(args.model_seeds) * args.n_test_seeds} trajectories per bin.\n",
        f"\n**Per-visit-count accuracy.** Visit count k = how many times "
        f"the current cell has been visited so far in the trajectory.\n",
        f"- k=1 = first visit (chance prediction; observation random for the model)",
        f"- k=2 = first revisit (one prior visit; tests one-shot generalization)",
        f"- k≥3 = repeated revisits (tests memorization / accumulation)\n",
    ]

    bins = list(range(1, args.max_visits + 1)) + ["over_max"]
    header = ["Variant"] + [f"k={k}" for k in bins]
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    out_lines.append("| " + " | ".join(header) + " |")
    out_lines.append(sep)

    def fmt(vals):
        if not vals: return "—"
        if len(vals) == 1: return f"{vals[0]:.3f}"
        return f"{st.mean(vals):.3f}±{st.pstdev(vals):.3f}"

    for v in args.variants:
        row = [v] + [fmt(results[v][k]) for k in bins]
        out_lines.append("| " + " | ".join(row) + " |")

    out_lines.append("\n---\n*Auto-generated by `per_visit_eval.py`.*\n")

    Path(args.output).write_text("\n".join(out_lines))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
