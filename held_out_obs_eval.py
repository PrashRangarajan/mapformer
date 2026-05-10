#!/usr/bin/env python3
"""Held-out observation vocabulary evaluation.

Tests structural transfer when the *content distribution* shifts at test
time. Models train with K=16 distinct observation types appearing roughly
uniformly across cells. At test we restrict which obs IDs can appear,
forcing the model to rely on a content distribution it never trained on
as the only signal.

Two restriction modes:

  Mode A — Restrict-by-blanking. Cells whose original obs ID was outside
           the allowed subset get the blank token instead. This INCREASES
           the blank fraction AND reduces obs vocabulary diversity. Tests
           joint effect of higher aliasing + content sparsity.

  Mode B — Restrict-by-reassignment. Cells whose original obs ID was
           outside the allowed subset get reassigned to a random ID
           within the allowed subset. Keeps the obs/blank fraction
           constant but increases per-obs-type aliasing (more cells
           share the same obs ID).

Mode B isolates the pure vocabulary-shrinkage effect; Mode A exercises
both axes at once. Both are run by default.

The held-out subset is chosen as the contiguous range {0, 1, ..., K'-1}
for K' in {1, 2, 4, 8, 16}. Since training samples obs IDs uniformly,
the choice of which K' indices to keep is (statistically) immaterial.

Usage:
  python3 -m mapformer.held_out_obs_eval \
      --runs-dir runs --config clean --output HELD_OUT_OBS_clean.md
  python3 -m mapformer.held_out_obs_eval \
      --runs-dir runs --config lm200 --output HELD_OUT_OBS_lm200.md
"""

import argparse
import sys
import datetime
import statistics as st
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


# ----------------------------------------------------------------------
# Restricted-vocabulary environments
# ----------------------------------------------------------------------

def restrict_obs_map_blanking(env: GridWorld, allowed_ids: list[int]) -> GridWorld:
    """In-place: cells with obs IDs outside `allowed_ids` become blank.

    Landmark cells (IDs >= first_landmark_rel) are preserved.
    """
    allowed = set(allowed_ids)
    new_map = env.obs_map.clone()
    # Mask: cell is a regular-obs cell with ID not in allowed_ids
    for i in range(env.n_obs_types):
        if i not in allowed:
            new_map[env.obs_map == i] = env.blank_token
    env.obs_map = new_map
    return env


def restrict_obs_map_reassign(env: GridWorld, allowed_ids: list[int],
                               rng: np.random.RandomState) -> GridWorld:
    """In-place: cells with obs IDs outside `allowed_ids` get reassigned
    to a uniformly-random ID drawn from `allowed_ids`.

    Landmark cells (IDs >= first_landmark_rel) are preserved.
    """
    allowed = set(allowed_ids)
    allowed_arr = np.array(sorted(allowed))
    new_map = env.obs_map.clone().numpy()
    # Mask: cell is a regular-obs cell with ID not in allowed_ids
    out_of_set = np.zeros(new_map.shape, dtype=bool)
    for i in range(env.n_obs_types):
        if i not in allowed:
            out_of_set |= (new_map == i)
    n_replace = int(out_of_set.sum())
    if n_replace > 0:
        new_map[out_of_set] = rng.choice(allowed_arr, size=n_replace)
    env.obs_map = torch.from_numpy(new_map).long()
    return env


# ----------------------------------------------------------------------
# Model loading + per-trial eval (mirrors zero_shot_eval.py)
# ----------------------------------------------------------------------

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


def eval_revisit(model, env, T: int, n_trials: int, device: str = "cuda"):
    correct = total = 0
    nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).to(device)
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]
            tgts = tt[0, 1:]
            mask = rm[1:].to(device)
            if mask.sum() == 0:
                continue
            correct += (preds[mask] == tgts[mask]).sum().item()
            total += mask.sum().item()
            idx = torch.arange(lp.shape[1], device=device)[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    if total == 0:
        return None, None
    return correct / total, nll_sum / total


def make_env(cfg, seed, train_n_lm):
    return GridWorld(
        size=cfg.get("grid_size", 64),
        n_obs_types=cfg.get("n_obs_types", 16),
        p_empty=cfg.get("p_empty", 0.5),
        n_landmarks=cfg.get("n_landmarks", train_n_lm),
        seed=seed,
    )


# ----------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------

def _fmt(vals, ndigits=3):
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]:.{ndigits}f}"
    return f"{st.mean(vals):.{ndigits}f}±{st.pstdev(vals):.{ndigits}f}"


def _table(rows, header):
    sep = "|" + "|".join("---" for _ in header) + "|"
    out = ["| " + " | ".join(header) + " |", sep]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config", default="clean", help="clean|noise|lm200")
    parser.add_argument("--variants", nargs="+",
                        default=["Vanilla", "RoPE", "Level1", "Level15", "PC"])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-test-seeds", type=int, default=3)
    parser.add_argument("--test-seed-base", type=int, default=20000)
    parser.add_argument("--k-prime", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16],
                        help="active obs vocabulary sizes K' to test")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--mode", choices=["both", "blank", "reassign"],
                        default="both",
                        help="restriction mode: blanking, reassignment, or both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="HELD_OUT_OBS.md")
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    train_n_lm = 200 if args.config == "lm200" else 0
    test_seeds = list(range(args.test_seed_base,
                            args.test_seed_base + args.n_test_seeds))

    out_lines = []
    out_lines.append("# Held-out Observation Vocabulary Evaluation\n")
    out_lines.append(f"Generated: {datetime.datetime.now()}\n")
    out_lines.append(f"Config: **{args.config}**, training n_landmarks: {train_n_lm}")
    out_lines.append(f"Model seeds: {args.model_seeds}, fresh test seeds per model: "
                     f"{args.n_test_seeds} (base {args.test_seed_base})")
    out_lines.append(f"Restriction mode(s): **{args.mode}**")
    out_lines.append(f"K' values: {args.k_prime}, lengths: {args.lengths}\n")
    out_lines.append("Each cell: mean ± std over (model seeds × fresh test seeds) "
                     f"= {len(args.model_seeds) * args.n_test_seeds} runs.\n")

    # All results keyed by (mode, T, variant, K_prime) -> list of values
    results: dict = {}

    modes = ["blank", "reassign"] if args.mode == "both" else [args.mode]

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
                for kp in args.k_prime:
                    allowed = list(range(kp))  # contiguous subset {0..kp-1}
                    for mode in modes:
                        # Fresh env each (ts, kp, mode) to avoid in-place pollution
                        env = make_env(cfg, ts, train_n_lm)
                        if mode == "blank":
                            restrict_obs_map_blanking(env, allowed)
                        else:
                            rng = np.random.RandomState(ts * 100 + kp)
                            restrict_obs_map_reassign(env, allowed, rng)

                        for T in args.lengths:
                            np.random.seed(ts * 7919 + T + kp)
                            torch.manual_seed(ts * 7919 + T + kp)
                            a, nll = eval_revisit(model, env, T, args.n_trials,
                                                   device=args.device)
                            key = (mode, T, variant, kp)
                            if key not in results:
                                results[key] = {"acc": [], "nll": []}
                            if a is not None:
                                results[key]["acc"].append(a)
                                results[key]["nll"].append(nll)
                            print(f"  {variant} ms={ms} ts={ts} mode={mode} "
                                  f"K'={kp} T={T}: "
                                  f"acc={a if a is None else f'{a:.3f}'}",
                                  file=sys.stderr, flush=True)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Tables: one section per mode, two sub-tables (acc, NLL) per length
    # ----------------------------------------------------------------
    mode_titles = {
        "blank":    "Mode A — Restrict-by-blanking (held-out IDs become blank)",
        "reassign": "Mode B — Restrict-by-reassignment (held-out IDs reassigned within allowed)",
    }

    for mode in modes:
        out_lines.append(f"\n## {mode_titles[mode]}\n")
        for T in args.lengths:
            out_lines.append(f"### T = {T}\n")
            header_acc = ["Variant"] + [f"K'={kp} acc" for kp in args.k_prime]
            header_nll = ["Variant"] + [f"K'={kp} NLL" for kp in args.k_prime]
            rows_acc, rows_nll = [], []
            for variant in args.variants:
                row_a = [variant]
                row_n = [variant]
                for kp in args.k_prime:
                    key = (mode, T, variant, kp)
                    if key in results:
                        row_a.append(_fmt(results[key]["acc"]))
                        row_n.append(_fmt(results[key]["nll"]))
                    else:
                        row_a.append("N/A")
                        row_n.append("N/A")
                rows_acc.append(row_a)
                rows_nll.append(row_n)
            out_lines.append("**Accuracy**\n")
            out_lines.append(_table(rows_acc, header_acc))
            out_lines.append("\n**NLL**\n")
            out_lines.append(_table(rows_nll, header_nll))
            out_lines.append("")

    out_lines.append("\n---")
    out_lines.append("*Auto-generated by `held_out_obs_eval.py`.*")

    text = "\n".join(out_lines) + "\n"
    Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
