#!/usr/bin/env python3
"""Zero-shot structural-transfer evaluation.

The project's stated goal is zero-shot generalization to fresh problem
instances with the same underlying structure (SO(2) toroidal navigation,
interleaved action/observation streams). The existing `long_sequence_eval.py`
tests one fresh obs_map per model seed; this script exercises the
zero-shot axis more systematically:

  Axis 1 — Fresh obs_map seeds. Average over many test seeds to estimate
           how well each model transfers to a brand-new observation map
           (fresh assignment of obs IDs to cells, fresh landmark layout).
           This is the primary zero-shot test.

  Axis 2 — Action-distribution bias. The trained models saw uniform-random
           actions. At test time, sample actions from biased distributions
           (mostly-East, mostly-NS, etc.) and check whether path integration
           still works on out-of-training-distribution trajectories.

  Axis 3 — Landmark-density transfer (only for landmark-trained models).
           A model trained with n_landmarks=200 has vocab slots for IDs
           0..199. At test we can use any subset (n_landmarks <= train).
           Tests whether the per-token R_t head's "informativeness"
           generalizes across landmark densities.

Constraints from the architecture (cannot vary at test time):
  - grid_size is baked into the learned omega frequencies.
  - K (n_obs_types) is baked into the embedding table layout.
  - n_landmarks at test cannot exceed training n_landmarks.

Usage:
  python3 -m mapformer.zero_shot_eval \
      --runs-dir runs --config clean --output ZERO_SHOT_TRANSFER.md
  python3 -m mapformer.zero_shot_eval \
      --runs-dir runs --config lm200 --include-bias --include-lm-sweep \
      --output ZERO_SHOT_TRANSFER_lm200.md
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
# Trajectory generators with optional action-distribution bias
# ----------------------------------------------------------------------

# Biased action distributions over [N, S, W, E] = [0, 1, 2, 3].
ACTION_BIASES = {
    "uniform":        np.array([0.25, 0.25, 0.25, 0.25]),
    "mostly_east":    np.array([0.10, 0.10, 0.10, 0.70]),
    "mostly_NS":      np.array([0.40, 0.40, 0.10, 0.10]),
    "diagonal_NE":    np.array([0.40, 0.10, 0.10, 0.40]),
}


def _gen_biased_trajectory(env: GridWorld, n_steps: int, action_probs: np.ndarray):
    """Variant of GridWorld.generate_trajectory with a biased action distribution.

    Mirrors the upstream method but samples actions from `action_probs` instead
    of uniformly. Returns the same (tokens, obs_mask, revisit_mask) triple.
    """
    x = np.random.randint(0, env.size)
    y = np.random.randint(0, env.size)

    tokens = []
    is_revisit = []
    seen = set()

    t = 0
    while t < n_steps:
        a = int(np.random.choice(env.N_ACTIONS, p=action_probs))
        k = np.random.randint(1, 11)
        for _ in range(k):
            if t >= n_steps:
                break
            dx, dy = env.ACTION_DELTAS[a]
            x = (x + dx) % env.size
            y = (y + dy) % env.size
            tokens.append(a + env.action_offset)
            obs_idx = env.obs_map[x, y].item()
            tokens.append(obs_idx + env.obs_offset)
            is_revisit.append((x, y) in seen)
            seen.add((x, y))
            t += 1

    tokens = torch.tensor(tokens, dtype=torch.long)
    obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
    obs_mask[1::2] = True
    revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
    for step_idx, rev in enumerate(is_revisit):
        if rev:
            revisit_mask[2 * step_idx + 1] = True
    return tokens, obs_mask, revisit_mask


# ----------------------------------------------------------------------
# Model loading + per-trial eval
# ----------------------------------------------------------------------

def build_model(variant: str, ckpt_path: Path, device: str = "cuda"):
    """Load a checkpoint and instantiate its model from the saved config."""
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


def eval_revisit(model, env, T: int, n_trials: int, action_probs=None,
                 device: str = "cuda"):
    """Compute (acc, NLL) over `n_trials` fresh trajectories of length T.

    If `action_probs` is None, uses the env's default uniform sampler.
    Otherwise samples actions from the biased distribution.
    """
    correct = total = 0
    nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            if action_probs is None:
                tokens, _, rm = env.generate_trajectory(T)
            else:
                tokens, _, rm = _gen_biased_trajectory(env, T, action_probs)
            tt = tokens.unsqueeze(0).to(device)
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None, None  # OOM at very long T
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


# ----------------------------------------------------------------------
# Markdown formatting helpers
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
    parser.add_argument("--n-test-seeds", type=int, default=5,
                        help="how many fresh obs_map seeds per model")
    parser.add_argument("--test-seed-base", type=int, default=10000,
                        help="fresh obs_map seeds = base, base+1, ..., base+N-1")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[128, 512, 1024, 2048])
    parser.add_argument("--n-trials-128", type=int, default=100)
    parser.add_argument("--n-trials-long", type=int, default=30)
    parser.add_argument("--include-bias", action="store_true",
                        help="add Axis 2: biased action distributions")
    parser.add_argument("--include-lm-sweep", action="store_true",
                        help="add Axis 3: landmark-density sweep (lm200 only)")
    parser.add_argument("--lm-densities", type=int, nargs="+",
                        default=[0, 50, 100, 200],
                        help="n_landmarks values to test (only used with --include-lm-sweep)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="ZERO_SHOT_TRANSFER.md")
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    train_n_lm = 200 if args.config == "lm200" else 0
    test_seeds = list(range(args.test_seed_base,
                            args.test_seed_base + args.n_test_seeds))

    out_lines = []
    out_lines.append(f"# Zero-Shot Structural-Transfer Evaluation\n")
    out_lines.append(f"Generated: {datetime.datetime.now()}\n")
    out_lines.append(f"Config: **{args.config}**, training n_landmarks: {train_n_lm}")
    out_lines.append(f"Model seeds: {args.model_seeds}, fresh test seeds per model: "
                     f"{args.n_test_seeds} (base {args.test_seed_base})")
    out_lines.append(f"Variants: {', '.join(args.variants)}")
    out_lines.append(f"Lengths: {args.lengths}\n")

    # ----------------------------------------------------------------
    # Axis 1 — fresh obs_map seeds
    # ----------------------------------------------------------------
    out_lines.append("## Axis 1 — Fresh obs_map seeds (primary zero-shot test)\n")
    out_lines.append("Each cell: mean ± std over (model seeds × fresh test seeds) "
                     f"= {len(args.model_seeds) * args.n_test_seeds} runs per length.\n")

    header_acc = ["Variant"] + [f"T={T} acc" for T in args.lengths]
    header_nll = ["Variant"] + [f"T={T} NLL" for T in args.lengths]
    rows_acc, rows_nll = [], []

    for variant in args.variants:
        accs_per_T = {T: [] for T in args.lengths}
        nlls_per_T = {T: [] for T in args.lengths}

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
                for T in args.lengths:
                    n = args.n_trials_long if T >= 2048 else args.n_trials_128
                    np.random.seed(ts * 7919 + T)
                    torch.manual_seed(ts * 7919 + T)
                    a, nll = eval_revisit(model, env, T, n, device=args.device)
                    if a is not None:
                        accs_per_T[T].append(a)
                        nlls_per_T[T].append(nll)
                    print(f"  {variant} ms={ms} ts={ts} T={T}: "
                          f"acc={a if a is None else f'{a:.3f}'}",
                          file=sys.stderr, flush=True)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

        rows_acc.append([variant] + [_fmt(accs_per_T[T]) for T in args.lengths])
        rows_nll.append([variant] + [_fmt(nlls_per_T[T]) for T in args.lengths])

    out_lines.append("### Accuracy\n")
    out_lines.append(_table(rows_acc, header_acc))
    out_lines.append("\n### NLL\n")
    out_lines.append(_table(rows_nll, header_nll))
    out_lines.append("")

    # ----------------------------------------------------------------
    # Axis 2 — biased action distributions
    # ----------------------------------------------------------------
    if args.include_bias:
        out_lines.append("\n## Axis 2 — Biased action distributions (T=512)\n")
        out_lines.append("Models trained on uniform actions; tested on biased ones. "
                         "One fresh obs_map seed per cell, averaged across model seeds.\n")
        bias_names = list(ACTION_BIASES.keys())
        header_b = ["Variant"] + [f"{n} acc" for n in bias_names]
        rows_b = []
        T_bias = 512

        for variant in args.variants:
            per_bias = {n: [] for n in bias_names}
            for ms in args.model_seeds:
                ckpt = runs / f"{variant}_{args.config}" / f"seed{ms}" / f"{variant}.pt"
                if not ckpt.exists():
                    continue
                try:
                    model, cfg = build_model(variant, ckpt, args.device)
                except Exception as e:
                    print(f"[skip] {variant} seed{ms}: {e}", file=sys.stderr)
                    continue
                for bn in bias_names:
                    env = GridWorld(
                        size=cfg.get("grid_size", 64),
                        n_obs_types=cfg.get("n_obs_types", 16),
                        p_empty=cfg.get("p_empty", 0.5),
                        n_landmarks=cfg.get("n_landmarks", train_n_lm),
                        seed=args.test_seed_base + 100,
                    )
                    np.random.seed(args.test_seed_base + 100 + ms)
                    torch.manual_seed(args.test_seed_base + 100 + ms)
                    a, _ = eval_revisit(model, env, T_bias, args.n_trials_128,
                                        action_probs=ACTION_BIASES[bn],
                                        device=args.device)
                    if a is not None:
                        per_bias[bn].append(a)
                    print(f"  bias {variant} ms={ms} {bn}: "
                          f"acc={a if a is None else f'{a:.3f}'}",
                          file=sys.stderr, flush=True)
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            rows_b.append([variant] + [_fmt(per_bias[n]) for n in bias_names])

        out_lines.append(_table(rows_b, header_b))
        out_lines.append("")

    # ----------------------------------------------------------------
    # Axis 3 — landmark density sweep (lm200 only)
    # ----------------------------------------------------------------
    if args.include_lm_sweep:
        if args.config != "lm200":
            out_lines.append("\n*Axis 3 (landmark density) requires --config lm200; skipping.*")
        else:
            out_lines.append("\n## Axis 3 — Landmark density transfer (T=512)\n")
            out_lines.append(f"Models trained with n_landmarks=200; tested with subsets "
                             f"{args.lm_densities}. Tests whether the per-token R_t head "
                             "generalizes its informativeness ranking to fewer/more landmarks.\n")
            header_d = ["Variant"] + [f"n_lm={d} acc" for d in args.lm_densities]
            rows_d = []
            T_lm = 512

            for variant in args.variants:
                per_d = {d: [] for d in args.lm_densities}
                for ms in args.model_seeds:
                    ckpt = runs / f"{variant}_{args.config}" / f"seed{ms}" / f"{variant}.pt"
                    if not ckpt.exists():
                        continue
                    try:
                        model, cfg = build_model(variant, ckpt, args.device)
                    except Exception as e:
                        print(f"[skip] {variant} seed{ms}: {e}", file=sys.stderr)
                        continue
                    train_lm = cfg.get("n_landmarks", train_n_lm)
                    for d in args.lm_densities:
                        if d > train_lm:
                            continue  # would need vocab IDs the model never saw
                        env = GridWorld(
                            size=cfg.get("grid_size", 64),
                            n_obs_types=cfg.get("n_obs_types", 16),
                            p_empty=cfg.get("p_empty", 0.5),
                            n_landmarks=d,
                            seed=args.test_seed_base + 200,
                        )
                        np.random.seed(args.test_seed_base + 200 + ms)
                        torch.manual_seed(args.test_seed_base + 200 + ms)
                        a, _ = eval_revisit(model, env, T_lm, args.n_trials_128,
                                            device=args.device)
                        if a is not None:
                            per_d[d].append(a)
                        print(f"  lm-sweep {variant} ms={ms} n_lm={d}: "
                              f"acc={a if a is None else f'{a:.3f}'}",
                              file=sys.stderr, flush=True)
                    del model
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                rows_d.append([variant] + [_fmt(per_d[d]) for d in args.lm_densities])

            out_lines.append(_table(rows_d, header_d))
            out_lines.append("")

    out_lines.append("\n---")
    out_lines.append("*Auto-generated by `zero_shot_eval.py`.*")

    text = "\n".join(out_lines) + "\n"
    Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
