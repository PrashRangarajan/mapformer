#!/usr/bin/env python3
"""Test-time ω rescaling — bounded zero-shot grid-size generalization.

Idea: A model trained on grid size N has ω frequencies covering
[2π/N, 2π]. To evaluate on grid size N', the natural rescaling is

    ω_test = ω_train · (N / N')

so that the smallest frequency becomes 2π/N' (the new Nyquist cutoff).
This is the YaRN/NTK-aware analogue for MapFormer's path integration.

Three test scales per model: 0.5×, 1×, 2× the training grid. For each,
we report:
  - revisit accuracy with ORIGINAL ω (untouched checkpoint)
  - revisit accuracy with RESCALED ω (multiplied by N_train/N_test)

If ω rescaling helps, rescaled > original at non-1× scales, and the
gap narrows as test_size approaches train_size.

Usage:
  python3 -m mapformer.rescale_eval \
      --runs-dir runs --config clean \
      --variants Vanilla Level15 Level15EM \
      --model-seeds 0 1 2 \
      --test-sizes 32 48 64 96 128 \
      --output OMEGA_RESCALE.md
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
    return m.to(device).eval(), cfg


def get_omega_param(model):
    """Return the (path_integrator) ω Parameter, robust across variants."""
    if hasattr(model, "path_integrator"):
        return model.path_integrator.omega
    raise RuntimeError(f"Model has no path_integrator.omega: {type(model).__name__}")


def eval_revisit(model, env, T: int, n_trials: int, device: str = "cuda"):
    correct = total = 0
    nll_sum = 0.0
    revisits = 0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).to(device)
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None, None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]
            tgts = tt[0, 1:]
            mask = rm[1:].to(device)
            n_rev = mask.sum().item()
            revisits += n_rev
            if n_rev == 0:
                continue
            correct += (preds[mask] == tgts[mask]).sum().item()
            total += n_rev
            idx = torch.arange(lp.shape[1], device=device)[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    if total == 0:
        return None, None, revisits / max(n_trials, 1)
    return correct / total, nll_sum / total, revisits / n_trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="clean")
    p.add_argument("--variants", nargs="+",
                   default=["Vanilla", "VanillaEM", "Level15", "Level15EM"])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-test-seeds", type=int, default=3)
    p.add_argument("--test-seed-base", type=int, default=20000)
    p.add_argument("--test-sizes", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="OMEGA_RESCALE.md")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    train_n_lm = 200 if args.config == "lm200" else 0
    test_seeds = list(range(args.test_seed_base,
                            args.test_seed_base + args.n_test_seeds))

    # variant -> test_size -> mode -> list of accuracies
    results = {v: {ts: {"orig": [], "rescaled": []} for ts in args.test_sizes}
               for v in args.variants}
    nrev    = {v: {ts: {"orig": [], "rescaled": []} for ts in args.test_sizes}
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
            train_size = cfg.get("grid_size", 64)
            try:
                omega = get_omega_param(model)
            except RuntimeError as e:
                print(f"[skip] {variant} {e}", file=sys.stderr)
                continue
            omega_orig = omega.data.clone()

            for test_size in args.test_sizes:
                scale_factor = train_size / test_size
                for mode in ("orig", "rescaled"):
                    if mode == "rescaled":
                        omega.data.copy_(omega_orig * scale_factor)
                    else:
                        omega.data.copy_(omega_orig)

                    for ts in test_seeds:
                        env = GridWorld(
                            size=test_size,
                            n_obs_types=cfg.get("n_obs_types", 16),
                            p_empty=cfg.get("p_empty", 0.5),
                            n_landmarks=cfg.get("n_landmarks", train_n_lm),
                            seed=ts,
                        )
                        np.random.seed(ts * 7919 + test_size)
                        torch.manual_seed(ts * 7919 + test_size)
                        a, _, nr = eval_revisit(model, env, args.T,
                                                args.n_trials, args.device)
                        if a is not None:
                            results[variant][test_size][mode].append(a)
                            nrev[variant][test_size][mode].append(nr)

            # restore original ω so next experiment is clean
            omega.data.copy_(omega_orig)
            print(f"  done {variant} ms={ms}", file=sys.stderr)

    # Markdown
    out = [
        f"# Test-Time ω Rescaling — Grid-Size Generalization\n",
        f"Generated: {datetime.datetime.now()}\n",
        f"Config: **{args.config}**, training grid size = 64, T = {args.T}",
        f"Variants: {', '.join(args.variants)}",
        f"Model seeds: {args.model_seeds}, fresh test seeds: {args.n_test_seeds} each",
        f"Test grid sizes: {args.test_sizes}\n",
        f"Each cell: revisit accuracy mean ± std over "
        f"{len(args.model_seeds) * args.n_test_seeds} runs.\n",
        f"`orig`     = ω left untouched (the trained checkpoint's frequencies)",
        f"`rescaled` = ω multiplied by (train_size / test_size) at eval time\n",
    ]

    def fmt(vals):
        if not vals: return "—"
        if len(vals) == 1: return f"{vals[0]:.3f}"
        return f"{st.mean(vals):.3f}±{st.pstdev(vals):.3f}"

    for variant in args.variants:
        out.append(f"## {variant}\n")
        header = ["Test grid size"] + [str(ts) for ts in args.test_sizes]
        sep = "|" + "|".join(["---"] * len(header)) + "|"
        out.append("| " + " | ".join(header) + " |")
        out.append(sep)
        for mode in ("orig", "rescaled"):
            row = [f"acc ({mode})"] + [
                fmt(results[variant][ts][mode]) for ts in args.test_sizes
            ]
            out.append("| " + " | ".join(row) + " |")
        # avg revisits per trajectory (helps interpret accuracy)
        row = ["mean #revisits/traj"] + [
            f"{st.mean(nrev[variant][ts]['orig']):.1f}" if nrev[variant][ts]["orig"] else "—"
            for ts in args.test_sizes
        ]
        out.append("| " + " | ".join(row) + " |")
        out.append("")

    out.append("\n---\n*Auto-generated by `rescale_eval.py`.*\n")
    Path(args.output).write_text("\n".join(out))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
