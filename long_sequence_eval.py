#!/usr/bin/env python3
"""Long-sequence evaluation — pushes T far beyond training length.

The bounded-error property of Kalman filtering predicts variants with
state correction should hold accuracy at T >> train_len while attention-
only models degrade. This script evaluates each variant at many lengths.

Usage:
  python3 -m mapformer.long_sequence_eval \
      --runs-dir runs/ --config clean \
      --lengths 128 256 512 1024 2048 4096 \
      --output LONG_SEQ_RESULTS.md
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_baselines_extra import EXTRA_BASELINES
from mapformer.model_ablations import ABLATIONS

VARIANT_CLS = {
    "Vanilla": MapFormerWM, "VanillaEM": MapFormerEM,
    "Level1": MapFormerWM_ParallelInEKF,
    "Level15": MapFormerWM_Level15InEKF,
    "Level15EM": MapFormerEM_Level15InEKF,
    "Level15PC":   MapFormerWM_Level15PC,
    "Level2": MapFormerWM_Level2InEKF,
    "PC": MapFormerWM_PredictiveCoding, "RoPE": MapFormerWM_RoPE,
    **ABLATIONS,
    **EXTRA_BASELINES,
}


def build_model(variant, ckpt_path, vocab_size, grid_size=64):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=grid_size)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, om, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            try:
                logits = model(tt[:, :-1])
            except Exception as e:
                # OOM or similar at very long T
                return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            c += (preds[mask] == tgts[mask]).sum().item()
            tot += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    return (c / tot if tot else None, nll_sum / tot if tot else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config", default="clean", help="clean|noise|lm200")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--variants", nargs="+",
                        default=["Vanilla", "VanillaEM", "RoPE", "Level1",
                                 "Level15", "Level15EM", "PC", "LSTM",
                                 "MambaLike", "Level15PC"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-trials-128", type=int, default=200)
    parser.add_argument("--n-trials-long", type=int, default=50,
                        help="trials at T>=2048 (slower)")
    parser.add_argument("--output", default="LONG_SEQ_RESULTS.md")
    args = parser.parse_args()

    n_lm = 200 if args.config == "lm200" else 0
    runs = Path(args.runs_dir)

    print(f"# Long-Sequence Evaluation\n")
    print(f"Generated: {__import__('datetime').datetime.now()}\n")
    print(f"Config: {args.config}, n_landmarks: {n_lm}, lengths: {args.lengths}")
    print(f"Seeds: {args.seeds}\n")

    # Per-variant table: T across columns, mean ± std across seeds
    header = "| Variant | " + " | ".join(f"T={T}" for T in args.lengths) + " |"
    sep = "|---------|" + "|".join("-------" for _ in args.lengths) + "|"

    print("## Accuracy\n")
    print(header); print(sep)
    for variant in args.variants:
        row = f"| {variant}"
        for T in args.lengths:
            vals = []
            for seed in args.seeds:
                ckpt = runs / f"{variant}_{args.config}" / f"seed{seed}" / f"{variant}.pt"
                if not ckpt.exists(): continue
                env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                n_landmarks=n_lm, seed=seed)
                m = build_model(variant, ckpt, env.unified_vocab_size)
                n_trials = args.n_trials_long if T >= 2048 else args.n_trials_128
                a, nll = eval_revisit(m, env, T, n_trials, seed=seed + 1000)
                del m; torch.cuda.empty_cache()
                if a is not None:
                    vals.append(a)
                sys.stdout.flush()
            if vals:
                cell = f"{st.mean(vals):.3f}±{st.pstdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"
            else:
                cell = "N/A"
            row += f" | {cell}"
        row += " |"
        print(row)

    print("\n## NLL\n")
    print(header); print(sep)
    for variant in args.variants:
        row = f"| {variant}"
        for T in args.lengths:
            vals = []
            for seed in args.seeds:
                ckpt = runs / f"{variant}_{args.config}" / f"seed{seed}" / f"{variant}.pt"
                if not ckpt.exists(): continue
                env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                n_landmarks=n_lm, seed=seed)
                m = build_model(variant, ckpt, env.unified_vocab_size)
                n_trials = args.n_trials_long if T >= 2048 else args.n_trials_128
                a, nll = eval_revisit(m, env, T, n_trials, seed=seed + 1000)
                del m; torch.cuda.empty_cache()
                if nll is not None:
                    vals.append(nll)
            if vals:
                cell = f"{st.mean(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"
            else:
                cell = "N/A"
            row += f" | {cell}"
        row += " |"
        print(row)


if __name__ == "__main__":
    main()
