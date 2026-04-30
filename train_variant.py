#!/usr/bin/env python3
"""Unified training script — trains any variant (main models + ablations + baselines).

Usage:
  python3 -m mapformer.train_variant \
      --variant Level15 --seed 0 --n-landmarks 0 --p-action-noise 0.0 \
      --output-dir runs/level15_clean_s0

Supported variants:
  Vanilla              — plain MapFormer-WM (no correction)
  VanillaEM            — MapFormer-EM (Hadamard product attention)
  Level1               — parallel InEKF (constant K* from DARE)
  Level15              — constant learnable Π, per-token R_t (on MapFormer-WM)
  Level15EM            — Level 1.5 InEKF on MapFormer-EM backbone
  Level2               — full heteroscedastic (Möbius scan), slow
  PC                   — predictive coding
  Grid                 — multi-orientation path integration (hex-capable)
  Grid_Free            — Grid with learnable orientation angles
  GridL15PC            — Grid + Level 1.5 InEKF + PC aux loss (combined)
  GridL15PC_Free       — GridL15PC with learnable orientations
  L15_ConstR, L15_NoMeas, L15_NoCorr, L15_DARE — Level 1.5 ablations
  RoPE                 — standard transformer with fixed RoPE (baseline)
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.train import train
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF, MapFormerEM_Level15InEKF_b5
from mapformer.model_grid import MapFormerWM_Grid, MapFormerWM_Grid_Free
from mapformer.model_grid_l15_pc import (
    MapFormerWM_GridL15PC,
    MapFormerWM_GridL15PC_Free,
)
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass
from mapformer.model_level15_pc_v3 import MapFormerWM_Level15PC_v3
from mapformer.model_level15_pc_v4 import MapFormerWM_Level15PC_v4
from mapformer.model_level15_dog import MapFormerWM_Level15_DoG
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_ablations import ABLATIONS
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_baselines_extra import EXTRA_BASELINES


VARIANT_MAP = {
    "Vanilla":    MapFormerWM,
    "VanillaEM":  MapFormerEM,
    "Level1":     MapFormerWM_ParallelInEKF,
    "Level15":    MapFormerWM_Level15InEKF,
    "Level15EM":  MapFormerEM_Level15InEKF,
    "Level15EM_b5": MapFormerEM_Level15InEKF_b5,
    "Grid":       MapFormerWM_Grid,
    "Grid_Free":  MapFormerWM_Grid_Free,
    "GridL15PC":  MapFormerWM_GridL15PC,
    "GridL15PC_Free": MapFormerWM_GridL15PC_Free,
    "Level15PC":  MapFormerWM_Level15PC,
    "Level15PC_NoBypass": MapFormerWM_Level15PC_NoBypass,
    "Level15PC_v3": MapFormerWM_Level15PC_v3,
    "Level15PC_v4": MapFormerWM_Level15PC_v4,
    "Level15_DoG": MapFormerWM_Level15_DoG,
    "Level2":     MapFormerWM_Level2InEKF,
    "PC":         MapFormerWM_PredictiveCoding,
    "RoPE":       MapFormerWM_RoPE,
    **ABLATIONS,
    **EXTRA_BASELINES,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-landmarks", type=int, default=0)
    parser.add_argument("--p-action-noise", type=float, default=0.0)
    parser.add_argument("--aux-coef", type=float, default=0.0,
                        help="Coefficient for auxiliary prediction-error loss "
                             "(used by PC and GridL15PC variants).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-batches", type=int, default=156)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128,
                        help="Override d_model. For Grid hexagonal config "
                             "(11 modules x 3 orientations) need d_model=132.")
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = GridWorld(
        size=64, n_obs_types=16, p_empty=0.5,
        n_landmarks=args.n_landmarks, seed=args.seed,
    )
    cls = VARIANT_MAP[args.variant]
    model = cls(
        vocab_size=env.unified_vocab_size,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, grid_size=64,
    )

    print(f"{args.variant} seed={args.seed} n_landmarks={args.n_landmarks} "
          f"p_noise={args.p_action_noise}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    losses = train(
        model, env,
        n_epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, n_steps=args.n_steps,
        n_batches=args.n_batches, device=args.device,
        p_action_noise=args.p_action_noise,
        aux_coef=args.aux_coef,
    )

    ckpt_path = out / f"{args.variant}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "variant": args.variant,
        "seed": args.seed,
        "config": {
            "vocab_size": env.unified_vocab_size,
            "d_model": args.d_model, "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "grid_size": 64, "n_obs_types": 16, "p_empty": 0.5,
            "n_landmarks": args.n_landmarks,
        },
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
