"""Per-seed accuracies for the H=12 allocentric budget curve.

The aggregate curve (H12_BUDGET_CURVE.md) is non-monotone: the position effect
goes +0.264 -> +0.383 -> +0.286 as the batch budget goes 980 -> 2000 -> 4000,
and the seed spread REOPENS at 4000 (+/-0.005 -> +/-0.060). Training loss shows
the same shape: nb=2000 lands all three seeds at 0.51-0.55 while nb=4000 gives
0.42 / 0.82 / 0.83 -- one seed better than any other run, two seeds worse.

That pattern is bimodal-basin, not undertraining, so this script prints the
per-seed numbers the aggregate hides. Standing rule 6: three seeds is not a
point estimate, and the variance here IS the finding.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from mapformer.eval_h12_budget import KW, SOURCES, score
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=128)
    args = ap.parse_args()
    dev = torch.device(args.device)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0,
                    seed=10000, **KW)
    for label, rel in SOURCES:
        base = _REPO / rel
        if not base.exists():
            continue
        for var in ("Vanilla", "RoPE"):
            for s in (0, 1, 2):
                ck = base / f"{var}_s{s}" / f"{var}.pt"
                if not ck.exists():
                    continue
                d = torch.load(ck, map_location="cpu", weights_only=False)
                c = d["config"]
                m = VARIANT_MAP[var](vocab_size=c["vocab_size"],
                                     d_model=c["d_model"], n_heads=c["n_heads"],
                                     n_layers=c["n_layers"],
                                     grid_size=c["grid_size"]).to(dev)
                m.load_state_dict(d["model_state_dict"])
                acc, floor = score(m, env, args.n_batches, args.batch_size,
                                   args.n_steps, dev, seed=5000 + s)
                del m
                torch.cuda.empty_cache()
                ls = d.get("losses") or [float("nan")]
                print(f"nb={label:>5s} {var:8s} s{s}  acc={acc:.4f}  "
                      f"final_loss={ls[-1]:.4f}  floor={floor:.4f}", flush=True)


if __name__ == "__main__":
    main()
