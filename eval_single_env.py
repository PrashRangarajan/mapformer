"""Single-env evaluation pass for checkpoints that don't have stored metrics.

Used to fill the single-env clean row in TEM_BACKGROUND_BASELINES.md (the
TEM checkpoints were trained via train_variant.py which doesn't save
train/test_acc). Generic — works with any variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


def build(variant, ckpt_path, device="cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c["config"]; cls = VARIANT_MAP[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg.get("grid_size", 64))
    if "n_modes" in cfg: kw["n_modes"] = cfg["n_modes"]
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def evaluate(model, env, T, n_trials, device, seed=2000):
    # GridWorld.generate_trajectory uses np.random directly (no rng arg),
    # so seed numpy once for reproducibility.
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll = 0.0
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
        if mask.sum() == 0: continue
        correct += (preds[mask] == tgts[mask]).sum().item()
        total += mask.sum().item()
        idx = torch.arange(lp.shape[1], device=device)[mask]
        nll += -lp[0, idx, tgts[mask]].sum().item()
    return (correct / total if total else None,
            nll / total if total else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--variant", required=True, type=str)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-trials-train", type=int, default=100)
    ap.add_argument("--n-trials-test-T1", type=int, default=100)
    ap.add_argument("--n-trials-test-T2", type=int, default=50)
    ap.add_argument("--train-T", type=int, default=128)
    ap.add_argument("--eval-T2", type=int, default=512)
    ap.add_argument("--train-env-seed", type=int, default=0,
                    help="env seed used during training (default 0)")
    ap.add_argument("--test-env-seed", type=int, default=2000,
                    help="env seed for held-out trajectories.")
    args = ap.parse_args()

    model, cfg = build(args.variant, Path(args.checkpoint), device=args.device)

    # Train env (same seed as training)
    env_train = GridWorld(
        size=cfg.get("grid_size", 64),
        n_obs_types=cfg.get("n_obs_types", 16),
        p_empty=cfg.get("p_empty", 0.5),
        n_landmarks=cfg.get("n_landmarks", 0),
        seed=args.train_env_seed,
    )
    # Note: single-env regime — we evaluate on the SAME env, same obs_map,
    # but different random trajectories (held-out via different rng seed).
    acc_train, nll_train = evaluate(model, env_train, args.train_T,
                                     args.n_trials_train, args.device,
                                     seed=args.test_env_seed)
    acc_t128, nll_t128 = evaluate(model, env_train, args.train_T,
                                   args.n_trials_test_T1, args.device,
                                   seed=args.test_env_seed + 1)
    acc_t512, nll_t512 = evaluate(model, env_train, args.eval_T2,
                                   args.n_trials_test_T2, args.device,
                                   seed=args.test_env_seed + 2)

    out = {
        "variant": args.variant, "ckpt": str(args.checkpoint),
        "config": cfg,
        "acc_train": acc_train, "nll_train": nll_train,
        "acc_T128": acc_t128, "nll_T128": nll_t128,
        "acc_T512": acc_t512, "nll_T512": nll_t512,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
