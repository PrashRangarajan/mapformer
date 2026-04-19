#!/usr/bin/env python3
"""
Test robustness of a trained MapFormer to action noise.

If the model is brittle to noise, InEKF (explicit uncertainty tracking +
landmark-based correction) should help. If it's already robust, InEKF
provides little benefit.

At eval time, with probability p_noise each action token is replaced with
a random action BEFORE being fed to the model — simulating sensor noise
or action execution failure. The OBSERVATIONS still reflect the agent's
true position.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_kalman import MapFormerWM_InEKF
from mapformer.model_inekf_proper import MapFormerWM_ProperInEKF
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF


def noisy_trajectory(env, n_steps, p_noise, rng):
    """Generate a trajectory, then corrupt a fraction of actions.

    The environment tracks the true position (agent moved as intended).
    Only the ACTION TOKENS fed to the model are corrupted.
    """
    tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)

    if p_noise > 0:
        # Corrupt actions at even positions (0, 2, 4, ...)
        for i in range(0, len(tokens), 2):
            if rng.random() < p_noise:
                tokens[i] = rng.randint(0, env.N_ACTIONS)  # random action

    return tokens, obs_mask, revisit_mask


def eval_with_noise(model, env, n_steps, p_noise, n_trials, device, seed=0):
    """Evaluate accuracy on revisits under action noise."""
    model.eval()
    rng = np.random.RandomState(seed)

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_trials):
            tokens, obs_mask, revisit_mask = noisy_trajectory(
                env, n_steps, p_noise, rng
            )
            tokens_t = tokens.unsqueeze(0).to(device)
            revisit_mask_t = revisit_mask.unsqueeze(0).to(device)

            logits = model(tokens_t[:, :-1])
            preds = logits.argmax(-1)
            targets = tokens_t[:, 1:]
            mask = revisit_mask_t[:, 1:]

            if mask.sum() == 0:
                continue
            correct += (preds[mask] == targets[mask]).sum().item()
            total += mask.sum().item()

    return correct / max(total, 1)


def build_model_from_config(config, model_class):
    return model_class(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        grid_size=config["grid_size"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-trials", type=int, default=200)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]

    name = Path(args.checkpoint).stem
    if "Level2InEKF" in name:
        model_class = MapFormerWM_Level2InEKF
    elif "PredictiveCoding" in name:
        model_class = MapFormerWM_PredictiveCoding
    elif "ParallelInEKF" in name:
        model_class = MapFormerWM_ParallelInEKF
    elif "ProperInEKF" in name:
        model_class = MapFormerWM_ProperInEKF
    elif "InEKF" in name:
        model_class = MapFormerWM_InEKF
    elif "WM" in name:
        model_class = MapFormerWM
    else:
        model_class = MapFormerEM
    model = build_model_from_config(config, model_class)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    env = GridWorld(
        size=config["grid_size"],
        n_obs_types=config["n_obs_types"],
        p_empty=config["p_empty"],
        seed=42,
    )

    print(f"Model: {name}")
    print(f"Task: T={args.n_steps}, grid={config['grid_size']}, trials={args.n_trials}")
    print()
    print(f"{'Action noise':>13} {'Accuracy':>10} {'Degradation':>12}")
    print("-" * 40)

    # Baseline (no noise)
    acc0 = eval_with_noise(model, env, args.n_steps, 0.0, args.n_trials, args.device, seed=0)
    print(f"{0.0:>12.2f}  {acc0:>9.3f}  {'baseline':>12}")

    for p in [0.05, 0.10, 0.20, 0.30, 0.50]:
        acc = eval_with_noise(model, env, args.n_steps, p, args.n_trials, args.device, seed=0)
        deg = acc0 - acc
        print(f"{p:>12.2f}  {acc:>9.3f}  {deg:>+11.3f}")

    print()
    print("If accuracy drops significantly with noise, the model is brittle")
    print("and explicit uncertainty tracking (InEKF) should help.")


if __name__ == "__main__":
    main()
