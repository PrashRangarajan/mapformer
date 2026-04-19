#!/usr/bin/env python3
"""
Gaussian Δ noise test — classical Kalman territory.

Unlike action replacement (discrete noise), this injects continuous Gaussian
perturbations directly into the Δ values INSIDE the model at eval time.

This models proprioceptive noise: the agent's internal estimate of its
action magnitude is noisy even when the intended action is clean.
Classical IMU/Kalman scenario.

Expected:
- Vanilla (clean train): big degradation (path integration drifts)
- Vanilla+noise (discrete noise train): some robustness (data augmentation)
- Kalman+noise: may show advantage if uncertainty mechanism activates
"""

import argparse
import sys
import torch
import torch.nn as nn
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


class GaussianDeltaWrapper(nn.Module):
    """Wraps a MapFormer to inject Gaussian noise into the Δ values.

    Monkey-patches the action_to_lie forward to add noise.
    """

    def __init__(self, model, noise_std: float = 0.0):
        super().__init__()
        self.model = model
        self.noise_std = noise_std
        # Save original forward
        self._orig_forward = model.action_to_lie.forward

        def noisy_forward(x):
            delta = self._orig_forward(x)
            if self.noise_std > 0:
                delta = delta + torch.randn_like(delta) * self.noise_std
            return delta

        model.action_to_lie.forward = noisy_forward

    def forward(self, tokens):
        return self.model(tokens)


def eval_with_delta_noise(model, env, n_steps, noise_std, n_trials, device, seed=0):
    """Evaluate accuracy on revisits with Gaussian Δ noise."""
    wrapper = GaussianDeltaWrapper(model, noise_std=noise_std)
    wrapper.eval()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)  # for reproducible noise

    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)
            tokens_t = tokens.unsqueeze(0).to(device)
            revisit_mask_t = revisit_mask.unsqueeze(0).to(device)

            logits = wrapper(tokens_t[:, :-1])
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


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    name = Path(checkpoint_path).stem
    if "Level2InEKF" in name:
        cls = MapFormerWM_Level2InEKF
    elif "PredictiveCoding" in name:
        cls = MapFormerWM_PredictiveCoding
    elif "ParallelInEKF" in name:
        cls = MapFormerWM_ParallelInEKF
    elif "ProperInEKF" in name:
        cls = MapFormerWM_ProperInEKF
    elif "InEKF" in name:
        cls = MapFormerWM_InEKF
    elif "WM" in name:
        cls = MapFormerWM
    else:
        cls = MapFormerEM
    model = build_model_from_config(config, cls)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device), config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="List of checkpoint paths to compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-trials", type=int, default=200)
    args = parser.parse_args()

    # Assume all models trained on the same env config
    _, first_config = load_model(args.checkpoints[0], args.device)
    env = GridWorld(
        size=first_config["grid_size"],
        n_obs_types=first_config["n_obs_types"],
        p_empty=first_config["p_empty"],
        seed=42,
    )

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]

    # Load each model once
    models = {}
    for ckpt in args.checkpoints:
        name = Path(ckpt).stem
        model, _ = load_model(ckpt, args.device)
        models[name] = model

    # Header
    print(f"T={args.n_steps}, trials={args.n_trials}")
    print()
    header = f"{'noise_std':>10}"
    for name in models:
        header += f"  {name:>30}"
    print(header)
    print("-" * len(header))

    for noise in noise_levels:
        row = f"{noise:>10.2f}"
        for name, model in models.items():
            acc = eval_with_delta_noise(
                model, env, args.n_steps, noise, args.n_trials, args.device, seed=0
            )
            row += f"  {acc:>30.3f}"
        print(row)


if __name__ == "__main__":
    main()
