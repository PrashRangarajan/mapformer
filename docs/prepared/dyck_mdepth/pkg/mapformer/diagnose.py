#!/usr/bin/env python3
"""
Diagnostic: load a trained MapFormer checkpoint and analyze it comprehensively.

Tests:
1. Disentanglement — do actions produce large ||Δ|| and observations ~0?
2. Prediction distribution — is the model just predicting blank?
3. Per-class accuracy — how well does it predict non-blank obs?
4. Omega — what did the learned angular velocities converge to?
5. Position consistency — do same (x,y) locations produce similar position states?
6. Path integration quality — does position drift over time?
7. Attention pattern — does attention at a revisit attend to the previous visit?
8. Repeat-visit accuracy — accuracy specifically at revisited locations
9. Loss decomposition — is loss dominated by blank predictions?
10. Action selectivity — do individual action tokens produce different ω·Δ?
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM


# ============================================================
# 1. Disentanglement: ||Δ|| per token type
# ============================================================

def analyze_delta_by_token_type(model, env, device):
    model.eval()
    all_tokens = torch.arange(env.unified_vocab_size).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.token_emb(all_tokens)
        delta = model.action_to_lie(x)
    return delta.norm(dim=(-2, -1))[0].cpu().numpy()


def print_disentanglement(model, env, device):
    print("=" * 60)
    print("1. DISENTANGLEMENT — ||Δ|| per token type")
    print("=" * 60)
    delta_norms = analyze_delta_by_token_type(model, env, device)
    action_mean = delta_norms[:env.N_ACTIONS].mean()
    obs_mean = delta_norms[env.N_ACTIONS:].mean()
    print(f"{'Token':>7} {'Type':>8} {'||Δ||':>10}")
    for i in range(env.unified_vocab_size):
        label = "ACTION" if i < env.N_ACTIONS else ("BLANK" if i == env.unified_blank else "OBS")
        print(f"{i:>7} {label:>8} {delta_norms[i]:>10.4f}")
    print()
    print(f"Mean action ||Δ||:      {action_mean:.4f}")
    print(f"Mean observation ||Δ||: {obs_mean:.4f}")
    ratio = action_mean / max(obs_mean, 1e-8)
    print(f"Action/Obs ratio:       {ratio:.2f}x")
    if ratio > 5:
        print("✓ STRONG DISENTANGLEMENT LEARNED")
    elif ratio > 2:
        print("~ PARTIAL DISENTANGLEMENT")
    else:
        print("✗ NO DISENTANGLEMENT — model treats actions like observations")


# ============================================================
# 2 & 3. Prediction distribution + per-class accuracy
# ============================================================

def analyze_predictions(model, env, device, n_trials=100, n_steps=128):
    model.eval()
    all_preds = []
    all_targets = []
    pred_correct_blank = 0
    pred_correct_nonblank = 0
    n_blank = 0
    n_nonblank = 0

    with torch.no_grad():
        for _ in range(n_trials):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)
            tokens_t = tokens.unsqueeze(0).to(device)
            obs_mask_t = obs_mask.unsqueeze(0).to(device)

            logits = model(tokens_t[:, :-1])
            preds = logits.argmax(-1)
            targets = tokens_t[:, 1:]
            mask = obs_mask_t[:, 1:]

            p = preds[mask].cpu().numpy()
            t = targets[mask].cpu().numpy()

            all_preds.extend(p.tolist())
            all_targets.extend(t.tolist())

            is_blank_t = t == env.unified_blank
            correct = p == t

            pred_correct_blank += np.logical_and(is_blank_t, correct).sum()
            pred_correct_nonblank += np.logical_and(~is_blank_t, correct).sum()
            n_blank += is_blank_t.sum()
            n_nonblank += (~is_blank_t).sum()

    return {
        "pred_hist": np.bincount(np.array(all_preds), minlength=env.unified_vocab_size),
        "target_hist": np.bincount(np.array(all_targets), minlength=env.unified_vocab_size),
        "blank_acc": pred_correct_blank / max(n_blank, 1),
        "nonblank_acc": pred_correct_nonblank / max(n_nonblank, 1),
        "n_blank": int(n_blank),
        "n_nonblank": int(n_nonblank),
    }


def print_predictions(stats, env):
    print("\n" + "=" * 60)
    print("2. PREDICTION DISTRIBUTION")
    print("=" * 60)
    overall = (stats["blank_acc"] * stats["n_blank"] +
               stats["nonblank_acc"] * stats["n_nonblank"]) / (stats["n_blank"] + stats["n_nonblank"])
    print(f"Overall accuracy:      {overall:.3f}")
    print(f"Blank-target accuracy: {stats['blank_acc']:.3f} ({stats['n_blank']} targets)")
    print(f"Non-blank target acc:  {stats['nonblank_acc']:.3f} ({stats['n_nonblank']} targets)")
    print(f"Chance (non-blank):    {1/env.n_obs_types:.3f} (1/K)")

    total_p = stats['pred_hist'].sum()
    total_t = stats['target_hist'].sum()
    print(f"\n{'Token':>7} {'Type':>8} {'Pred %':>8} {'Target %':>10}")
    for i in range(env.unified_vocab_size):
        label = "ACTION" if i < env.N_ACTIONS else ("BLANK" if i == env.unified_blank else "OBS")
        pp = 100 * stats['pred_hist'][i] / max(total_p, 1)
        tp = 100 * stats['target_hist'][i] / max(total_t, 1)
        print(f"{i:>7} {label:>8} {pp:>7.1f}% {tp:>9.1f}%")

    # Is model collapsed to blank?
    pred_blank_pct = 100 * stats['pred_hist'][env.unified_blank] / max(total_p, 1)
    if pred_blank_pct > 90:
        print("\n✗ MODEL COLLAPSED: predicts blank >90% of the time")
    elif pred_blank_pct > 70:
        print("\n~ MODEL BIASED toward blank prediction")


# ============================================================
# 4. Omega analysis
# ============================================================

def print_omega(model):
    print("\n" + "=" * 60)
    print("4. OMEGA (learned angular velocities)")
    print("=" * 60)
    omega = model.path_integrator.omega.detach().cpu().numpy()
    print(f"Shape: {omega.shape}")
    print(f"Range: [{omega.min():.4f}, {omega.max():.4f}]  Mean: {omega.mean():.4f}  Std: {omega.std():.4f}")
    for h in range(omega.shape[0]):
        print(f"Head {h}: min={omega[h].min():.4f}  max={omega[h].max():.4f}  "
              f"first 8: {omega[h, :8]}")


# ============================================================
# 5. Position consistency: same (x,y) → similar position states?
# ============================================================

def analyze_position_consistency(model, env, device, n_samples=200, n_steps=64):
    """Gather (x,y) locations and model's position state (cos, sin).
    Compute: for pairs visiting the same (x,y), how similar are their states?
    Compare to pairs at different (x,y) locations.
    """
    model.eval()
    states_per_loc = {}

    with torch.no_grad():
        for _ in range(n_samples):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)
            tokens_t = tokens.unsqueeze(0).to(device)
            cos_a, sin_a = model.get_position_state(tokens_t)
            # cos_a: (1, H, 2*n_steps, n_blocks) — we want states at obs positions
            obs_positions = torch.arange(1, 2 * n_steps, 2)
            cos_at_obs = cos_a[0, :, obs_positions, :]  # (H, n_steps, n_blocks)
            sin_at_obs = sin_a[0, :, obs_positions, :]
            state_per_t = torch.cat([
                cos_at_obs.permute(1, 0, 2).reshape(n_steps, -1),
                sin_at_obs.permute(1, 0, 2).reshape(n_steps, -1),
            ], dim=-1).cpu().numpy()  # (n_steps, feat_dim)

            for t, (x, y) in enumerate(env.visited_locations):
                key = (x, y)
                if key not in states_per_loc:
                    states_per_loc[key] = []
                states_per_loc[key].append(state_per_t[t])

    # Compute intra-location and inter-location cosine similarity
    same_loc_sims = []
    diff_loc_sims = []

    loc_keys = list(states_per_loc.keys())

    # Intra: pairs at same location
    for key, states in states_per_loc.items():
        if len(states) < 2:
            continue
        S = np.array(states)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        sims = S @ S.T
        mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
        same_loc_sims.extend(sims[mask].tolist())

    # Inter: random pairs across locations
    rng = np.random.RandomState(0)
    for _ in range(min(5000, len(loc_keys) * 50)):
        k1, k2 = rng.choice(len(loc_keys), 2, replace=False)
        s1 = states_per_loc[loc_keys[k1]][0]
        s2 = states_per_loc[loc_keys[k2]][0]
        n1 = s1 / (np.linalg.norm(s1) + 1e-8)
        n2 = s2 / (np.linalg.norm(s2) + 1e-8)
        diff_loc_sims.append(float(n1 @ n2))

    return {
        "same_mean": float(np.mean(same_loc_sims)) if same_loc_sims else 0.0,
        "same_std": float(np.std(same_loc_sims)) if same_loc_sims else 0.0,
        "diff_mean": float(np.mean(diff_loc_sims)) if diff_loc_sims else 0.0,
        "diff_std": float(np.std(diff_loc_sims)) if diff_loc_sims else 0.0,
        "n_same": len(same_loc_sims),
        "n_diff": len(diff_loc_sims),
    }


def print_position_consistency(stats):
    print("\n" + "=" * 60)
    print("5. POSITION CONSISTENCY — same (x,y) → similar state?")
    print("=" * 60)
    print(f"Same-location cosine sim:  {stats['same_mean']:.4f} ± {stats['same_std']:.4f}  (n={stats['n_same']})")
    print(f"Diff-location cosine sim:  {stats['diff_mean']:.4f} ± {stats['diff_std']:.4f}  (n={stats['n_diff']})")
    gap = stats['same_mean'] - stats['diff_mean']
    print(f"Gap (same - diff):         {gap:.4f}")
    if gap > 0.3:
        print("✓ POSITION STATES CLUSTER BY LOCATION")
    elif gap > 0.1:
        print("~ WEAK POSITION CLUSTERING")
    else:
        print("✗ POSITION STATES DO NOT ENCODE LOCATION")


# ============================================================
# 6. Revisit accuracy: does model do better at revisited locations?
# ============================================================

def analyze_revisit_accuracy(model, env, device, n_trials=100, n_steps=128):
    """Compute accuracy SEPARATELY for first visits vs revisits.
    A model that learned path integration should do much better at revisits.
    """
    model.eval()
    first_visit_correct = 0
    first_visit_total = 0
    revisit_correct = 0
    revisit_total = 0

    with torch.no_grad():
        for _ in range(n_trials):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)
            tokens_t = tokens.unsqueeze(0).to(device)
            locations = env.visited_locations  # list of (x, y) for each step

            logits = model(tokens_t[:, :-1])
            preds = logits.argmax(-1)[0].cpu().numpy()  # (2*n_steps - 1,)
            targets = tokens_t[0, 1:].cpu().numpy()

            visited = set()
            # Observation prediction positions in the interleaved seq:
            # target at position 1, 3, 5, ... (odd indices)
            # Each obs position corresponds to step t (0-indexed)
            for t in range(n_steps):
                target_idx = 2 * t + 1 - 1  # target array index (= 2t, since target = tokens[1:])
                # Wait: tokens[1:] at idx i corresponds to tokens at original idx i+1
                # Obs at original position 2t+1. In tokens[1:], index = 2t.
                if target_idx >= len(preds):
                    break
                loc = locations[t]
                is_correct = preds[target_idx] == targets[target_idx]
                if loc in visited:
                    revisit_correct += int(is_correct)
                    revisit_total += 1
                else:
                    first_visit_correct += int(is_correct)
                    first_visit_total += 1
                visited.add(loc)

    return {
        "first_visit_acc": first_visit_correct / max(first_visit_total, 1),
        "revisit_acc": revisit_correct / max(revisit_total, 1),
        "n_first": first_visit_total,
        "n_revisit": revisit_total,
    }


def print_revisit_accuracy(stats):
    print("\n" + "=" * 60)
    print("6. REVISIT ACCURACY — key test from paper")
    print("=" * 60)
    print(f"First-visit accuracy: {stats['first_visit_acc']:.3f} ({stats['n_first']} cases)")
    print(f"Revisit accuracy:     {stats['revisit_acc']:.3f} ({stats['n_revisit']} cases)")
    lift = stats['revisit_acc'] - stats['first_visit_acc']
    print(f"Revisit lift:         {lift:+.3f}")
    if lift > 0.2:
        print("✓ MODEL USES PATH INTEGRATION (remembers revisits)")
    elif lift > 0.05:
        print("~ WEAK PATH INTEGRATION")
    else:
        print("✗ NO PATH INTEGRATION — no benefit at revisited locations")


# ============================================================
# 7. Action selectivity: each action type → different ω·Δ?
# ============================================================

def print_action_selectivity(model, env, device):
    """For each action token, check the actual θ = ω·Δ it produces."""
    print("\n" + "=" * 60)
    print("7. ACTION SELECTIVITY — do 4 actions produce different rotations?")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        action_tokens = torch.tensor([[0, 1, 2, 3]]).to(device)  # N, S, W, E
        x = model.token_emb(action_tokens)
        delta = model.action_to_lie(x)  # (1, 4, H, n_blocks)
        omega = model.path_integrator.omega  # (H, n_blocks)
        theta = delta * omega.unsqueeze(0).unsqueeze(0)  # (1, 4, H, n_blocks)
    theta_np = theta[0].cpu().numpy()  # (4, H, n_blocks)
    names = ["N", "S", "W", "E"]
    print(f"{'Action':>6} {'||θ||':>10} {'θ[h=0,b=0]':>12} {'θ[h=0,b=1]':>12}")
    for i, n in enumerate(names):
        norm = np.linalg.norm(theta_np[i])
        print(f"{n:>6} {norm:>10.4f} {theta_np[i, 0, 0]:>12.4f} {theta_np[i, 0, 1]:>12.4f}")

    # Pairwise differences (opposite actions N/S and W/E should produce opposite rotations)
    print(f"\nθ(N) + θ(S) norm:  {np.linalg.norm(theta_np[0] + theta_np[1]):.4f}  (should be ~0 if learned)")
    print(f"θ(W) + θ(E) norm:  {np.linalg.norm(theta_np[2] + theta_np[3]):.4f}  (should be ~0 if learned)")
    print(f"θ(N) - θ(S) norm:  {np.linalg.norm(theta_np[0] - theta_np[1]):.4f}")
    print(f"θ(W) - θ(E) norm:  {np.linalg.norm(theta_np[2] - theta_np[3]):.4f}")


# ============================================================
# Main
# ============================================================

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
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]

    name = Path(args.checkpoint).stem
    if "WM" in name:
        model_class = MapFormerWM
    elif "EM" in name:
        model_class = MapFormerEM
    else:
        raise ValueError(f"Cannot infer model type from {name}")

    model = build_model_from_config(config, model_class)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    env = GridWorld(
        size=config["grid_size"],
        n_obs_types=config["n_obs_types"],
        p_empty=config["p_empty"],
        seed=42,
    )

    print(f"Loaded {name} from {args.checkpoint}")
    print(f"Final training loss: {ckpt['losses'][-1]:.4f}")
    print(f"Vocab: {env.unified_vocab_size}")
    print(f"  Actions: 0..3, Obs: 4..{env.obs_offset + env.n_obs_types - 1}, Blank: {env.unified_blank}")
    print()

    print_disentanglement(model, env, args.device)

    pred_stats = analyze_predictions(model, env, args.device)
    print_predictions(pred_stats, env)

    print_omega(model)

    print_action_selectivity(model, env, args.device)

    pos_stats = analyze_position_consistency(model, env, args.device)
    print_position_consistency(pos_stats)

    rev_stats = analyze_revisit_accuracy(model, env, args.device)
    print_revisit_accuracy(rev_stats)

    print("\nDone.")


if __name__ == "__main__":
    main()
