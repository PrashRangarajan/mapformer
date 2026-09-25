"""
Evaluation and interpretability tools for MapFormer.

Implements the key analyses from Rambaud et al. (2025):
1. Length generalisation test (train on T, test on 4T, 8T, 16T)
2. Position state PCA (verify cognitive map geometry)
3. Grid cell autocorrelation (check for hexagonal structure)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

from .environment import GridWorld


def eval_accuracy(
    model: nn.Module,
    env: GridWorld,
    n_steps: int,
    n_trials: int = 200,
    device: str = "cpu",
) -> float:
    """Evaluate next-observation prediction accuracy on interleaved tokens.

    Only measures accuracy on observation predictions (after action tokens).

    Args:
        model: trained model
        env: GridWorld
        n_steps: number of (action, observation) steps
        n_trials: number of test trajectories
        device: device

    Returns:
        Accuracy (fraction of observations correctly predicted)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_trials):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(n_steps)
            tokens = tokens.unsqueeze(0).to(device)
            revisit_mask = revisit_mask.unsqueeze(0).to(device)

            logits = model(tokens[:, :-1])
            preds = logits.argmax(-1)
            targets = tokens[:, 1:]
            # Paper: accuracy at revisited locations only
            mask = revisit_mask[:, 1:]

            if mask.sum() == 0:
                continue
            correct += (preds[mask] == targets[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def eval_length_generalisation(
    model: nn.Module,
    env: GridWorld,
    train_len: int = 64,
    test_lens: Optional[list[int]] = None,
    n_trials: int = 200,
    device: str = "cpu",
) -> dict[int, float]:
    """Length generalisation test: train on T, test on multiples of T.

    This is the headline experiment from Rambaud et al. (2025).

    Args:
        model: trained model
        env: GridWorld
        train_len: training sequence length
        test_lens: list of test sequence lengths
        n_trials: trials per length
        device: device

    Returns:
        Dict mapping sequence length -> accuracy
    """
    if test_lens is None:
        test_lens = [train_len, 2 * train_len, 4 * train_len,
                     8 * train_len, 16 * train_len]

    results = {}
    for n_steps in test_lens:
        acc = eval_accuracy(model, env, n_steps, n_trials, device)
        results[n_steps] = acc

    return results


def extract_position_states(
    model: nn.Module,
    env: GridWorld,
    n_samples: int = 500,
    traj_len: int = 32,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract position states and true locations for PCA analysis.

    Args:
        model: trained MapFormer (must have get_position_state method)
        env: GridWorld
        n_samples: number of trajectory endpoints to sample
        traj_len: length of each trajectory
        device: device

    Returns:
        positions: (n_samples, 2) true (x, y) locations
        states: (n_samples, d) flattened position state vectors
    """
    model.eval()
    positions = []
    states = []

    with torch.no_grad():
        for _ in range(n_samples):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(traj_len)
            tokens_t = tokens.unsqueeze(0).to(device)

            cos_a, sin_a = model.get_position_state(tokens_t)
            # Use last position's cos/sin as state (last token = last obs)
            state = torch.cat([cos_a[0, :, -1, :].flatten(),
                               sin_a[0, :, -1, :].flatten()]).cpu().numpy()

            positions.append((env.last_x, env.last_y))
            states.append(state)

    return np.array(positions), np.array(states)


def compute_rate_map(
    model: nn.Module,
    env: GridWorld,
    cell_idx: int = 0,
    n_trajectories: int = 1000,
    traj_len: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Compute firing rate map for a single position state dimension.

    Averages the value of position_state[cell_idx] at each grid location
    across many trajectories.

    Args:
        model: trained MapFormer
        env: GridWorld
        cell_idx: which dimension of the flattened position state to use
        n_trajectories: number of trajectories to average over
        traj_len: trajectory length
        device: device

    Returns:
        (grid_size, grid_size) rate map
    """
    model.eval()
    rate_map = np.zeros((env.size, env.size))
    counts = np.zeros((env.size, env.size))

    with torch.no_grad():
        for _ in range(n_trajectories):
            tokens, obs_mask, revisit_mask = env.generate_trajectory(traj_len)
            tokens_t = tokens.unsqueeze(0).to(device)

            cos_a, sin_a = model.get_position_state(tokens_t)
            # cos_a: (1, H, 2*traj_len, n_blocks) — interleaved tokens
            # Extract state at observation positions (odd indices)
            n_tokens = 2 * traj_len
            obs_indices = torch.arange(1, n_tokens, 2)  # odd positions
            cos_obs = cos_a[0, :, obs_indices, :]  # (H, traj_len, n_blocks)
            sin_obs = sin_a[0, :, obs_indices, :]
            P_flat = torch.cat([
                cos_obs.permute(1, 0, 2).reshape(traj_len, -1),
                sin_obs.permute(1, 0, 2).reshape(traj_len, -1),
            ], dim=-1).cpu().numpy()

            for t, (x, y) in enumerate(env.visited_locations):
                if cell_idx < P_flat.shape[1]:
                    rate_map[x, y] += P_flat[t, cell_idx]
                    counts[x, y] += 1

    rate_map = np.where(counts > 0, rate_map / counts, 0)
    return rate_map


# ============================================================
# Plotting functions for figure replication
# ============================================================

def plot_length_generalisation(
    results: dict[str, dict[int, float]],
    save_path: Optional[str] = None,
):
    """Plot length generalisation comparison across models.

    Replicates Figure 3 / Table 1 from Rambaud et al. (2025).

    Args:
        results: {model_name: {seq_len: accuracy}}
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for i, (name, res) in enumerate(results.items()):
        lens = sorted(res.keys())
        accs = [res[l] for l in lens]
        ax.plot(lens, accs, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], label=name,
                linewidth=2, markersize=8)

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Next-Obs Prediction Accuracy", fontsize=12)
    ax.set_title("Length Generalisation", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_position_pca(
    positions: np.ndarray,
    states: np.ndarray,
    save_path: Optional[str] = None,
):
    """PCA of position states, coloured by true grid location.

    Replicates Figure 4 from Rambaud et al. (2025).

    Args:
        positions: (N, 2) true (x, y) locations
        states: (N, d) flattened position state vectors
        save_path: if provided, save figure
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    projected = pca.fit_transform(states)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Colour by x position
    sc1 = ax1.scatter(projected[:, 0], projected[:, 1],
                      c=positions[:, 0], cmap='viridis', s=15, alpha=0.7)
    plt.colorbar(sc1, ax=ax1, label='True x position')
    ax1.set_title("Position State PCA (coloured by x)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")

    # Colour by y position
    sc2 = ax2.scatter(projected[:, 0], projected[:, 1],
                      c=positions[:, 1], cmap='plasma', s=15, alpha=0.7)
    plt.colorbar(sc2, ax=ax2, label='True y position')
    ax2.set_title("Position State PCA (coloured by y)")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_grid_cell_autocorrelation(
    rate_map: np.ndarray,
    cell_idx: int = 0,
    save_path: Optional[str] = None,
):
    """Plot rate map and its 2D autocorrelation.

    Replicates Figure 5 from Rambaud et al. (2025).
    A genuine grid cell should show 6 peaks at 60 degree intervals
    in the autocorrelation.

    Args:
        rate_map: (grid_size, grid_size) firing rate map
        cell_idx: which cell this is (for title)
        save_path: if provided, save figure
    """
    from scipy.signal import correlate2d

    autocorr = correlate2d(rate_map, rate_map, mode='full')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(rate_map, cmap='hot', interpolation='nearest')
    ax1.set_title(f'Rate Map (cell {cell_idx})')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')

    ax2.imshow(autocorr, cmap='RdBu_r', interpolation='nearest')
    ax2.set_title('Autocorrelation (6 peaks at 60deg = grid cell)')
    ax2.set_xlabel('lag y')
    ax2.set_ylabel('lag x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_curves(
    all_losses: dict[str, list[float]],
    save_path: Optional[str] = None,
):
    """Plot training loss curves for all models.

    Args:
        all_losses: {model_name: [epoch_losses]}
        save_path: if provided, save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for i, (name, losses) in enumerate(all_losses.items()):
        ax.plot(losses, label=name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_uncertainty_over_time(
    covariances: torch.Tensor,
    save_path: Optional[str] = None,
):
    """Plot InEKF covariance trace over sequence length.

    Shows uncertainty growth during path integration and
    reduction at landmark corrections.

    Args:
        covariances: (seq_len, d, d) covariance matrices
        save_path: if provided, save figure
    """
    traces = torch.diagonal(covariances, dim1=-2, dim2=-1).sum(-1).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(traces, linewidth=2, color='#2196F3')
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Trace(Sigma)", fontsize=12)
    ax.set_title("Position Uncertainty Over Time (InEKF)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
