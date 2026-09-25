"""
Parallel prefix scan for cumulative matrix products.

Implements O(log T) parallel prefix product for sequences of matrices,
used for path integration: P_t = M_t @ M_{t-1} @ ... @ M_1.

Also includes a sequential reference implementation for validation.
"""

import torch


def parallel_prefix_product(matrices: torch.Tensor) -> torch.Tensor:
    """Parallel prefix product via Blelloch-style up-sweep.

    Computes cumulative products: output[b, t] = M_t @ M_{t-1} @ ... @ M_1

    Note on convention: MapFormer uses left-to-right accumulation where
    new actions multiply on the left: P_t = M_t @ P_{t-1}.

    Args:
        matrices: (batch, seq_len, n, n) sequence of matrices

    Returns:
        (batch, seq_len, n, n) cumulative products
    """
    B, T, n, _ = matrices.shape
    result = matrices.clone()

    # Up-sweep: at each level, combine elements that are 'step' apart
    step = 1
    while step < T:
        # result[:, t] = result[:, t] @ result[:, t - step] for t >= step
        left = result[:, step:].reshape(-1, n, n)
        right = result[:, :-step].reshape(-1, n, n)
        result[:, step:] = torch.bmm(left, right).reshape(B, T - step, n, n)
        step *= 2

    return result


def sequential_prefix_product(matrices: torch.Tensor) -> torch.Tensor:
    """Sequential cumulative product (reference implementation).

    Args:
        matrices: (batch, seq_len, n, n)

    Returns:
        (batch, seq_len, n, n) cumulative products
    """
    B, T, n, _ = matrices.shape
    result = torch.zeros_like(matrices)
    result[:, 0] = matrices[:, 0]

    for t in range(1, T):
        result[:, t] = matrices[:, t] @ result[:, t - 1]

    return result


def parallel_prefix_product_with_covariance(
    matrices: torch.Tensor,
    F: torch.Tensor,
    Q: torch.Tensor,
    Sigma_0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel prefix product for both position and covariance.

    The covariance recurrence Sigma_t = F @ Sigma_{t-1} @ F^T + Q
    can be parallelised when F is constant (InEKF invariance property).

    For constant F, we reformulate as an associative scan over augmented
    state (M_cumul, Sigma_cumul).

    Args:
        matrices: (batch, seq_len, n, n) action rotation matrices
        F: (n_lie, n_lie) state transition Jacobian (constant due to invariance)
        Q: (n_lie, n_lie) process noise covariance
        Sigma_0: (batch, n_lie, n_lie) initial covariance

    Returns:
        positions: (batch, seq_len, n, n) cumulative position matrices
        covariances: (batch, seq_len, n_lie, n_lie) covariance at each step
    """
    B, T, n, _ = matrices.shape
    n_lie = F.shape[0]

    # Position prefix product (same as before)
    positions = parallel_prefix_product(matrices)

    # Covariance propagation: Sigma_t = F @ Sigma_{t-1} @ F^T + Q
    # With constant F, this is a linear recurrence that can be computed sequentially
    # (full parallel scan for Lyapunov recurrence is complex; sequential is fine
    # since n_lie is small)
    covariances = torch.zeros(B, T, n_lie, n_lie, device=matrices.device)
    Sigma = Sigma_0.clone()
    FT = F.T

    for t in range(T):
        Sigma = F @ Sigma @ FT + Q
        covariances[:, t] = Sigma

    return positions, covariances
