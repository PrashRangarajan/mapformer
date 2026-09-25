"""
Lie group utilities for MapFormer.

Implements SO(2) and SO(n) via block-diagonal 2x2 rotations.
- Skew-symmetric matrix construction
- Exponential map (Lie algebra -> Lie group)
- Logarithmic map (Lie group -> Lie algebra)
- Block-diagonal rotation matrix construction
"""

import torch
import torch.nn as nn
import math


def skew_symmetric_2d(theta: torch.Tensor) -> torch.Tensor:
    """Construct 2x2 skew-symmetric matrices from angles.

    Args:
        theta: (...,) tensor of angles

    Returns:
        (..., 2, 2) skew-symmetric matrices A where A = [[0, -theta], [theta, 0]]
    """
    shape = theta.shape
    zeros = torch.zeros_like(theta)
    # Stack into [[0, -theta], [theta, 0]]
    A = torch.stack([
        torch.stack([zeros, -theta], dim=-1),
        torch.stack([theta, zeros], dim=-1)
    ], dim=-2)
    return A


def exp_map_2d(theta: torch.Tensor) -> torch.Tensor:
    """Closed-form exponential map for SO(2).

    Args:
        theta: (...,) tensor of angles

    Returns:
        (..., 2, 2) rotation matrices R = [[cos, -sin], [sin, cos]]
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1)
    ], dim=-2)
    return R


def log_map_2d(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map for SO(2): rotation matrix -> angle.

    Args:
        R: (..., 2, 2) rotation matrices

    Returns:
        (...,) angles theta
    """
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def build_block_diagonal_rotations(delta: torch.Tensor) -> torch.Tensor:
    """Build block-diagonal rotation matrices from Lie algebra parameters.

    Each pair of dimensions gets an independent 2x2 rotation block.
    If delta has odd dimension, the last dimension gets a trivial (identity) block.

    Args:
        delta: (batch, seq_len, n_rot_dims) Lie algebra parameters

    Returns:
        (batch, seq_len, d, d) block-diagonal rotation matrices
        where d = 2 * n_rot_dims
    """
    B, T, n_rot = delta.shape
    d = 2 * n_rot

    # Build each 2x2 rotation block
    R_blocks = exp_map_2d(delta)  # (B, T, n_rot, 2, 2)

    # Assemble into block-diagonal matrix
    M = torch.zeros(B, T, d, d, device=delta.device, dtype=delta.dtype)
    for i in range(n_rot):
        M[:, :, 2*i:2*i+2, 2*i:2*i+2] = R_blocks[:, :, i]

    return M


def build_block_diagonal_rotations_fast(delta: torch.Tensor) -> torch.Tensor:
    """Vectorized version of build_block_diagonal_rotations (no Python loop).

    Args:
        delta: (batch, seq_len, n_rot_dims) Lie algebra parameters

    Returns:
        (batch, seq_len, d, d) block-diagonal rotation matrices
    """
    B, T, n_rot = delta.shape
    d = 2 * n_rot

    cos_t = torch.cos(delta)  # (B, T, n_rot)
    sin_t = torch.sin(delta)

    M = torch.zeros(B, T, d, d, device=delta.device, dtype=delta.dtype)

    idx = torch.arange(n_rot, device=delta.device)
    # Diagonal entries (cos)
    M[:, :, 2*idx, 2*idx] = cos_t
    M[:, :, 2*idx+1, 2*idx+1] = cos_t
    # Off-diagonal entries (sin)
    M[:, :, 2*idx, 2*idx+1] = -sin_t
    M[:, :, 2*idx+1, 2*idx] = sin_t

    return M


def exp_map_so_n(A: torch.Tensor) -> torch.Tensor:
    """General exponential map for SO(n) via torch.matrix_exp.

    Args:
        A: (..., n, n) skew-symmetric matrices

    Returns:
        (..., n, n) rotation matrices
    """
    return torch.matrix_exp(A)


def log_map_so_n(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map for SO(n) via Rodrigues-like formula.

    Uses the relation: log(R) = (theta / 2*sin(theta)) * (R - R^T)
    where theta = arccos((tr(R) - 1) / 2).

    Args:
        R: (..., n, n) rotation matrices

    Returns:
        (..., n, n) skew-symmetric matrices in the Lie algebra
    """
    n = R.shape[-1]
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)

    # Handle small angles with Taylor expansion
    small = theta.abs() < 1e-6
    scale = torch.where(
        small,
        0.5 + theta**2 / 12,  # Taylor expansion of theta/(2*sin(theta))
        theta / (2 * torch.sin(theta.clamp(min=1e-10)))
    )

    # Expand scale for matrix multiplication
    for _ in range(len(R.shape) - len(scale.shape)):
        scale = scale.unsqueeze(-1)

    return scale * (R - R.transpose(-1, -2))


def is_orthogonal(R: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check if matrices are orthogonal: R^T @ R = I."""
    n = R.shape[-1]
    eye = torch.eye(n, device=R.device, dtype=R.dtype)
    return torch.allclose(R.transpose(-1, -2) @ R, eye.expand_as(R), atol=atol)


def is_special_orthogonal(R: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check if matrices are in SO(n): orthogonal with det = +1."""
    if not is_orthogonal(R, atol):
        return False
    dets = torch.det(R)
    return torch.allclose(dets, torch.ones_like(dets), atol=atol)
