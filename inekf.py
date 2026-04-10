"""
Invariant Extended Kalman Filter (InEKF) layer for MapFormer.

Adds uncertainty tracking to the position state. Maintains:
  - mu_t in SO(n): mean position estimate (same as MapFormer's P_t)
  - Sigma_t in R^{d x d}: covariance in the Lie algebra

Key property (invariance): error dynamics are state-independent because
the InEKF exploits Lie group symmetry. This means the covariance
prediction step has constant coefficients and can be parallelised.

Reference: Barrau & Bonnabel (2017), "The Invariant Extended Kalman
Filter as a Stable Observer."
"""

import torch
import torch.nn as nn
import math

from .lie_groups import log_map_2d, exp_map_2d


class InEKFLayer(nn.Module):
    """Invariant EKF that runs alongside MapFormer's position state.

    Predict step: propagates covariance alongside the prefix scan.
    Update step: corrects position when a landmark is detected.
    """

    def __init__(self, lie_algebra_dim: int, obs_dim: int):
        """
        Args:
            lie_algebra_dim: dimension of Lie algebra (n_rot_dims)
            obs_dim: dimension of landmark observation signal
        """
        super().__init__()
        self.lie_dim = lie_algebra_dim
        self.obs_dim = obs_dim

        # Learned process noise (diagonal, in log space for positivity)
        self.log_Q_diag = nn.Parameter(torch.zeros(lie_algebra_dim))

        # Learned measurement noise (diagonal, in log space)
        self.log_R_diag = nn.Parameter(torch.zeros(obs_dim))

        # Observation model: maps Lie algebra position to expected landmark observation
        self.H = nn.Linear(lie_algebra_dim, obs_dim, bias=False)

        # Landmark detector: maps content embedding to landmark probability + observation
        self.landmark_detector = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # landmark probability (logit)
        )
        self.landmark_obs_proj = nn.Linear(obs_dim, obs_dim)

    @property
    def Q(self) -> torch.Tensor:
        """Process noise covariance (diagonal)."""
        return torch.diag(self.log_Q_diag.exp())

    @property
    def R(self) -> torch.Tensor:
        """Measurement noise covariance (diagonal)."""
        return torch.diag(self.log_R_diag.exp())

    def predict(
        self, Sigma: torch.Tensor, F: torch.Tensor
    ) -> torch.Tensor:
        """Predict step: propagate covariance forward.

        Sigma_t = F @ Sigma_{t-1} @ F^T + Q

        Due to InEKF invariance, F is constant (state-independent).

        Args:
            Sigma: (batch, d, d) current covariance
            F: (d, d) state transition Jacobian (constant)

        Returns:
            (batch, d, d) predicted covariance
        """
        Q = self.Q.to(Sigma.device)
        return F @ Sigma @ F.T + Q.unsqueeze(0)

    def update(
        self,
        mu_lie: torch.Tensor,
        Sigma: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update step: correct position estimate using landmark observation.

        Innovation: y = z - H @ mu_lie
        Kalman gain: K = Sigma @ H^T @ (H @ Sigma @ H^T + R)^{-1}
        Correction: mu_lie_new = mu_lie + K @ y
        Covariance: Sigma_new = (I - K @ H) @ Sigma

        Args:
            mu_lie: (batch, d) mean position in Lie algebra
            Sigma: (batch, d, d) covariance
            z: (batch, obs_dim) landmark observation

        Returns:
            mu_lie_corrected: (batch, d) corrected mean
            Sigma_corrected: (batch, d, d) reduced covariance
        """
        H_w = self.H.weight  # (obs_dim, lie_dim)
        R = self.R.to(Sigma.device)

        # Innovation
        predicted_obs = self.H(mu_lie)  # (batch, obs_dim)
        y = z - predicted_obs

        # Kalman gain
        S = H_w @ Sigma @ H_w.T + R.unsqueeze(0)  # (batch, obs_dim, obs_dim)
        # K = Sigma @ H^T @ S^{-1}
        K = Sigma @ H_w.T @ torch.linalg.solve(S, torch.eye(
            self.obs_dim, device=S.device
        ).unsqueeze(0).expand_as(S))

        # Correction
        mu_corrected = mu_lie + (K @ y.unsqueeze(-1)).squeeze(-1)

        # Joseph form for numerical stability
        I = torch.eye(self.lie_dim, device=Sigma.device).unsqueeze(0)
        IKH = I - K @ H_w.unsqueeze(0)
        Sigma_corrected = IKH @ Sigma @ IKH.transpose(-1, -2) + \
            K @ R.unsqueeze(0) @ K.transpose(-1, -2)

        return mu_corrected, Sigma_corrected

    def forward(
        self,
        positions: torch.Tensor,
        content_emb: torch.Tensor,
        F: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run InEKF over a sequence.

        Predict at every step; update only when landmark detected.

        Args:
            positions: (batch, seq_len, d_head, d_head) position matrices from prefix scan
            content_emb: (batch, seq_len, obs_dim) content embeddings
            F: (lie_dim, lie_dim) state transition Jacobian

        Returns:
            corrected_positions: (batch, seq_len, d_head, d_head)
            covariances: (batch, seq_len, lie_dim, lie_dim)
            landmark_probs: (batch, seq_len) probability of landmark at each step
        """
        B, T, d, _ = positions.shape
        device = positions.device

        # Initialize covariance
        Sigma = torch.eye(self.lie_dim, device=device).unsqueeze(0).expand(B, -1, -1) * 0.01

        covariances = torch.zeros(B, T, self.lie_dim, self.lie_dim, device=device)
        corrected = positions.clone()
        landmark_logits = self.landmark_detector(content_emb).squeeze(-1)  # (B, T)
        landmark_probs = torch.sigmoid(landmark_logits)

        for t in range(T):
            # Predict: propagate covariance
            Sigma = self.predict(Sigma, F)

            # Check for landmarks (soft gating during training)
            prob = landmark_probs[:, t]  # (B,)
            is_landmark = prob > 0.5  # hard decision for position correction

            if is_landmark.any():
                # Extract Lie algebra coordinates from position matrix
                # For block-diagonal SO(2) rotations, extract angles
                mu_lie = torch.zeros(B, self.lie_dim, device=device)
                for i in range(self.lie_dim):
                    block = corrected[:, t, 2*i:2*i+2, 2*i:2*i+2]
                    mu_lie[:, i] = log_map_2d(block)

                z = self.landmark_obs_proj(content_emb[:, t])

                mu_corrected, Sigma_new = self.update(mu_lie, Sigma, z)

                # Soft blend based on landmark probability
                alpha = prob.unsqueeze(-1)  # (B, 1)

                # Reconstruct corrected position matrix
                for i in range(self.lie_dim):
                    new_block = exp_map_2d(mu_corrected[:, i])
                    old_block = corrected[:, t, 2*i:2*i+2, 2*i:2*i+2]
                    blended = alpha.unsqueeze(-1) * new_block + \
                        (1 - alpha.unsqueeze(-1)) * old_block
                    corrected[:, t, 2*i:2*i+2, 2*i:2*i+2] = blended

                Sigma = alpha.unsqueeze(-1) * Sigma_new + \
                    (1 - alpha.unsqueeze(-1)) * Sigma

            covariances[:, t] = Sigma

        return corrected, covariances, landmark_probs
