"""MapFormer-Grid + Level 1.5 InEKF + PC auxiliary loss.

Three independent extensions to MapFormer-WM stacked into one architecture,
each addressing a different falsified bio-plausibility prediction in the
original Level 1.5 paper:

  - Grid path integration (model_grid.py): replaces the single-block-per-ω
    structure with modules of blocks at {0°, 60°, 120°} orientations,
    making hexagonal interference patterns architecturally accessible.
    Targets the falsified hexagonal-grid-cell prediction (paper Section 6.10).

  - Level 1.5 InEKF correction (model_inekf_level15.py): adds the missing
    update step to MapFormer's path-integration-only state. Per-token R_t
    head modulates a per-token Kalman gain K_t = Π/(Π + R_t) with a
    constant learnable Π. Wrapped innovation gives bounded-error length
    generalisation. Targets the missing-update-step problem.

  - PC auxiliary loss (model_predictive_coding.py): a forward model
    g(cos θ̂, sin θ̂) → predicted observation embedding, with auxiliary
    loss ‖x - g‖² at observation positions. Acts as a representation
    regulariser pushing the corrected angle θ̂ to be position-
    discriminative. Targets the falsified R_t Bayesian-informativeness
    ordering, by giving R_t an explicit predictability signal to learn from.

Note that PC contributes ONLY the auxiliary loss here, not its δθ
correction. State correction is fully handled by Level 1.5; PC's role is
purely as a representation regulariser via the aux loss.

Default config requires ``d_model=132`` (n_modules=11, n_orientations=3,
n_heads=2 ⇒ n_blocks=33 ⇒ d_head=66 ⇒ d_model=132). Override at construction
or pass --d-model 132 when training.

Forward pass (parallelism preserved end-to-end):

    token → x = embed(token)                                   [parallel]
    delta_2d = ActionToLie2D(x)                  # (B,L,H,M,2)  [parallel]
    Δ_block = cos(θ_o) Δ_x + sin(θ_o) Δ_y        # (B,L,H,M,O)  [parallel]
    cum_delta = cumsum(Δ_block, dim=L)                          [O(log L)]
    θ_path = ω · cum_delta → reshape → (B,L,H,NB)               [parallel]
    θ̂ = θ_path + InEKFLevel15.scan(...)                         [O(log L)]
    pos_feat = [cos θ̂, sin θ̂]                                  [parallel]
    ô = forward_model(pos_feat)                                 [parallel]
    err = (x - ô) · obs_mask     # accumulate for aux_loss      [parallel]
    RoPE(θ̂) on Q, K → standard causal attention                 [standard]

Use with::

    python -m mapformer.train_variant --variant GridL15PC \\
        --d-model 132 --aux-coef 0.1 --epochs 50 ...

The aux_coef hyperparameter (default 0.1) controls the strength of the PC
auxiliary loss relative to next-token loss. Inherited from the original PC
implementation; tune if needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import WMTransformerLayer
from .model_grid import MapFormerWM_Grid
from .model_inekf_level15 import InEKFLevel15


class MapFormerWM_GridL15PC(MapFormerWM_Grid):
    """Combined Grid + Level 1.5 InEKF + PC-aux-loss MapFormer-WM.

    Inherits Grid path integration from ``MapFormerWM_Grid`` (which itself
    inherits the embedding, transformer layers, and output projection from
    ``MapFormerWM``). Adds the Level 1.5 InEKF correction module and the
    PC forward model.

    The model exposes ``prediction_error_loss()`` returning the masked
    ‖x - ô‖² at observation positions, to be summed with the next-token
    loss in the training step::

        total_loss = next_token_loss + aux_coef * model.prediction_error_loss()

    Args:
        vocab_size: Token vocabulary size from the GridWorld environment.
        d_model: Hidden width. Must satisfy ``d_model // n_heads // 2 ==
            n_modules * n_orientations``. Default 132 with the default
            (11, 3) gives ``n_blocks = 33``.
        n_heads: Attention heads. Default 2.
        n_layers: Transformer layers. Default 1 (paper-faithful).
        dropout: Dropout rate. Default 0.1.
        grid_size: Environment side length, used for ω init range. Default 64.
        bottleneck_r: Low-rank bottleneck for ActionToLie2D. Default 2.
        n_modules: Number of distinct ω frequencies (modules). Default 11.
        n_orientations: Per-module orientation count. Default 3 (hex-optimal).
        learnable_orientations: If True, orientations are learnable
            parameters initialised at hex angles. Default False (fixed at
            {0°, 60°, 120°}) — tests the structural hypothesis directly.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 132,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        grid_size: int = 64,
        bottleneck_r: int = 2,
        n_modules: int = 11,
        n_orientations: int = 3,
        learnable_orientations: bool = False,
    ):
        super().__init__(
            vocab_size, d_model, n_heads, n_layers, dropout,
            grid_size, bottleneck_r,
            n_modules=n_modules, n_orientations=n_orientations,
            learnable_orientations=learnable_orientations,
        )
        # Use plain WMTransformerLayer (Grid parent already does this via
        # inherited MapFormerWM, but be explicit for clarity).
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Level 1.5 InEKF correction module. Operates on (B, L, H, NB)-
        # shaped angles where NB = n_modules * n_orientations.
        self.inekf = InEKFLevel15(d_model, n_heads, self.n_blocks)

        # PC forward model: predicts observation embedding from corrected
        # angle. Same shape as the original PC implementation.
        pos_feat_dim = 2 * n_heads * self.n_blocks
        self.forward_model = nn.Sequential(
            nn.Linear(pos_feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

        # Buffer to accumulate aux loss for retrieval after forward pass
        self.prediction_error_squared = torch.tensor(0.0)

    def _compute_theta_path(
        self, delta_2d: torch.Tensor
    ) -> torch.Tensor:
        """Reproduce GridPathIntegrator's angle computation but return
        unwrapped angles in (B, L, H, NB) shape for InEKF compatibility.

        GridPathIntegrator.forward returns ``cos(angles), sin(angles)`` in
        a (B, H, L, M, O) shape; we instead need the raw angles for the
        Kalman update and for the PC forward model's ``cos/sin`` input.
        """
        B, L, H, M, _ = delta_2d.shape
        delta_x = delta_2d[..., 0]
        delta_y = delta_2d[..., 1]
        cos_orient = torch.cos(self.path_integrator.orientation_angles)
        sin_orient = torch.sin(self.path_integrator.orientation_angles)
        d_block = (
            delta_x.unsqueeze(-1) * cos_orient
            + delta_y.unsqueeze(-1) * sin_orient
        )                                                 # (B, L, H, M, O)
        cum_delta = torch.cumsum(d_block, dim=1)
        omega_expanded = (
            self.path_integrator.omega
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )                                                 # (1, 1, H, M, 1)
        angles = cum_delta * omega_expanded               # (B, L, H, M, O)
        # Flatten (M, O) → NB so InEKF / PC see one block dim
        return angles.reshape(B, L, H, self.n_blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)                        # (B, L, d_model)

        # 2D per-module deltas (Grid)
        delta_2d = self.action_to_lie(x)                  # (B, L, H, M, 2)

        # Path-integrated angles, shape compatible with InEKF & PC
        theta_path = self._compute_theta_path(delta_2d)   # (B, L, H, NB)

        # Level 1.5 InEKF correction
        theta_hat, Pi, K, R = self.inekf(theta_path, x)   # (B, L, H, NB)

        # PC forward model on the *corrected* θ̂ (representation pressure
        # operates on the model's best position estimate, not the raw
        # dead-reckoning).
        pos_feat = torch.cat(
            [torch.cos(theta_hat), torch.sin(theta_hat)], dim=-1
        )                                                 # (B, L, H, 2*NB)
        pos_feat = pos_feat.reshape(B, L, -1)             # (B, L, H*2*NB)
        obs_pred = self.forward_model(pos_feat)           # (B, L, d_model)

        # Aux loss: ‖x - ô‖² at observation positions (odd indices)
        err_raw = x - obs_pred                            # (B, L, d_model)
        obs_mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        obs_mask[:, 1::2] = True
        mask_f = obs_mask.unsqueeze(-1).to(err_raw.dtype)
        err = err_raw * mask_f
        denom = max(1.0, mask_f.sum().item() * err.shape[-1])
        self.prediction_error_squared = err.pow(2).sum() / denom

        # Save tensors for diagnostics / downstream analysis
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K.detach()
        self.last_R = R.detach()
        self.last_pred_err = err_raw.detach()

        # RoPE with corrected θ̂. theta_hat shape (B, L, H, NB) →
        # (B, H, L, NB) for the layer's _apply_rope helper.
        theta_for_rope = theta_hat.transpose(1, 2)        # (B, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)

    def prediction_error_loss(self) -> torch.Tensor:
        """Auxiliary loss: ‖prediction error‖² at observation positions.

        The aux loss returned here is intended to be added to the next-token
        loss during training, weighted by ``aux_coef``::

            total_loss = next_token_loss + aux_coef * model.prediction_error_loss()

        Acts as a representation regulariser: forces the forward model to
        predict observation embeddings from corrected-position features,
        which in turn pressures upstream computation (path integration +
        InEKF correction) to produce position-discriminative θ̂. The aux
        loss is masked at action positions, since action tokens cannot be
        predicted from a rotation alone.
        """
        return self.prediction_error_squared


class MapFormerWM_GridL15PC_Free(MapFormerWM_GridL15PC):
    """Combined model with LEARNABLE orientations.

    Tests whether fixed {0°, 60°, 120°} orientations are over-constraining.
    Initialised at hex angles, but the orientations are nn.Parameters that
    gradient descent can adjust during training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 132,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        grid_size: int = 64,
        bottleneck_r: int = 2,
        n_modules: int = 11,
        n_orientations: int = 3,
    ):
        super().__init__(
            vocab_size, d_model, n_heads, n_layers, dropout,
            grid_size, bottleneck_r,
            n_modules=n_modules, n_orientations=n_orientations,
            learnable_orientations=True,
        )
