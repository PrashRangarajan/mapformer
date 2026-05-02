"""MapFormer variants for continuous (action, observation) inputs.

Used by the Cueva/Wei/Sorscher-style continuous 2D nav task in
``continuous_nav.py``. The model takes:
    actions:     (B, T, action_dim)   continuous (v, ω) per step
    obs:         (B, T, obs_dim)      DoG place-cell vectors per step

and outputs predicted DoG vectors at action positions (i.e., predicting
the obs that follows each action). Trained with MSE.

Two variants:
  - MapFormerWM_Continuous          : vanilla MapFormer (no correction)
  - MapFormerWM_Continuous_Level15  : Level 1.5 InEKF on θ̂

The architecture re-uses ``ActionToLieAlgebra``, ``PathIntegrator``, and
``WMTransformerLayer`` from ``model.py`` and ``InEKFLevel15`` from
``model_inekf_level15.py``. Only the input/output heads differ from
the discrete-token MapFormer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .model import (
    ActionToLieAlgebra, PathIntegrator, WMTransformerLayer,
    EMTransformerLayer, _apply_rope,
)
from .model_inekf_level15 import InEKFLevel15


def _interleave(a: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Build interleaved [a0, o0, a1, o1, ...] from (B, T, D) tensors a, o."""
    B, T, D = a.shape
    x = torch.empty(B, 2 * T, D, device=a.device, dtype=a.dtype)
    x[:, 0::2] = a
    x[:, 1::2] = o
    return x


class _ContinuousBackbone(nn.Module):
    """Common forward to the LayerNorm — produces (B, 2T, d_model) hidden states.

    Subclasses provide:
      - ``_compute_theta_for_rope(x)`` -> theta angles for RoPE; for vanilla
        this is the path-integrated θ; for Level15 it's the InEKF-corrected θ̂.
    """

    def __init__(self, action_dim: int, obs_dim: int, d_model: int = 128,
                 n_heads: int = 2, n_layers: int = 1, dropout: float = 0.1,
                 grid_size: int = 64, bottleneck_r: int = 2,
                 n_grid_units: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.grid_size = grid_size
        self.n_grid_units = n_grid_units

        # Continuous input projections (replace token_emb)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.obs_proj    = nn.Linear(obs_dim,    d_model)

        # Same path-integration machinery as discrete MapFormer
        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, bottleneck_r)
        self.path_integrator = PathIntegrator(n_heads, self.n_blocks, grid_size)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        # Output head:
        #  - n_grid_units == 0: direct Linear (d_model -> obs_dim)
        #  - n_grid_units >  0: Linear -> ReLU (the candidate hex layer) -> Linear
        # The ReLU layer satisfies Sorscher's non-negativity condition; with
        # DoG-of-position targets it should produce hex-grid representations
        # if the architecture meets all three of his conditions.
        if n_grid_units > 0:
            self.grid_proj = nn.Linear(d_model, n_grid_units)
            self.out_head  = nn.Linear(n_grid_units, obs_dim)
        else:
            self.grid_proj = None
            self.out_head  = nn.Linear(d_model, obs_dim)
        self.last_grid_activations: torch.Tensor | None = None

    def _embed(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        a = self.action_proj(actions)
        o = self.obs_proj(obs)
        return _interleave(a, o)                                # (B, 2T, d_model)

    def _path_theta(self, x: torch.Tensor) -> torch.Tensor:
        """Path-integrated θ before any correction. (B, L, H, NB)."""
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        return cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

    def _run_attention(self, x: torch.Tensor, theta_for_rope: torch.Tensor) -> torch.Tensor:
        """Run all transformer layers and produce final hidden states (B, L, d_model)."""
        L = x.shape[1]
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        return self.out_norm(x)

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        """Apply optional ReLU bottleneck + DoG output head."""
        if self.grid_proj is not None:
            g = torch.relu(self.grid_proj(h))
            self.last_grid_activations = g
            return self.out_head(g)
        return self.out_head(h)


class MapFormerWM_Continuous(_ContinuousBackbone):
    """Vanilla continuous MapFormer-WM: path integration with no correction."""

    def forward(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = self._embed(actions, obs)                          # (B, 2T, d_model)
        theta = self._path_theta(x)                            # (B, 2T, H, NB)
        theta_for_rope = theta.transpose(1, 2)                 # (B, H, 2T, NB)
        h = self._run_attention(x, theta_for_rope)             # (B, 2T, d_model)
        return self._readout(h)                                # (B, 2T, obs_dim)


class MapFormerWM_Continuous_Level15(_ContinuousBackbone):
    """Level 1.5 InEKF on θ̂, with continuous inputs. Identical attention path
    to the vanilla variant; only the rotation angles fed to RoPE are corrected."""

    def __init__(self, *args, log_R_init_bias: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.inekf = InEKFLevel15(self.d_model, self.n_heads, self.n_blocks,
                                   log_R_init_bias=log_R_init_bias)

    def forward(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = self._embed(actions, obs)
        theta_path = self._path_theta(x)
        theta_hat, Pi, K, R = self.inekf(theta_path, x)
        # Diagnostic stash (mirrors discrete Level15)
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat  = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K  = K.detach()
        self.last_R  = R.detach()

        theta_for_rope = theta_hat.transpose(1, 2)
        h = self._run_attention(x, theta_for_rope)
        return self._readout(h)


# =============================================================================
# EM (Episodic Memory) variants — Hadamard A_X ⊙ A_P attention with separate
# learnable position-only q_0^p / k_0^p rotated by path-integrated angle.
# =============================================================================


class _ContinuousBackboneEM(nn.Module):
    """EM-style backbone: position and content decoupled into two attention
    pathways, combined multiplicatively. Otherwise identical interface to
    the WM backbone (same input projections, same readout / hex bottleneck)."""

    def __init__(self, action_dim: int, obs_dim: int, d_model: int = 128,
                 n_heads: int = 2, n_layers: int = 1, dropout: float = 0.1,
                 grid_size: int = 64, bottleneck_r: int = 2,
                 n_grid_units: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.grid_size = grid_size
        self.n_grid_units = n_grid_units

        self.action_proj = nn.Linear(action_dim, d_model)
        self.obs_proj    = nn.Linear(obs_dim,    d_model)

        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, bottleneck_r)
        self.path_integrator = PathIntegrator(n_heads, self.n_blocks, grid_size)

        # EM's distinguishing parameters: separate position-only Q and K vectors
        self.q0_pos = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.k0_pos = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)

        self.layers = nn.ModuleList([
            EMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        if n_grid_units > 0:
            self.grid_proj = nn.Linear(d_model, n_grid_units)
            self.out_head  = nn.Linear(n_grid_units, obs_dim)
        else:
            self.grid_proj = None
            self.out_head  = nn.Linear(d_model, obs_dim)
        self.last_grid_activations: torch.Tensor | None = None

    def _embed(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        a = self.action_proj(actions)
        o = self.obs_proj(obs)
        return _interleave(a, o)

    def _path_theta(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        return cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

    def _run_attention_em(self, x: torch.Tensor, theta_for_rope: torch.Tensor) -> torch.Tensor:
        """EM attention: rotate position-only q0_pos / k0_pos with the given
        angles, pass q_pos, k_pos to EMTransformerLayer (Hadamard with content)."""
        B, L, _ = x.shape
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        # Expand learnable position vectors and rotate them
        q0 = self.q0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        k0 = self.k0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        q_pos = _apply_rope(q0, cos_a, sin_a)
        k_pos = _apply_rope(k0, cos_a, sin_a)

        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, q_pos, k_pos, causal_mask)
        return self.out_norm(x)

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        if self.grid_proj is not None:
            g = torch.relu(self.grid_proj(h))
            self.last_grid_activations = g
            return self.out_head(g)
        return self.out_head(h)


class MapFormerEM_Continuous(_ContinuousBackboneEM):
    """Vanilla continuous MapFormer-EM (no InEKF correction)."""

    def forward(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = self._embed(actions, obs)
        theta = self._path_theta(x)
        theta_for_rope = theta.transpose(1, 2)
        h = self._run_attention_em(x, theta_for_rope)
        return self._readout(h)


class MapFormerEM_Continuous_Level15(_ContinuousBackboneEM):
    """Continuous MapFormer-EM + Level 1.5 InEKF on θ̂.

    Uses ``log_R_init_bias=3.0`` by default (matches the discrete Level15EM
    convention) — EM's Hadamard A_X ⊙ A_P attention has no fallback path
    if A_P is corrupted by random θ̂ at init, so we make the InEKF a
    near-no-op at init and let R_t learn down from there."""

    def __init__(self, *args, log_R_init_bias: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.inekf = InEKFLevel15(self.d_model, self.n_heads, self.n_blocks,
                                   log_R_init_bias=log_R_init_bias)

    def forward(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = self._embed(actions, obs)
        theta_path = self._path_theta(x)
        theta_hat, Pi, K, R = self.inekf(theta_path, x)
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat  = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K  = K.detach()
        self.last_R  = R.detach()

        theta_for_rope = theta_hat.transpose(1, 2)
        h = self._run_attention_em(x, theta_for_rope)
        return self._readout(h)
