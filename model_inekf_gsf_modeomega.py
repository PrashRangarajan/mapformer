"""GSF with mode-conditioned per-block frequencies omega_k.

Extends Level15GSF (which varies only theta_init_k across modes) to also
vary the path-integrator's omega scales per mode. Each chain k now has:
  - Its own initial angle theta_init_k     (unchanged from base GSF)
  - Its own per-block frequencies omega_k   (NEW)

Motivation: GSF as currently implemented has K chains that all share the
same "scale of position encoding" via shared omega. Mode-conditioned omega
lets each chain interpret position at a different effective scale, analogous
to hippocampal place-cell remapping between environments.

Bird's-eye view: this is the (2)-variation of GSF — mode-conditioned cognitive
maps. Shared:
  - action_to_lie (action semantics)
  - measurement model (measure_head, log_R_head)
  - Pi, Kalman update structure
  - attention layers
  - output projection

Mode-specific (NEW):
  - omega_modes: (K, H, NB) per-block frequencies per mode
  - theta_init: (K, H, NB) starting angles (unchanged from base GSF)

Param overhead: K * H * NB extra (e.g., 8 * 2 * 16 = 256 params for default
config). ~0.1% of model parameters. Compute overhead: same as base GSF (K×).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .model_inekf_gsf import MapFormerWM_Level15GSF
from .model_inekf_level15 import assoc_scan_affine_scalar


class MapFormerWM_Level15GSF_ModeOmega(MapFormerWM_Level15GSF):
    """GSF with mode-conditioned per-block frequencies."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, n_modes: int = 8):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r, n_modes=n_modes)
        H = n_heads
        NB = self.n_blocks
        # Initialize each mode with the shared omega + small perturbation to break symmetry
        shared_omega = self.path_integrator.omega.detach().clone()  # (H, NB)
        omega_modes = shared_omega.unsqueeze(0).expand(n_modes, H, NB).clone()
        # Perturbation magnitude proportional to typical omega values
        perturb_scale = 0.10 * shared_omega.abs().mean().item()
        omega_modes = omega_modes + perturb_scale * torch.randn(n_modes, H, NB)
        self.omega_modes = nn.Parameter(omega_modes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        K = self.n_modes
        x = self.token_emb(tokens)

        # Path integration — different omega per mode now
        delta = self.action_to_lie(x)                                      # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)                             # (B, L, H, NB)

        # Mode-conditioned path-integrated angle
        # cum_delta: (B, L, H, NB) → (B, 1, L, H, NB)
        # omega_modes: (K, H, NB) → (1, K, 1, H, NB)
        omega_b = self.omega_modes.unsqueeze(0).unsqueeze(2)
        theta_path_modes_base = cum_delta.unsqueeze(1) * omega_b           # (B, K, L, H, NB)

        # Add per-mode initial offset (same as base GSF)
        theta_init_b = self.theta_init.unsqueeze(0).unsqueeze(2)           # (1, K, 1, H, NB)
        theta_path_modes = theta_path_modes_base + theta_init_b            # (B, K, L, H, NB)

        # K parallel InEKF corrections (delegate to parent's helper)
        theta_hat_modes = self._inekf_per_mode(theta_path_modes, x)        # (B, K, L, H, NB)

        # K parallel attention passes (delegate)
        h_modes = self._attention_per_mode(x, theta_hat_modes)             # (B, K, L, d_model)

        # K parallel logits
        logits_modes = self.out_proj(h_modes)                              # (B, K, L, vocab)

        # Mixture log-weights via cumulative log-likelihood (same as base GSF)
        log_p_per_step = torch.nn.functional.log_softmax(logits_modes, dim=-1)
        tgt = tokens[:, 1:].unsqueeze(1).expand(B, K, L - 1)
        log_p_tgt = log_p_per_step[:, :, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        zeros = torch.zeros(B, K, 1, device=tokens.device, dtype=log_p_tgt.dtype)
        log_lik_cum = torch.cumsum(torch.cat([zeros, log_p_tgt], dim=-1), dim=-1)
        log_w = self.log_prior.view(1, K, 1) + log_lik_cum
        log_w_norm = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        weighted_logits = logits_modes + log_w_norm.unsqueeze(-1)
        marginal_logits = torch.logsumexp(weighted_logits, dim=1)

        # Stash for diagnostics
        self.last_theta_hat_modes = theta_hat_modes.detach()
        self.last_log_w = log_w.detach()
        self.last_omega_modes = self.omega_modes.detach().clone()

        return marginal_logits


class MapFormerWM_Level15GSF_NoDrop_ModeOmega(MapFormerWM_Level15GSF_ModeOmega):
    """Mode-conditioned omega + post-attn dropout removed (combines all our wins)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, n_modes: int = 8):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r, n_modes=n_modes)
        # Replace transformer layers with NoDrop variants
        from .model_inekf_level15_nodrop import WMTransformerLayer_NoDrop
        self.layers = nn.ModuleList([
            WMTransformerLayer_NoDrop(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
