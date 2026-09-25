"""Gaussian Sum Filter (GSF) — K parallel Level 1.5 InEKFs with mixture weighting.

Tests whether multi-modal Bayesian filtering recovers the TEMFaithful
landmark advantage while preserving MapFormer's parallel-scan property.

Each of K modes is an independent Level 1.5 InEKF chain that shares the
same action-driven Δ but starts from a different learnable initial offset
θ_init_k. The K chains evolve in parallel via independent affine scans.
Mixture weights update via cumulative log-likelihood of the observed
tokens under each mode (also computed as a parallel cumsum).

Forward pass structure:

  1. action_to_lie + cumsum  (shared across K, parallel)
  2. K different θ_init_k offsets added → K parallel path-integrated chains
  3. K parallel InEKF corrections (each chain runs independently)
  4. K parallel attention passes (each with its own RoPE rotation from θ̂^k)
  5. K parallel output logits
  6. Mixture log-weights: per-step log p(o_t | mode k), cumsum'd in parallel
  7. Final logits = log Σ_k p(mode k) * exp(logits^k)
                  = logsumexp_k(logits^k + log_w_k)

This is the "Interactive Multiple Model" / Gaussian Sum Filter from
radar tracking literature, adapted to our InEKF + transformer setting.

Total compute is ~K× single Level 1.5 (each chain is independent).
For K=8, ~8× wall-clock per step. Memory similarly K×.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerWM, WMTransformerLayer, _apply_rope
from .model_inekf_level15 import assoc_scan_affine_scalar, InEKFLevel15


class MapFormerWM_Level15GSF(MapFormerWM):
    """Level 1.5 InEKF with K parallel modes (Gaussian Sum Filter).

    The K modes differ in their initial position offset θ_init_k. All
    other parameters (action_to_lie, ω, measure_head, log_R_head, log_Π,
    attention, output projection) are SHARED across modes.

    Why share most parameters? The mode index serves as a multi-modal
    prior over "where did the trajectory start"; once the prior is fixed,
    the dynamics (action transitions) and observation model (measure_head)
    are the same. Sharing keeps the model compact while letting the K
    modes hedge over starting-position hypotheses.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 n_modes: int = 8):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = InEKFLevel15(d_model, n_heads, self.n_blocks)
        self.n_modes = n_modes

        # K learnable initial offsets θ_init_k ∈ R^{n_heads × n_blocks}.
        # Spread them uniformly on each block's angular range at init to
        # encourage mode diversity. Per-block we span [-π/2, π/2]; the
        # softmax mixture weight + gradient pressure will adjust at training.
        offsets = torch.linspace(-math.pi / 2, math.pi / 2, n_modes)
        theta_init = offsets.view(n_modes, 1, 1).expand(n_modes, n_heads, self.n_blocks).clone()
        # Small per-head/block perturbation to break symmetry
        theta_init = theta_init + 0.05 * torch.randn(n_modes, n_heads, self.n_blocks)
        self.theta_init = nn.Parameter(theta_init)               # (K, H, NB)

        # Learnable prior over modes (initialised uniform)
        self.log_prior = nn.Parameter(torch.zeros(n_modes))

    def _inekf_per_mode(self, theta_path_modes: torch.Tensor, content_emb: torch.Tensor):
        """Run InEKF separately for K modes. Shapes:
            theta_path_modes: (B, K, L, H, NB)
            content_emb:      (B, L, d_model)
        Returns:
            theta_hat: (B, K, L, H, NB)
        """
        B, K, L, H, NB = theta_path_modes.shape

        # log_R_head and measure_head depend on content only; same across modes.
        log_R = self.inekf.log_R_head(content_emb).view(B, L, H, NB).clamp(min=-5.0, max=5.0)
        R = log_R.exp()                                                  # (B, L, H, NB)
        z = math.pi * torch.tanh(
            self.inekf.measure_head(content_emb).view(B, L, H, NB)
        )                                                                # (B, L, H, NB)

        # Per-mode wrapped innovation: z is mode-independent; θ_path is mode-specific.
        z_b = z.unsqueeze(1)                                              # (B, 1, L, H, NB)
        diff = z_b - theta_path_modes                                     # (B, K, L, H, NB)
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))

        # K_t per mode: same Pi, same R, but the affine scan d_t differs because nu differs.
        Pi = self.inekf.log_Pi.exp()                                      # (H, NB)
        Pi_b = Pi.unsqueeze(0).unsqueeze(0).unsqueeze(0)                  # (1, 1, 1, H, NB)
        R_b = R.unsqueeze(1)                                              # (B, 1, L, H, NB)
        K_gain = Pi_b / (Pi_b + R_b).clamp_min(1e-8)                      # (B, 1, L, H, NB), shared
        K_gain = K_gain.expand(B, K, L, H, NB)
        alpha = 1.0 - K_gain
        u = K_gain * nu                                                   # (B, K, L, H, NB)

        # Affine scan over time dimension. assoc_scan_affine_scalar expects
        # (B, L, ...) where ... is the broadcast tail. Fold K into the tail
        # by transposing.
        # alpha, u: (B, K, L, H, NB) -> (B, L, K, H, NB) for scan -> back.
        alpha_t = alpha.transpose(1, 2).contiguous()                      # (B, L, K, H, NB)
        u_t = u.transpose(1, 2).contiguous()                              # (B, L, K, H, NB)
        d_t = assoc_scan_affine_scalar(alpha_t, u_t)                      # (B, L, K, H, NB)
        d = d_t.transpose(1, 2).contiguous()                              # (B, K, L, H, NB)

        return theta_path_modes + d                                       # (B, K, L, H, NB)

    def _attention_per_mode(self, x: torch.Tensor, theta_hat_modes: torch.Tensor):
        """Run the transformer layers for each of K modes in parallel.

        x:                (B, L, d_model)          — same content for all modes
        theta_hat_modes:  (B, K, L, H, NB)         — per-mode RoPE angles
        Returns: (B, K, L, d_model)
        """
        B, K, L, H, NB = theta_hat_modes.shape
        # Replicate x across modes; fold mode dim into batch.
        x_bk = x.unsqueeze(1).expand(B, K, L, x.shape[-1]).reshape(B * K, L, x.shape[-1])

        # Per-mode RoPE angles, folded into batch dim
        theta_bk = theta_hat_modes.reshape(B * K, L, H, NB)
        theta_for_rope = theta_bk.transpose(1, 2)                          # (B*K, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x_bk = layer(x_bk, cos_a, sin_a, causal_mask)

        x_bk = self.out_norm(x_bk)
        return x_bk.reshape(B, K, L, x.shape[-1])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        K = self.n_modes
        x = self.token_emb(tokens)                                         # (B, L, d_model)

        # Path integration — shared across modes
        delta = self.action_to_lie(x)                                      # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path_shared = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        #                                                                  (B, L, H, NB)

        # K modes: each starts at its own offset θ_init_k
        theta_init_b = self.theta_init.unsqueeze(0).unsqueeze(2)           # (1, K, 1, H, NB)
        theta_path_shared_k = theta_path_shared.unsqueeze(1)               # (B, 1, L, H, NB)
        theta_path_modes = theta_path_shared_k + theta_init_b              # (B, K, L, H, NB)

        # K parallel InEKF corrections
        theta_hat_modes = self._inekf_per_mode(theta_path_modes, x)        # (B, K, L, H, NB)

        # K parallel attention passes
        h_modes = self._attention_per_mode(x, theta_hat_modes)             # (B, K, L, d_model)

        # K parallel logits
        logits_modes = self.out_proj(h_modes)                              # (B, K, L, vocab)

        # Mixture log-weights at each time step:
        #   log p(mode k | observations up to t) ∝ log p_prior(k) + Σ_{s≤t} log p(o_s | mode k)
        # We compute log p(o_s | mode k) using each mode's logits at position s-1 (one-step-ahead
        # prediction), evaluated at the actually-observed token o_s. For our task all loss is at
        # revisit positions; here we just accumulate the log-likelihood at obs positions (odd
        # indices in the input, mirroring train.py's loss masking pattern).
        #
        # Implementation: get log p_{mode k}(token at t+1) = log_softmax(logits_modes[..., t, :])[..., token_{t+1}]
        # then cumsum across time.
        log_p_per_step = F.log_softmax(logits_modes, dim=-1)               # (B, K, L, vocab)
        # Gather log-prob of the NEXT token at each position. tokens[:, 1:] is target.
        # For position t in [0, L-1), the model at position t predicts tokens[:, t+1].
        tgt = tokens[:, 1:].unsqueeze(1).expand(B, K, L - 1)                # (B, K, L-1)
        log_p_tgt = log_p_per_step[:, :, :-1, :].gather(
            -1, tgt.unsqueeze(-1)
        ).squeeze(-1)                                                       # (B, K, L-1)

        # Cumsum the log-likelihood (parallel scan). Pad with 0 at position 0.
        zeros = torch.zeros(B, K, 1, device=tokens.device, dtype=log_p_tgt.dtype)
        log_lik_cum = torch.cumsum(
            torch.cat([zeros, log_p_tgt], dim=-1), dim=-1
        )                                                                   # (B, K, L)

        # Mixture posterior over modes at each step:
        #   log_w_k(t) = log p_prior(k) + log_lik_cum(t, k)
        log_w = self.log_prior.view(1, K, 1) + log_lik_cum                  # (B, K, L)

        # Marginalised logits at each step:
        #   log p(token_{t+1}) = logsumexp_k [log w_k(t) - logsumexp_k' log w_k'(t) + logits_k(t)]
        # Equivalent to: softmax(log_w) over k, then weighted sum of softmax(logits_k).
        # In log-space:
        log_w_norm = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)     # (B, K, L)
        # Add log mixture weight to each mode's logits and logsumexp over modes
        # logits_modes: (B, K, L, vocab); log_w_norm: (B, K, L, 1)
        weighted_logits = logits_modes + log_w_norm.unsqueeze(-1)
        marginal_logits = torch.logsumexp(weighted_logits, dim=1)            # (B, L, vocab)

        # Stash diagnostics
        self.last_theta_path = theta_path_shared.detach()
        self.last_theta_hat_modes = theta_hat_modes.detach()
        self.last_log_w = log_w.detach()

        return marginal_logits                                               # (B, L, vocab)
