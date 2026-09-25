"""Level 1.5 InEKF + Successor-Representation auxiliary objective.

The standard next-token CE objective trains the hidden state to encode "where
am I now," but not "which tokens will I see in the next K steps." That's
why goal-distance probes and active-inference protocols failed: the hidden
state has no representation of multi-step reachability.

This variant adds an auxiliary head + loss:

  - `self.sr_head: Linear(d_model, vocab_size)` projects the hidden state at
    every position to a per-vocab logit.
  - Target: binary indicator "did token v appear in positions [t+1, t+K]?"
  - Loss: BCEWithLogits, averaged over (position, token) pairs.

Concretely the model now learns, alongside p(o_{t+1} | seq), a representation
that supports p(v ∈ next-K tokens | seq) for every vocab v. For landmarks
this approximates a successor representation: "will I see landmark X within
K steps?"

If this works, downstream probes (`probe_goal_distance.py`,
`probe_active_inference.py`) should improve dramatically without changing
their decision rules — the model now natively encodes multi-step reachability.

Aux loss is exposed via `prediction_error_loss()`, the standard hook
`train.py` uses (same mechanism as `Level15_DoG`, `Level15PC`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_inekf_level15 import MapFormerWM_Level15InEKF


class MapFormerWM_Level15_SR(MapFormerWM_Level15InEKF):
    """Level 1.5 + successor-representation auxiliary head (horizon K)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 sr_horizon: int = 8):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.sr_horizon = sr_horizon
        self.sr_head = nn.Linear(d_model, vocab_size)
        self._last_sr_logits = None
        self._last_tokens = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        theta_hat, Pi, K_, R = self.inekf(theta_path, x)

        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K_.detach()
        self.last_R = R.detach()

        theta_for_rope = theta_hat.transpose(1, 2)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        # Stash for SR aux loss
        self._last_sr_logits = self.sr_head(x)
        self._last_tokens = tokens
        return self.out_proj(x)

    def prediction_error_loss(self) -> torch.Tensor:
        """BCE-with-logits over the next-K-tokens multi-label target.

        target[b, t, v] = 1 if token v appears at any of positions t+1..t+K
        in the input sequence; 0 otherwise. Loss averaged over valid (b, t)
        pairs (i.e. t such that t+K < L).
        """
        if self._last_sr_logits is None or self._last_tokens is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        tokens = self._last_tokens                                   # (B, L)
        sr_logits = self._last_sr_logits                             # (B, L, V)
        B, L = tokens.shape
        V = sr_logits.shape[-1]
        K = self.sr_horizon
        Lw = L - K                                                   # # valid positions
        if Lw <= 0:
            return torch.tensor(0.0, device=tokens.device)

        # Build (B, Lw, K) windows of future tokens: windows[b, t, k] = tokens[b, t+1+k]
        # via unfold on the shifted sequence.
        shifted = tokens[:, 1:]                                      # (B, L-1)
        # unfold gives (B, L-K, K); first row t=0 covers tokens[1..K]
        windows = shifted.unfold(1, K, 1)                            # (B, Lw, K)

        # Scatter into multi-label targets.
        targets = torch.zeros(B, Lw, V, device=tokens.device)
        b_idx = torch.arange(B, device=tokens.device).view(B, 1, 1).expand(B, Lw, K)
        t_idx = torch.arange(Lw, device=tokens.device).view(1, Lw, 1).expand(B, Lw, K)
        targets[b_idx.flatten(), t_idx.flatten(), windows.flatten()] = 1.0

        sr_logits_valid = sr_logits[:, :Lw, :]
        return F.binary_cross_entropy_with_logits(sr_logits_valid, targets)
