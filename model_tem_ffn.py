"""TEMFaithful + a per-position FFN — direct test of the transformer-machinery hypothesis.

TEMFaithful lags MapFormer by ~1-3pp on clean / well-structured regimes.
Hypothesis: that small lag is because TEM lacks per-position nonlinear
feature processing — the FFN, a core piece of transformer machinery.

This variant keeps TEMFaithful's fixed Hopfield retrieval bank entirely
unchanged, and adds ONE piece of transformer machinery: a per-position
FFN applied to the retrieved content x_hat before the output decoder:

    x_hat <- x_hat + FFN(LayerNorm(x_hat))

If this closes the clean-regime lag, the missing-FFN hypothesis is
confirmed: TEM's small clean-task deficit is the absence of per-position
nonlinear processing, not the absence of the whole transformer.

FFN params: 2 * d_x * (4*d_x) ~ 33K at d_x=64 — TEMFaithful_FFN totals
~78K, still ~4x smaller than Level15.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_tem_faithful import TEMFaithful


class TEMFaithful_FFN(TEMFaithful):
    """TEMFaithful + per-position FFN on the retrieved content."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, n_actions=4,
                 d_g=None, d_x=None, beta_init=1.0,
                 identity_init_scale=0.05, **kwargs):
        super().__init__(vocab_size, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, dropout=dropout,
                         grid_size=grid_size, n_actions=n_actions,
                         d_g=d_g, d_x=d_x, beta_init=beta_init,
                         identity_init_scale=identity_init_scale, **kwargs)
        # Per-position FFN on the retrieved content (the one bit of
        # transformer machinery being added). Applied as a residual.
        self.ffn_norm = nn.LayerNorm(self.d_x)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_x, 4 * self.d_x),
            nn.GELU(),
            nn.Linear(4 * self.d_x, self.d_x),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Identical to TEMFaithful.forward except x_hat passes through a
        per-position FFN residual before the output decoder."""
        B, L = tokens.shape
        device = tokens.device
        dtype = self.g_init.dtype

        W_a_all = self._orthogonal_W()
        g = self.g_init.unsqueeze(0).expand(B, -1).contiguous().to(dtype)

        mem_g_list: list[torch.Tensor] = []
        mem_x_list: list[torch.Tensor] = []
        outputs = []

        for t in range(L):
            tok = tokens[:, t]
            is_action = tok < self.n_actions
            obs_mask = (~is_action).to(dtype)

            # ----- UPDATE g FIRST (action-driven) -----
            action_idx = torch.where(is_action, tok, torch.zeros_like(tok))
            W_batch = W_a_all[action_idx]
            g_updated = torch.bmm(W_batch, g.unsqueeze(-1)).squeeze(-1)
            action_mask = (~obs_mask.bool()).to(dtype).unsqueeze(-1)
            g = action_mask * g_updated + (1.0 - action_mask) * g

            # ----- PREDICT using updated g -----
            if len(mem_g_list) > 0:
                Mg = torch.stack(mem_g_list, dim=1)
                Mx = torch.stack(mem_x_list, dim=1)
                scores = torch.bmm(g.unsqueeze(1), Mg.transpose(1, 2)).squeeze(1)
                scores = scores * self.beta
                attn = F.softmax(scores, dim=-1)
                x_hat = torch.bmm(attn.unsqueeze(1), Mx).squeeze(1)
            else:
                x_hat = torch.zeros(B, self.d_x, device=device, dtype=dtype)

            # ----- NEW: per-position FFN residual on the retrieved content -----
            x_hat = x_hat + self.ffn(self.ffn_norm(x_hat))

            logits_t = self.out_proj(self.out_norm(x_hat))
            outputs.append(logits_t)

            # ----- BIND (g, x) at obs tokens -----
            x_full = self.content_emb(tok)
            x_masked = x_full * obs_mask.unsqueeze(-1)
            mem_g_list.append(g)
            mem_x_list.append(x_masked)

        return torch.stack(outputs, dim=1)
