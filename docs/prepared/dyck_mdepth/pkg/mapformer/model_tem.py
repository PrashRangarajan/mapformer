"""TEM-style recurrent baseline (Whittington et al. 2020 — Cell).

A pared-down implementation of the Tolman-Eichenbaum Machine designed
for our 2D-grid revisit-prediction task. Keeps the three load-bearing
TEM ideas:

  1. **Factorised representation**: the recurrent hidden state is split
     into a structural code g (env-invariant, "where") and a content
     code x (sensory, "what").
  2. **Hebbian outer-product memory**: at each step we bind (g_t, x_t)
     into a memory matrix M_t = M_{t-1} + g_t · x_t^T. The memory grows
     additively without parameters; capacity is bounded by d_g · d_x.
  3. **Pattern completion via memory retrieval**: to predict the next
     observation, we use the model's predicted next-step g as a query;
     the retrieved x = g · M^T is decoded to obs.

Differences from the published TEM (be honest about scope):
  - **Single environment**: TEM's primary advantage is multi-env
    compositional generalisation. Trained on one env (our setup), TEM
    loses its main lever. We expect single-env TEM to be ~LSTM tier;
    this is a faithful baseline for our task, NOT a faithful TEM
    deployment.
  - **No explicit action transition matrices**: real TEM uses a
    learned per-action transition operator on g. Here we let the GRU's
    standard recurrence do that implicitly.
  - **No multi-frequency g**: TEM has multiple g modules at different
    spatial frequencies (matching grid-cell modules). We use a single
    monolithic g.
  - **No explicit pattern separation**: TEM uses sparsity / DG-style
    decorrelation. We omit this.

Even simplified, this remains a fair "RNN with structured memory"
comparison against MapFormer's structured rotation. Interface matches
our other baselines: ``forward(tokens) -> logits``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TEMRecurrent(nn.Module):
    """TEM-style recurrent baseline.

    State at each step:
      h_t   : GRU hidden state, dim d_h
      g_t   : structural code, dim d_g (read off from h_t)
      x_t   : content code, dim d_x  (read off from h_t)
      M_t   : Hebbian memory, shape (d_g, d_x)

    Per-step update:
      h_t = GRU(token_emb(s_t), h_{t-1})
      g_t = g_proj(h_t)
      x_t = x_proj(h_t)                           # current sensory code
      M_t = M_{t-1} + g_t · x_t^T                 # Hebbian binding
      x̂_{t+1} = (1−λ)·x_proj_pred(h_t) + λ·(g_query · M_t)   # retrieval
      logits_t = out(x̂_{t+1})

    The model's prediction for "next obs token" is the retrieval-blended
    content, decoded via a final linear layer.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1, grid_size: int = 64,
                 d_g: int | None = None, d_x: int | None = None,
                 retrieval_weight: float = 0.5, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Default: split d_model evenly between structure and content.
        self.d_g = d_g if d_g is not None else d_model // 2
        self.d_x = d_x if d_x is not None else d_model // 2
        self.lambda_retrieve = retrieval_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Single-layer GRU backbone (n_layers > 1 stacks them)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.n_layers = n_layers

        # Read structural and content codes off of the GRU hidden state
        self.g_proj = nn.Linear(d_model, self.d_g)
        self.x_proj = nn.Linear(d_model, self.d_x)
        # Separate "predicted next x" head — used when memory is empty/cold
        self.x_pred = nn.Linear(d_model, self.d_x)

        # Decode the retrieved/predicted content back to the unified vocab
        self.out_norm = nn.LayerNorm(self.d_x)
        self.out_proj = nn.Linear(self.d_x, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L). Returns logits: (B, L, vocab_size)."""
        B, L = tokens.shape
        device = tokens.device
        emb = self.token_emb(tokens)  # (B, L, d_model)

        # Hebbian memory matrix per batch element.
        M = torch.zeros(B, self.d_g, self.d_x, device=device, dtype=emb.dtype)

        # Initial GRU state.
        h = torch.zeros(self.n_layers, B, self.d_model, device=device, dtype=emb.dtype)

        out_logits = []
        # Step through the sequence so we can write/read the Hebbian memory.
        for t in range(L):
            inp = emb[:, t : t + 1, :]           # (B, 1, d_model)
            y, h = self.gru(inp, h)              # y: (B, 1, d_model)
            h_t = y[:, 0, :]                     # (B, d_model)

            g_t = self.g_proj(h_t)               # (B, d_g)
            x_t = self.x_proj(h_t)               # (B, d_x)
            x_p = self.x_pred(h_t)               # (B, d_x) — direct prediction

            # Retrieve via inner-product query against memory.
            #   M:   (B, d_g, d_x)
            #   g_t: (B, d_g)
            #   retr = g_t @ M  -> (B, d_x)
            retr = torch.bmm(g_t.unsqueeze(1), M).squeeze(1)  # (B, d_x)

            # Blend retrieval with direct prediction
            x_hat = (1 - self.lambda_retrieve) * x_p + self.lambda_retrieve * retr

            # Decode to vocab
            logits_t = self.out_proj(self.out_norm(x_hat))  # (B, vocab)
            out_logits.append(logits_t)

            # Bind (g_t, x_t) into memory AFTER reading (so the prediction
            # at step t doesn't get to peek at its own answer).
            M = M + torch.bmm(g_t.unsqueeze(2), x_t.unsqueeze(1))

        return torch.stack(out_logits, dim=1)  # (B, L, vocab_size)
