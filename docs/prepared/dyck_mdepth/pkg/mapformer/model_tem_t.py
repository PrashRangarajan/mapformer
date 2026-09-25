"""TEM-t (Whittington et al. 2022, ICLR): TEM as a Transformer.

Faithful implementation of the architecture from
"Relating Transformers to Models and Neural Representations of the
Hippocampal Formation" (arXiv:2112.04035). Three load-bearing
modifications to a standard transformer:

  (1) Q and K share weights and come ONLY from position encodings:
        Q = K = E · W_e         (i.e., the "Ẽ" matrix in the paper)
      V comes ONLY from stimuli:
        V = X · W_x             (i.e., the "X̃" matrix)
      Standard self-attention then operates on these:
        y_t = softmax(ẽ_t · Ẽ^T / √d_k) · X̃

  (2) Causal: each query attends to previous positions only.

  (3) Position encodings are recurrently generated via per-action
      transition matrices with a nonlinearity:
        e_{t+1} = σ(e_t · W_a)
      where W_a is per-action and σ = ReLU per the paper's diagrams.

Adaptation to our task:
  - Our token stream interleaves (a_0, o_0, a_1, o_1, ...). At input
    position 2t (action a_t) we update e via W_{a_t}; at input
    position 2t+1 (obs o_t) we bind (e_t, stim_emb(o_t)) into the
    causal memory. All predictions still come from attention over
    past memory.

  - Single-environment training, like our other baselines. TEM-t's
    distinctive advantage is multi-env compositional generalisation;
    that's a separate experiment (open in project_state.md).

Compared to MapFormer-EM (the paper's structurally-equivalent claim):
  - MapFormer-EM: parallel cumsum of f_Δ(x_t) over content; rotation
    update of q0_pos / k0_pos. Hadamard A_X ⊙ A_P attention.
  - TEM-t: sequential per-action W_a update of e (with ReLU).
    Q=K from positions, V from stimuli — extreme separation.

Same total parameter scale (transformer scaffolding: FFN + multi-head
projections), only differ in the position-encoding mechanism.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TEM_T(nn.Module):
    """TEM-t: causal transformer with recurrent position encodings via per-action W_a."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        grid_size: int = 64,           # unused (kept for interface parity)
        n_actions: int = 4,
        identity_init_scale: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_actions = n_actions
        self.dropout = dropout

        # Stimulus embedding — feeds the V branch (X̃ in the paper)
        self.stim_emb = nn.Embedding(vocab_size, d_model)

        # Initial position encoding e_0
        self.e_init = nn.Parameter(torch.randn(d_model) * 0.02)

        # Per-action transition matrices W_a (e_{t+1} = ReLU(e_t · W_a))
        # Initialised as identity + small noise so actions start as
        # near-no-ops and learn to specialise.
        W = torch.eye(d_model).unsqueeze(0).repeat(n_actions, 1, 1)
        W = W + identity_init_scale * torch.randn_like(W)
        self.W_a = nn.Parameter(W)                          # (n_actions, d_model, d_model)

        # LayerNorm — TWO instances, with different roles:
        #
        # 1) `e_pre_attn`: the paper-faithful location. From the TEM-t
        #    appendix: "We find that using layernorm on the positional
        #    encodings (NOT in the RNN, but on the input to transformer)
        #    to be beneficial. For simplicity, we use fixed weights on
        #    the layer norm, i.e. it is just a z-score of g." We use
        #    learnable LayerNorm rather than fixed-weight z-score —
        #    minor difference; both standardise.
        #
        # 2) `e_in_rnn`: an additional in-RNN normalisation that the
        #    paper explicitly does NOT use. They stabilise the RNN via
        #    sensory-landmark-based memory retrieval ("what positional
        #    encoding did I have the last time I saw this landmark") —
        #    a soft reset mechanism. We don't have that, and at our
        #    sequence length (L=255 input) ||e|| explodes to ~1e13 in
        #    the unconstrained recurrence (smoke-tested), producing NaN
        #    gradients by epoch 5. The in-RNN LN is our pragmatic
        #    stabilisation; flagged here as a deviation from the paper.
        #
        # The honest characterisation: this is TEM-t with one extra LN.
        self.e_pre_attn = nn.LayerNorm(d_model)
        self.e_in_rnn   = nn.LayerNorm(d_model)

        # Position-encoding projection: Q = K = E · W_e
        # Stimulus projection:           V = X · W_x
        self.W_e = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=False)

        # Output projection (after attention, residual, FFN)
        self.o_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Final output head (predict next token)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        if n_layers != 1:
            # Paper uses a single attention block; we stay faithful.
            # Keeping the kwarg for interface parity but ignoring it.
            pass

    def _compute_position_encodings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Sequentially compute e_t for each input position.

        At input position 2t (action token a_t), update e via W_a.
        At input position 2t+1 (obs token), e is unchanged.

        Returns: (B, L, d_model).
        """
        B, L = tokens.shape
        device = tokens.device
        dtype = self.e_init.dtype

        e = self.e_init.unsqueeze(0).expand(B, -1).contiguous().to(dtype)  # (B, d_model)
        e_seq = []

        for t in range(L):
            tok = tokens[:, t]                              # (B,)
            is_action = tok < self.n_actions                # (B,) bool

            # Action branch: e <- LN(ReLU(e · W_{a_t}))
            # Obs branch: e unchanged.
            # Compute action update for the whole batch, mask afterwards.
            action_idx = torch.where(is_action, tok, torch.zeros_like(tok))
            W_batch = self.W_a[action_idx]                  # (B, d, d)
            # e: (B, d), W_batch: (B, d, d) — apply as e_t = e @ W
            e_act = torch.bmm(e.unsqueeze(1), W_batch).squeeze(1)
            e_act = F.relu(e_act)                           # σ = ReLU per paper
            e_act = self.e_in_rnn(e_act)                    # stabilisation (deviation from paper)

            mask = is_action.to(dtype).unsqueeze(-1)
            e = mask * e_act + (1.0 - mask) * e
            e_seq.append(e)

        return torch.stack(e_seq, dim=1)                    # (B, L, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L). Returns logits: (B, L, vocab_size)."""
        B, L = tokens.shape

        # 1) Sequentially compute position encodings e_t
        E = self._compute_position_encodings(tokens)         # (B, L, d_model)

        # 2) Paper-faithful: LayerNorm on E before the transformer
        #    ("on the input to transformer"). Standardises the memory
        #    retrieval process so no one memory is up-weighted.
        E = self.e_pre_attn(E)

        # 3) Stimulus embeddings
        X = self.stim_emb(tokens)                            # (B, L, d_model)

        # 4) Q, K from positions; V from stimuli (per Eq. 2 of TEM-t paper)
        Q = self.W_e(E)                                      # (B, L, d_model)
        K = self.W_e(E)                                      # same as Q
        V = self.W_x(X)                                      # (B, L, d_model)

        # Reshape for multi-head
        def split(z):
            return z.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        Qh = split(Q); Kh = split(K); Vh = split(V)          # (B, H, L, d_h)

        # Causal attention
        scores = torch.matmul(Qh, Kh.transpose(-1, -2)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        y = torch.matmul(attn, Vh)                           # (B, H, L, d_h)
        y = y.transpose(1, 2).reshape(B, L, self.d_model)

        # Residual + norm + FFN (standard transformer block)
        # Residual on the V-input side (X embedding) since that's the "content"
        y = self.norm1(self.o_proj(y) + X)
        y = self.norm2(y + self.ffn(y))

        # Output head
        return self.out_proj(self.out_norm(y))
