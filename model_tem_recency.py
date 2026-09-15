"""TEM on the recency (k-back) task. Adapted from `model_tem_faithful.py`; every change is stated here.

TEM's defining mechanics are kept: a structural code g updated by input-specific ORTHOGONAL transition matrices
(W = exp(skew(A)), as in TEMFaithful), memory binding of (g, x) conjunctions, and modern-Hopfield retrieval that
queries memory with a STRUCTURAL code only (content never enters the query).

Changes needed for a task with no actions, all stated:
  1. Transitions per TOKEN ID (89 tokens), not per action. Nothing tells the model which tokens are symbols,
     filler or queries; it must learn e.g. that filler should be the identity. (MapFormer likewise learns a
     step per token.)
  2. Every token is bound into memory, with a learned per-token scalar added to its retrieval score, so the
     model can learn to ignore filler and query entries. (TEMFaithful binds observation tokens by a fixed
     convention; recency has no such convention to exploit.)
  3. Two variants of how the retrieval query is formed:
       TEMRecency        query_t = g_t, the state itself (faithful TEM). A query token's transition then
                         also moves the state for every later token.
       TEMRecency_Query  query_t = V_tok . g_t with a second per-token orthogonal matrix V, so a query token can
                         look back without moving the state (a non-committing lookup).
  4. Retrieval is computed in parallel (one causal attention over all earlier bound entries) instead of a Python
     loop; the state recursion g_t = W_tok(t) g_{t-1} is still sequential. `sequential_reference` checks the
     two agree.

With g in 2-D rotation blocks, g_t . g_s = sum_i |g_i|^2 cos(omega_i (S_t - S_s)), so TEM's retrieval kernel is
MapEM's single-origin kernel. The difference is that TEM's per-query transform V (or W) is a full orthogonal
matrix, not a rank-4 bottleneck scaled by frequency -- which makes this arm a test of whether EM's recency deficit
is the bottleneck parameterisation or the shared-kernel design itself.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TEMRecency(nn.Module):
    SEPARATE_QUERY = False

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64,
                 d_g=None, d_x=None, identity_init_scale=0.05, **kw):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_g = d_g or d_model // 2
        self.d_x = d_x or d_model // 2
        self.g_init = nn.Parameter(torch.randn(self.d_g) / math.sqrt(self.d_g))
        self.A = nn.Parameter(identity_init_scale * torch.randn(vocab_size, self.d_g, self.d_g))
        if self.SEPARATE_QUERY:
            self.Aq = nn.Parameter(identity_init_scale * torch.randn(vocab_size, self.d_g, self.d_g))
        self.content_emb = nn.Embedding(vocab_size, self.d_x)
        self.bind_bias = nn.Embedding(vocab_size, 1)
        nn.init.zeros_(self.bind_bias.weight)
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))
        self.out_norm = nn.LayerNorm(self.d_x)
        self.out_proj = nn.Linear(self.d_x, vocab_size)

    @staticmethod
    def _orth(A):
        return torch.matrix_exp(0.5 * (A - A.transpose(-1, -2)))

    def states(self, tokens):
        B, L = tokens.shape
        W = self._orth(self.A)                              # (V, d, d)
        g = self.g_init.unsqueeze(0).expand(B, -1)
        out = []
        for t in range(L):
            g = torch.bmm(W[tokens[:, t]], g.unsqueeze(-1)).squeeze(-1)
            out.append(g)
        G = torch.stack(out, 1)                             # (B, L, d)
        if self.SEPARATE_QUERY:
            Q = torch.einsum("blij,blj->bli", self._orth(self.Aq)[tokens], G)
        else:
            Q = G
        return G, Q

    def forward(self, tokens):
        B, L = tokens.shape
        G, Q = self.states(tokens)
        scores = self.log_beta.exp() * (Q @ G.transpose(1, 2)) / math.sqrt(self.d_g)
        scores = scores + self.bind_bias(tokens).transpose(1, 2)            # (B, 1, L) key bias
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), 0)  # only s < t
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)          # row 0 has no memory
        x_hat = attn @ self.content_emb(tokens)
        return self.out_proj(self.out_norm(x_hat))

    @torch.no_grad()
    def sequential_reference(self, tokens):
        """Token-by-token TEM loop (predict from memory of earlier tokens, then bind). For checking only."""
        B, L = tokens.shape
        G, Q = self.states(tokens)
        X = self.content_emb(tokens); bb = self.bind_bias(tokens).squeeze(-1)
        outs = []
        for t in range(L):
            if t == 0:
                x_hat = torch.zeros(B, self.d_x, device=tokens.device)
            else:
                s = self.log_beta.exp() * (Q[:, t:t + 1] @ G[:, :t].transpose(1, 2)).squeeze(1) / math.sqrt(self.d_g)
                a = F.softmax(s + bb[:, :t], dim=-1)
                x_hat = (a.unsqueeze(-1) * X[:, :t]).sum(1)
            outs.append(self.out_proj(self.out_norm(x_hat)))
        return torch.stack(outs, 1)


class TEMRecency_Query(TEMRecency):
    SEPARATE_QUERY = True


class TEMRecency_Query_Installed(TEMRecency_Query):
    """Existence check (rule 29): the exact k-back solution written into the transitions and FROZEN.

    g lives in d_g/2 rotation blocks with frequencies omega_i spaced geometrically from pi/2 down to
    2*pi/512 (no aliasing over a few hundred symbols). Symbol tokens rotate g by omega (one count step);
    filler, query and mask tokens leave the state unchanged; query token q_k's separate query transform
    rotates by -(k-1) omega, a rewind that does not move the state. Only content embeddings, binding biases,
    beta and the decoder are trained.
    """

    def __init__(self, vocab_size, d_model=128, n_symbols=16, n_filler=8, k_max=64, **kw):
        super().__init__(vocab_size, d_model=d_model, **kw)
        nb = self.d_g // 2
        om = (math.pi / 2) * ((2 * math.pi / 512) / (math.pi / 2)) ** (torch.arange(nb) / (nb - 1))
        J = torch.zeros(self.d_g, self.d_g)
        for i in range(nb):
            J[2 * i + 1, 2 * i] = om[i]; J[2 * i, 2 * i + 1] = -om[i]
        A = torch.zeros(vocab_size, self.d_g, self.d_g); Aq = torch.zeros_like(A)
        A[:n_symbols] = J                                             # symbols advance the count
        q0 = n_symbols + n_filler
        for k in range(1, k_max + 1):
            Aq[q0 + k - 1] = -(k - 1) * J                             # q_k rewinds the query only
        with torch.no_grad():
            self.A.copy_(A); self.Aq.copy_(Aq)
            self.g_init.copy_(torch.ones(self.d_g) / math.sqrt(2))    # equal energy in every block
        self.A.requires_grad_(False); self.Aq.requires_grad_(False); self.g_init.requires_grad_(False)
