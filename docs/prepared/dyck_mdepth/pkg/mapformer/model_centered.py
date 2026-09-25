"""Bounded-growth path integration: centre the increment so the accumulator is a random walk.

THEORY_MAPPOPE.md T1. On data whose increments do not cancel, S_t = cumsum(Delta) grows linearly
(alpha = 1, a CLOCK) and S_t - S_s leaves its trained range in proportion to sequence length. The
account under test says that is what MapPoPE cannot survive, because PoPE's kernel has no pairwise
phase with which to compensate.

`ActionToLieAlgebra` is linear (W_out W_in, no bias), so subtracting the frequency-weighted mean
EMBEDDING is exactly equivalent to subtracting the expected increment:

    E_{x~p}[Delta] = A( sum_v p_v e_v ) = A(e_bar)   =>   Delta'(x) = A(x - e_bar)  has  E[Delta'] = 0

so S becomes a mean-zero random walk, range ~ sqrt(T) rather than ~T. Additivity and the relative
(difference-only) structure are untouched, which wrapping S or squashing it with a tanh would both
destroy. `p` is the training-split token frequency, a fixed buffer.
"""
import torch
import torch.nn as nn


class CenteredActionToLie(nn.Module):
    def __init__(self, inner, token_emb, freq):
        super().__init__()
        self.inner, self.token_emb = inner, token_emb
        self.register_buffer("freq", freq)          # (vocab,), sums to 1

    def forward(self, x):
        e_bar = (self.freq.unsqueeze(0) @ self.token_emb.weight).squeeze(0)   # (d_model,)
        return self.inner(x - e_bar)


def center_model(model, freq):
    """Wrap a MapFormer's increment map in place. Returns the model."""
    model.action_to_lie = CenteredActionToLie(model.action_to_lie, model.token_emb, freq)
    return model


def token_frequencies(X, M, vocab_size):
    f = torch.bincount(X[M].reshape(-1), minlength=vocab_size).float()
    return f / f.sum()
