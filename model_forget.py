"""MapFormer with a forget gate -- the empty cell of the unified table.

WHERE THIS COMES FROM. In the log-polar form of mapformer_math.tex sec 3.3,

    q_t = exp(c(x_t) - conj(S_t)),   k_s = exp(c(x_s) + S_s),   S_t = sum_{u<=t} G,

real parts of the accumulator are magnitude and imaginary parts are phase. Every
MapFormer variant built here has Re G = 0: there is no forget gate anywhere. That
is the cell Selective RoPE's design principle says should not be empty -- "recall
needs rotation and decay", and their classification of the softmax Transformer is
that it already has rotation implicitly, so "only a forget gate is needed to fully
satisfy the rotation + decay recipe" (the Forgetting Transformer, Lin et al. 2025).

MapFormer makes the rotation EXPLICIT and structured, and adds no decay. This
tests whether the second half of their principle transfers.

THE MECHANISM. A content-dependent decay bias on the logit, per head, added before
the softmax -- additive in log space, multiplicative on the attention weight:

    logit_ts  <-  logit_ts + (L_t - L_s),     L_t = sum_{u<=t} log gamma_u
    log gamma_t^(h) = -lambda * sigmoid(W_f x_t)^(h)

For s <= t this bias is <= 0 when lambda > 0: keys further back are downweighted.
It is a cumsum plus a broadcast difference, so the O(log T) scan property is kept.

WHY lambda IS A RAW SCALAR INITIALISED AT ZERO, and how that deviates from FoX.
The question is whether the optimiser WANTS a forget gate on a cognitive-map task,
so a null has to mean "declined", not "could not express". Initialising a sigmoid
gate near 1 saturates it and its gradient vanishes -- the null would then be an
artifact. A raw multiplicative scale at lambda = 0 has gradient
dL/dlambda = -sum_h sum_t sigmoid(.) dL/dbias, which does NOT vanish with lambda,
so the arm starts EXACTLY at vanilla MapFormer and can leave. This is the
LoopedRefine pattern, where the same construction was verified escapable
(gradient 1.9e-03 at zero) and the optimiser declined anyway.

lambda is unconstrained in sign, so the arm can also discover ANTI-recency
(upweighting distant keys) if that is what the task wants. That would itself be a
result.

PRE-REGISTERED, with a sign. On the torus revisit task the scored event is
retrieval of the FIRST visit to a cell, at a median lag of 33-47 steps with a long
tail. A monotone decay penalises exactly the retrieval being scored, so the
prediction is NEUTRAL-TO-NEGATIVE here, and the informative outcome is what
lambda learns. Positive would be the surprise. The regime the mechanism was built
for is language, which this does not test.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import MapFormerWM, WMTransformerLayer, _apply_rope, USE_SDPA


class ForgetGate(nn.Module):
    """log gamma_t^(h) = -lambda * sigmoid(W_f x_t)^(h);  bias_ts = L_t - L_s."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_heads)
        # start at EXACTLY no decay, with a live gradient -- see the module docstring
        self.lam = nn.Parameter(torch.zeros(1))

    def forward(self, x):                       # x: (B, T, d_model)
        g = torch.sigmoid(self.proj(x))         # (B, T, H) in (0,1)
        log_gamma = -self.lam * g
        L = torch.cumsum(log_gamma, dim=1).transpose(1, 2)          # (B, H, T)
        return L.unsqueeze(-1) - L.unsqueeze(-2)                    # (B, H, T, T)


class WMTransformerLayerForget(WMTransformerLayer):
    """WMTransformerLayer + an additive decay bias on the attention logits."""

    def forward(self, x, cos_a, sin_a, causal_mask, logit_bias=None):
        B, T, _ = x.shape
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q = _apply_rope(Q, cos_a, sin_a)
        K = _apply_rope(K, cos_a, sin_a)

        if logit_bias is None:
            return super().forward(x, cos_a, sin_a, causal_mask)

        if USE_SDPA:
            # SDPA takes an ADDITIVE float mask, so causality and the decay bias
            # travel together and the fast path is kept. is_causal must be False
            # when an explicit mask is supplied, or the two compose wrongly.
            m = logit_bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0),
                                       float("-inf"))
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=m,
                dropout_p=self.dropout.p if self.training else 0.0)
        else:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head)
            scores = scores + logit_bias
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0),
                                        float("-inf"))
            out = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), V)

        out = self.o_proj(out.transpose(1, 2).reshape(B, T, self.d_model))
        x = x + self.dropout(out)
        return x + self.ffn(self.norm2(x))


class MapFormerWM_Forget(MapFormerWM):
    """MapFormer-WM with a Forgetting-Transformer-style decay on the logits."""

    BOTTLENECK_R = 2

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, self.BOTTLENECK_R)
        self.layers = nn.ModuleList([
            WMTransformerLayerForget(d_model, n_heads, dropout)
            for _ in range(n_layers)])
        self.forget = ForgetGate(d_model, n_heads)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.action_to_lie(x))
        bias = self.forget(x)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m, bias)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_Forget_r4(MapFormerWM_Forget):
    BOTTLENECK_R = 4


class MapFormerWM_ForgetFrozen(MapFormerWM_Forget):
    """The control: gate parameters PRESENT, mechanism provably OFF.

    FORGET_GATE.md establishes that the gate is worth +0.086 at r=2 while the
    gain is ANTI-correlated with how much the model forgets (r(lambda, gain) =
    -0.516; five of eight seeds learn lambda < 0). So the named mechanism is
    ruled out and what remains is 259 parameters on the attention path. This
    arm separates those: lambda becomes a BUFFER pinned at zero, so

      - the decay bias is identically 0 and the forward pass is exactly vanilla
        MapFormer;
      - W_f still exists, still consumes the same initialisation draws in the
        same order (lambda is created by torch.zeros and draws no RNG, so
        converting it after construction leaves the stream untouched);
      - W_f receives zero gradient, because the gate is multiplicative in
        lambda -- the parameters are present and inert, which is exactly the
        control.

    If this matches Forget, the gain is parameter count and initialisation shift,
    not the gate. If it matches Vanilla, something about having a live lambda is
    doing the work even though what lambda learns is ~0. This is the control
    V4_MULTISEED.md identified and never ran.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        del self.forget.lam
        self.forget.register_buffer("lam", torch.zeros(1))
