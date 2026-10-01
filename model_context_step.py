"""Context-dependent steps for MapFormer (CONTEXT_STEP_DESIGN.md).

MapFormer's step Delta_t = W_out W_in emb(x_t) depends on the token alone. Suppressing a word's step
from context ("she did not go north") needs a MULTIPLICATIVE dependence on the context; an additive
or linear one cannot cancel a direction-specific step with a direction-independent cue.

CtxGateWM      Delta_t = g_t * W_out W_in emb(x_t), g_t = sigmoid(v . GELU(conv_k(emb))_t), one gate per
               head; conv_k is a causal conv over the last k=4 tokens MIXING channels. Gate bias starts
               at +3 (g ~ 0.95), i.e. at the context-free model.
HiddenStepWM   two layers. Layer 1 is an ordinary index-RoPE layer; the step is computed from its output,
               Delta_t = W_out W_in LN(h1_t); layer 2 uses the path-integrated phase. The per-layer
               placement of Mamba-3 / Selective RoPE.

Both keep cumsum over Delta (the parallel scan is untouched) and depend only on tokens <= t. Both are
built on the r=4 MapFormer base FIRST and add their modules after, so every shared weight equals
Vanilla_r4's (1 layer) or Vanilla_r4 at n_layers=2's at the same seed.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM


class _CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(c_in, c_out, k)

    def forward(self, x):                                  # (B, T, C) -> (B, T, C_out)
        z = nn.functional.pad(x.transpose(1, 2), (self.k - 1, 0))
        return self.conv(z).transpose(1, 2)


class CtxGateWM(MapFormerWM):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64,
                 bottleneck_r=4, k=4, d_gate=32, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 4)
        self.gate_conv = _CausalConv(d_model, d_gate, k)
        self.gate_out = nn.Linear(d_gate, n_heads)
        nn.init.zeros_(self.gate_out.weight); nn.init.constant_(self.gate_out.bias, 3.0)

    def gate(self, x):                                     # (B, T, H)
        return torch.sigmoid(self.gate_out(nn.functional.gelu(self.gate_conv(x))))

    def step(self, tokens):
        x = self.token_emb(tokens)
        return self.action_to_lie(x) * self.gate(x)[..., None], x

    def forward(self, tokens):
        B, L = tokens.shape
        delta, x = self.step(tokens)
        cos_a, sin_a = self.path_integrator(delta)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class HiddenStepWM(MapFormerWM):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=2, dropout=0.1, grid_size=64,
                 bottleneck_r=4, base=10000.0, **kw):
        if n_layers != 2:
            raise ValueError(f"HiddenStepWM is the 2-layer design; got n_layers={n_layers}")
        super().__init__(vocab_size, d_model, n_heads, 2, dropout, grid_size, 4)
        kk = torch.arange(self.n_blocks, dtype=torch.float32)
        self.register_buffer("inv_freq", base ** (-kk / self.n_blocks))
        self.step_norm = nn.LayerNorm(d_model)

    def _index_cos_sin(self, B, L, device, dtype):
        ang = torch.outer(torch.arange(L, device=device, dtype=dtype), self.inv_freq.to(device, dtype))
        c = torch.cos(ang)[None, None].expand(B, self.n_heads, L, -1)
        s = torch.sin(ang)[None, None].expand(B, self.n_heads, L, -1)
        return c, s

    def step(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        c, s = self._index_cos_sin(B, L, tokens.device, x.dtype)
        h1 = self.layers[0](x, c, s, m)
        return self.action_to_lie(self.step_norm(h1)), h1

    def forward(self, tokens):
        B, L = tokens.shape
        delta, h1 = self.step(tokens)
        cos_a, sin_a = self.path_integrator(delta)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        x = self.layers[1](h1, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class HiddenStepResWM(HiddenStepWM):
    """HS with the word's own step kept and context ADDED as a correction (CTXSTEP_HS_RECIPE.md, fix):

        Delta_t = W_out W_in ( emb(x_t) + alpha * LN(h1_t) ),   alpha a learned scalar, initialised at 0.

    At initialisation the step is exactly the context-free step of the embedding (the CF model, which
    reliably learns a movement step); attention can then correct it, e.g. cancel a decoy's step from a
    cue anywhere in context. Built as HiddenStepWM first, so every shared weight equals HS's at the seed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ctx_alpha = nn.Parameter(torch.zeros(1))

    def step(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        c, s = self._index_cos_sin(B, L, tokens.device, x.dtype)
        h1 = self.layers[0](x, c, s, m)
        return self.action_to_lie(x + self.ctx_alpha * self.step_norm(h1)), h1
