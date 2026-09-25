"""COUNTER_BATCH.md: the symbol counter installed identically in MapWM, MapEM and TEM, so the only thing left
to learn about position is how each model reaches the k-th offset.

Counter: symbols (token ids 0..n_symbols-1) step +1, every other token 0; angles theta_i = omega_i * count with
FIXED omega_i spaced geometrically from pi/2 to 2*pi/512 (the same frequencies as
`model_tem_recency._install_counter`). 32 blocks at d_head 64.

  WM_Counter   MapWM layer; position = the counter only (no learned increment). The offset must come from
               content phase inside the query-key comparison.
  EM_Counter   single-origin MapEM layer; position = counter + the standard learned rank-4 increment, whose
               output map starts at zero so the model starts as the pure counter. The offset must come from
               query tokens learning a rewind through the bottleneck.
  (TEM: `TEMRecency_Query_CounterInstalled`, offset via a full orthogonal query transform per token.)
"""
import math

import torch
import torch.nn as nn

from mapformer.model import _apply_rope
from mapformer.model_rank import MapFormerWM_r4
from mapformer.model_em_fixed import MapFormerEM_SingleP0_r4


def _omega(nb):
    return (math.pi / 2) * ((2 * math.pi / 512) / (math.pi / 2)) ** (torch.arange(nb, dtype=torch.float32) / (nb - 1))


class _Counter(nn.Module):
    def __init__(self, vocab_size, n_heads, n_blocks, n_symbols=16):
        super().__init__()
        c = torch.zeros(vocab_size); c[:n_symbols] = 1.0
        self.register_buffer("step", c)
        self.register_buffer("omega", _omega(n_blocks))
        self.n_heads, self.n_blocks = n_heads, n_blocks

    def delta(self, tokens):
        B, L = tokens.shape
        return self.step[tokens].view(B, L, 1, 1).expand(B, L, self.n_heads, self.n_blocks)

    def angles(self, delta):
        ang = torch.cumsum(delta, dim=1) * self.omega      # (B, L, H, nb)
        ang = ang.transpose(1, 2)
        return torch.cos(ang), torch.sin(ang)


class MapFormerWM_Counter(MapFormerWM_r4):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size)
        del self.action_to_lie, self.path_integrator
        self.counter = _Counter(vocab_size, n_heads, self.n_blocks)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.counter.angles(self.counter.delta(tokens))
        mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, mask)
        return self.out_proj(self.out_norm(x))


class MapFormerEM_Counter(MapFormerEM_SingleP0_r4):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size)
        del self.path_integrator
        self.counter = _Counter(vocab_size, n_heads, self.n_blocks)
        nn.init.zeros_(self.action_to_lie.w_out.weight)     # starts as the pure counter

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.counter.delta(tokens) + self.action_to_lie(x)
        cos_a, sin_a = self.counter.angles(delta)
        p0 = self.p0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        p = _apply_rope(p0, cos_a, sin_a)
        mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, p, p, mask)
        return self.out_proj(self.out_norm(x))
