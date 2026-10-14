"""Published monotone content-dependent position methods, faithful to their papers, for GENERALITY_PREREG.md.

CARoPEF (Veisi et al., CARoPE, `papers/txt/carope.txt` sec. 2): per head h and token t a scalar rate
    f_h(x_t) = 1 / (softplus(x_t W)_h + 1)  in (0, 1),  W in R^{d x H},
replaces RoPE's base frequency theta_1, so channel i (i = 0 .. d_head/2 - 1) advances by f_h(x_t)^i per token and the
phase is phi_i^h(m) = sum_{t <= m} f_h(x_t)^i. "Initialised using the standard RoPE formulation": W = 0 and a per-head bias b
with 1 / (softplus(b) + 1) = 10000^(-2 / d_head), so at init the model IS index RoPE (base 10000). The bias is ours (the
paper's W has none; without it the init is RoPE with base ~2e7). Every token advances the phase (no observation-free
variant: the method has none). Steps lie in (0, 1] rad per token, so two moves can never cancel mod 2 pi (their angles sum
to at most 2 < 2 pi): it CANNOT represent a ring / torus map (Case II without the wrap). Same layer as MapWM
(model.WMTransformerLayer, score Q^T R(dtheta) K), so only the phase differs.

CoPEF (Golovneva et al. 2024, `papers/txt/cope.txt` App. B, Listing 1, verbatim in torch): per layer one CoPE module shared
by the heads; gates g_ij = sigmoid(q_i.k_j / sqrt(d_h)) (masked keys -> 0), p_ij = sum_{k=j}^{i} g_ik (flip-cumsum-flip),
clamped to npos_max - 1, position logit = linear interpolation of q_i . e[floor p], q_i . e[ceil p] with e = pos_emb
(1, d_h, npos_max), zeros at init; added to the attention logits. No rotary position. Pre-LN block as the repo's
CoPEBaseline (model_baselines_extra.py), whose attention materialises (B, H, T, T, d_h) and uses p = sum_{k=j+1}^{i} with
gates on unmasked logits -- not the paper's; this module replaces it for the new batch. npos_max = 2048 (the torus context;
on the ring, 255 tokens, positions above 254 are never indexed and their table entries never move). NOTE (GENERALITY audit
finding 1): CoPE CAN represent ring cancellation -- gates 17/16 and 15/16 per (move, observation) pair make p_ij mod 2 a
function of net displacement, read by a period-2 position table (1.000 revisit accuracy, constructed).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import MapFormerWM


class MapFormerWM_CARoPEF(MapFormerWM):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        del self.action_to_lie, self.path_integrator                                     # replaced by the CARoPE phase
        d_head = d_model // n_heads; self.nb = d_head // 2
        self.rate = nn.Linear(d_model, n_heads, bias=True)
        theta1 = 10000.0 ** (-2.0 / d_head)
        with torch.no_grad():
            self.rate.weight.zero_()
            self.rate.bias.fill_(math.log(math.expm1(1.0 / theta1 - 1.0)))              # softplus(b) = 1/theta1 - 1
        self.register_buffer("expo", torch.arange(self.nb, dtype=torch.float32), persistent=False)

    def phase(self, tokens):
        f = 1.0 / (F.softplus(self.rate(self.token_emb(tokens))) + 1.0)                 # (B, L, H) in (0, 1)
        step = f[..., None] ** self.expo                                                 # (B, L, H, nb): f^i
        ang = step.cumsum(1).transpose(1, 2)                                             # (B, H, L, nb)
        return torch.cos(ang), torch.sin(ang)

    def get_position_state(self, tokens):
        return self.phase(tokens)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.phase(tokens)
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        return self.out_proj(self.out_norm(x))


class CoPE(nn.Module):
    """Listing 1 of the CoPE paper (heads folded into the batch dimension there; here pos_emb is shared across heads)."""

    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)


class CoPEAttn(nn.Module):
    def __init__(self, d_model, n_heads, npos_max, dropout=0.1):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.cope = CoPE(npos_max, self.dh)

    def forward(self, x, mask_log):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)        # (B, H, L, dh) each
        attn_logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + mask_log
        attn_logits = attn_logits + self.cope(q, attn_logits)
        attn = self.dropout(torch.softmax(attn_logits, dim=-1))
        return self.o_proj((attn @ v).transpose(1, 2).reshape(B, L, D))


class CoPEFBlock(nn.Module):
    def __init__(self, d_model, n_heads, npos_max, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CoPEAttn(d_model, n_heads, npos_max, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask_log):
        x = x + self.dropout(self.attn(self.norm1(x), mask_log))
        return x + self.ffn(self.norm2(x))


class CoPEF(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, npos_max=2048, **kw):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([CoPEFBlock(d_model, n_heads, npos_max, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, L = tokens.shape
        mask = torch.ones(L, L, device=tokens.device).tril()
        mask_log = mask.log()                                                            # 0 / -inf, as Listing 1
        x = self.token_emb(tokens)
        for layer in self.layers:
            x = layer(x, mask_log)
        return self.out_proj(self.out_norm(x))
