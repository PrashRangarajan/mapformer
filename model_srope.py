"""Faithful Selective RoPE (Movahedi et al., ICLR 2026, arXiv:2511.17388) in MapFormer's scaffold (SROPE_TORUS_PREREG.md).

SOURCE. The paper's Fig. 4 pseudocode (omega = W_omega @ q; conv1d; temp * cumsum; sincos; rope), its Sec. 4.1 implementation
paragraph (learnable omega, sigmoid gate on the rotation angles, learnable bias, weight norm on the input projection), Table 2
(l2-normalised input x instead of q is more stable; + SiLU after the conv and + phase gate give the gains), and -- because the
paper's Appendix B "code listing" is absent from the text we hold -- the authors' released code, github.com/timurcarstensen/
selective-rope at d12e825 (2026-03-01): src/selective_rope/modules/rotary.py::SelectiveRoPE (read in full), its configs, and
the two flash-linear-attention primitives it calls (fla rev 64248fc: ShortConvolution = depthwise causal Conv1d + optional
SiLU; l2_norm = x / sqrt(sum x^2 + 1e-6)). Their best language-model configuration (configs/language_modeling/
language_modeling.yaml) is: phi_parametrization full-rank, conv (d_conv 4) with conv_act silu, use_gate true with
gate_parametrization standard, use_bias false, temp_type rope, temp_theta 500000, temp_max 1.0, temp_grad false.

WHAT THE OFFICIAL FORWARD DOES (non-Triton path, per attention layer; reproduced here line for line):
    xh    = l2_norm(hidden_states viewed (B, T, H, d_model / H))        # the layer's NORMED input, per-head chunk
    phi   = ChunkedLinear(xh)                                          # per head: (d_model/H) -> pe_dim, weight norm:
                                                                       #   W_h = g_h * W_h / ||W_h||_col, g_h init 1
    phi   = ShortConvolution(phi)  [+ SiLU]                            # depthwise causal conv over positions, kernel 4
    phi  += exp(phi_bias)                     (use_bias only)
    phi  *= sigmoid(Linear(d_model -> H)(xh flattened))[..., None]     # one gate per head and position (standard)
    angle = temperature * cumsum(phi, dim=T)                           # temperature (pe_dim,) fixed: 500000^(-2c/d_head)
    q, k  = rotate(q, k, angle)                                        # pe_dim = d_head / 2 rotation pairs per head
Init as their HF model applies it (_init_weights): every nn.Linear / nn.Conv1d ~ N(0, 0.02), biases 0 (so the conv taps and the
gate start near 0 / 0.5); ChunkedLinear keeps kaiming_uniform (a = 0) per head and g = 1 (it has no reset_parameters).

DEVIATIONS (each one stated in the prereg):
  * rotation pairs: MapFormer's _apply_rope pairs (2c, 2c+1); theirs (GPT-NeoX) pairs (c, c + pe_dim). The paper calls the two
    "equivalent up to an index permutation"; with q_proj / k_proj initialised iid per output row the permutation is invisible
    to the function class and the init distribution. Temperature c goes to pair c in both.
  * scaffold: MapFormer's block (LayerNorm pre-norm, GELU FFN, dropout 0.1, learned token embedding) instead of their
    RMSNorm / SwiGLU HF block; the generator reads the block's norm1 output, as theirs reads attn_norm's.
  * optimiser: our train() puts every parameter in one AdamW group (weight decay 0.05); theirs exempts g (_no_weight_decay).
  * no Triton kernel (the PyTorch path of the same module).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import MapFormerWM


def rotary_temperature(temp_type, theta, head_dim, temp_max):
    """Their rotary.py::rotary_temperature, verbatim in effect; returns (head_dim // 2,)."""
    if temp_type == "rfa":
        assert 0.0 < theta < 1.0
        th = torch.linspace(0.0, float(theta * math.pi), steps=head_dim // 2, dtype=torch.float32)
        t = torch.sqrt((1 - torch.cos(th)) / (1 + torch.cos(th)))
        return temp_max * t / torch.max(t)
    assert temp_type == "rope" and theta >= 1.0
    return temp_max / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def l2_norm(x, eps=1e-6):
    """fla.modules.l2norm.l2_norm: x / sqrt(sum(x^2) + eps) over the last dim (fla rev 64248fc)."""
    return x * torch.rsqrt((x.float() * x.float()).sum(-1, keepdim=True) + eps).to(x.dtype)


class SelectiveRoPEAngle(nn.Module):
    """The angle generator of their SelectiveRoPE module. forward(h) -> angle (B, T, H, pe_dim); steps(h) -> the per-position
    increment BEFORE the temperature and cumsum (B, T, H, pe_dim)."""

    def __init__(self, d_model, n_heads, d_head, d_conv=4, conv_act="silu", use_gate=True, use_bias=False,
                 temp_type="rope", temp_theta=500000.0, temp_max=1.0, phi_parametrization="full-rank", phi_proj_rank=32,
                 skip_conv=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.H, self.pe = n_heads, d_head // 2
        self.d_in = d_model // n_heads
        self.conv_act, self.use_gate, self.use_bias, self.skip_conv = conv_act, use_gate, use_bias, skip_conv
        self.phi_parametrization = phi_parametrization
        C = n_heads * self.pe
        if phi_parametrization == "full-rank":            # ChunkedLinear: per head, weight-normed, g init 1
            self.weight = nn.Parameter(torch.empty(n_heads, self.d_in, self.pe))
            for i in range(n_heads):
                nn.init.kaiming_uniform_(self.weight.data[i], a=0.0)
            self.scalar = nn.Parameter(torch.ones(n_heads))
        else:                                             # their low-rank option (shared across heads)
            self.phi_proj = nn.Sequential(nn.Linear(d_model, phi_proj_rank, bias=False),
                                          nn.Linear(phi_proj_rank, C, bias=False))
            for mod in self.phi_proj:
                nn.init.normal_(mod.weight, 0.0, 0.02)
        if not skip_conv:
            self.conv = nn.Conv1d(C, C, d_conv, groups=C, bias=False, padding=d_conv - 1)
            nn.init.normal_(self.conv.weight, 0.0, 0.02)  # their _init_weights on nn.Conv1d
        if use_bias:
            self.phi_bias = nn.Parameter(torch.zeros(1, 1, n_heads, self.pe))
        if use_gate:
            self.gate = nn.Linear(d_model, n_heads, bias=True)
            nn.init.normal_(self.gate.weight, 0.0, 0.02); nn.init.zeros_(self.gate.bias)
        self.register_buffer("temperature", rotary_temperature(temp_type, temp_theta, d_head, temp_max))  # temp_grad false

    def steps(self, h):
        B, T, _ = h.shape
        xh = l2_norm(h.view(B, T, self.H, self.d_in))
        if self.phi_parametrization == "full-rank":
            W = self.scalar.view(-1, 1, 1) * F.normalize(self.weight, dim=-2)
            phi = torch.einsum("bthi,hio->btho", xh, W).reshape(B, T, self.H * self.pe)
        else:
            phi = self.phi_proj(xh.reshape(B, T, -1))
        if not self.skip_conv:
            phi = self.conv(phi.transpose(1, 2))[..., :T].transpose(1, 2)
            if self.conv_act is not None:
                phi = F.silu(phi)
        else:                                             # their skip_conv branch: first difference in time
            phi = phi - torch.cat([torch.zeros_like(phi[:, :1]), phi[:, :-1]], dim=1)
        phi = phi.reshape(B, T, self.H, self.pe)
        if self.use_bias:
            phi = phi + torch.exp(self.phi_bias)
        if self.use_gate:
            phi = phi * torch.sigmoid(self.gate(xh.reshape(B, T, -1))).unsqueeze(-1)
        return phi

    def forward(self, h):
        return self.temperature * torch.cumsum(self.steps(h), dim=1)


class MapFormerWM_SRoPE(MapFormerWM):
    """MapFormer-WM's scaffold with MapFormer's path integrator replaced by faithful Selective RoPE (their best LM config), one
    generator per layer reading that layer's norm1 output. Everything else (embedding, blocks, readout) is MapFormerWM's, built
    first so the scaffold's initial weights equal MapFormerWM's at the same seed."""
    CFG = dict(conv_act="silu", use_gate=True, use_bias=False, temp_type="rope", temp_theta=500000.0, temp_max=1.0,
               phi_parametrization="full-rank")

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        del self.action_to_lie, self.path_integrator
        self.srope = nn.ModuleList([SelectiveRoPEAngle(d_model, n_heads, self.d_head, **self.CFG) for _ in range(n_layers)])

    def angles(self, tokens):
        """Per-layer angle tensors (B, T, H, pe_dim) for the given tokens (analysis helper; same ops as forward)."""
        x = self.token_emb(tokens); B, L = tokens.shape; out = []
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer, gen in zip(self.layers, self.srope):
            th = gen(layer.norm1(x)); out.append(th)
            x = layer(x, torch.cos(th.transpose(1, 2)), torch.sin(th.transpose(1, 2)), m)
        return out

    def get_position_state(self, tokens):
        th = self.srope[0](self.layers[0].norm1(self.token_emb(tokens))).transpose(1, 2)
        return torch.cos(th), torch.sin(th)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer, gen in zip(self.layers, self.srope):
            th = gen(layer.norm1(x)).transpose(1, 2)      # (B, H, T, pe_dim)
            x = layer(x, torch.cos(th), torch.sin(th), m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_SRoPE_NoSiLU(MapFormerWM_SRoPE):
    """The same, conv_act None (a single-knob arm; Table 2's rows are ambiguous about whether this exact combination was run): the step can be negative as freely as positive."""
    CFG = dict(MapFormerWM_SRoPE.CFG, conv_act=None)


class MapFormerWM_SRoPE_ST(MapFormerWM_SRoPE):
    """The authors' state-tracking configuration (configs/state_tracking/state_tracking.yaml over the base
    selective_rope.yaml; the setting of their 1-layer softmax-Transformer parity result, Fig. 7): low-rank 32 projection
    shared across heads, conv WITHOUT SiLU, no gate, learnable bias exp(phi_bias) (+1 rad per token per channel at init),
    temperature rope-type with theta 1.0 (every channel's temperature = 1.0). SROPE_TORUS Amendment 1."""
    CFG = dict(conv_act=None, use_gate=False, use_bias=True, temp_type="rope", temp_theta=1.0, temp_max=1.0,
               phi_parametrization="low-rank", phi_proj_rank=32)
