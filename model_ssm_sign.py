"""G-3 (G3_SSM_PREREG.md): the sign test beyond attention. A one-layer diagonal complex state-space model whose
transition is an input-dependent rotation (Mamba-3 Sec. 3.2), with the phase step parameterised exactly as in the
attention sign arms.

THE RECURRENCE (per head; N = d_head / 2 complex channels, value width P = d_head):

    H_t = diag(exp(-i omega Delta(x_t))) H_{t-1} + k_t v_t^T        H_t in C^{N x P}
    y_t = Re(conj(q_t)^T H_t)                                        y_t in R^P

with k_t, q_t in C^N read as (even, odd) pairs of a real d_head vector (B_t and C_t of Mamba; Mamba-3 Prop. 2 writes the
same complex SSM as a real one with 2 x 2 rotation blocks), v_t the value ("x_t" in Mamba's notation). The spec's
h_t = exp(i omega Delta(x_t)) h_{t-1} + B x_t is this with omega -> -omega (omega's sign is a free gauge of every arm).
Unrolled, H_t = sum_{s<=t} exp(-i (Phi_t - Phi_s)) k_s v_s^T with Phi_t = omega * cumsum(Delta)_t, so

    y_t = sum_{s<=t} [ q_t . R(Phi_s - Phi_t) k_s ] v_s = sum_{s<=t} [ R(Phi_t) q_t . R(Phi_s) k_s ] v_s,

the "RoPE trick" (Mamba-3 Prop. 3): rotate q and k by their cumulative angle and take a causal, UNNORMALISED (no softmax)
product. `forward` uses that cumsum-of-angles form (exact; quadratic in T, cheap at T <= 1024);
`sequential_reference` runs the recurrence itself in complex128 and `test_ssm_sign.py` checks the two agree (rule 19).

WHAT DIFFERS FROM THE ATTENTION ARMS (model_signroute.ARMS), and nothing else does:
  - softmax(Q K^T / sqrt(d)) with attention dropout -> the causal linear kernel above (no normalisation across positions,
    no competition between keys); the current token is included (s <= t), as in Mamba;
  - y is RMS-normalised per head before o_proj (Mamba-2/3's norm before the output projection; without it y grows with the
    number of past tokens). No output gate (Mamba's silu(z)); declared.
  Embedding, q/k/v/o shapes, FFN, LayerNorms, out_proj, ActionToLie (the step map) and PathIntegrator (omega, geometric
  init 2 pi .. 2 pi / grid_size) are the attention model's, so parameter counts differ by the RMSNorm scale only (128).

STEP CLASSES (`mode`; the readout code is model_signroute.SignRouteActionToLie, i.e. the attention arms' own code):
  signed    Delta = W_out W_in e           anchor
  abs       |W_out W_in e|                 the non-negative class of the attention sign line
  softplus  softplus(W_out W_in e)         Mamba's Delta parameterisation (Delta_t > 0)
  exp       exp(W_out W_in e)              LRU-style positive phase (exp of an unconstrained parameter)
  index     every token steps 1            input-independent rotation (an LTI rotary SSM, S4D/LRU-like): integration gate
`steps`:
  moves     only the move tokens step; every other token's Delta is 0 BY CONSTRUCTION (the identity transition). The
            attention ring anchors (LRSWEEP, SIGNROUTE incl. its exp/softplus breadth arms) are of this kind.
  all       every token steps (Mamba-3-faithful); observations must learn Delta = 0 (signed, abs) or a full turn
            (softplus, exp: Prop. 3(f), (g)). Observation tokens are then "stay" inputs.
`mixer`: "ssm" (above) or "none" (the mixer output is 0: a memoryless model, a gate arm).
`decay`: "none" (registered: |a| = 1, see G3_SSM_PREREG.md Sec. 2 for why) or "learned" (Mamba-2/3 style data-dependent
  a_t = exp(-softplus(w_a . x_t + b_a)) per head, b_a initialised for a = 0.999; NOT in the registered batch).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import ActionToLieAlgebra, PathIntegrator, _apply_rope
from mapformer.model_signroute import SignRouteActionToLie

MODES = ("signed", "abs", "softplus", "exp", "index")
STEPS = ("moves", "all")


class RMSNormHeads(nn.Module):
    """RMSNorm over each head's P channels, one learned scale per model channel (Mamba-2 grouped norm)."""

    def __init__(self, d_model, n_heads, eps=1e-5):
        super().__init__()
        self.n_heads, self.eps = n_heads, eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, y):                                   # y: (B, H, T, P)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        B, H, T, P = y.shape
        return y.transpose(1, 2).reshape(B, T, H * P) * self.weight


class SSMLayer(nn.Module):
    """One pre-norm block: rotary diagonal complex SSM mixer + the attention model's FFN."""

    def __init__(self, d_model, n_heads, dropout, mixer="ssm", decay="none"):
        super().__init__()
        assert mixer in ("ssm", "none") and decay in ("none", "learned")
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.mixer, self.decay = mixer, decay
        self.q_proj = nn.Linear(d_model, d_model)            # C_t
        self.k_proj = nn.Linear(d_model, d_model)            # B_t
        self.v_proj = nn.Linear(d_model, d_model)            # x_t (Mamba's value)
        self.o_proj = nn.Linear(d_model, d_model)
        self.y_norm = RMSNormHeads(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        if decay == "learned":
            self.a_proj = nn.Linear(d_model, n_heads)
            nn.init.zeros_(self.a_proj.weight)
            with torch.no_grad():                            # softplus(b) = -ln(0.999)
                self.a_proj.bias.fill_(math.log(math.expm1(-math.log(0.999))))

    def qkv(self, x):
        B, T, _ = x.shape
        h = self.norm1(x)
        sh = lambda z: z.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        loga = None
        if self.decay == "learned":
            loga = -F.softplus(self.a_proj(h)).transpose(1, 2)        # (B, H, T), log a_t <= 0
        return sh(self.q_proj(h)), sh(self.k_proj(h)), sh(self.v_proj(h)), loga

    @staticmethod
    def mix(Q, K, V, cos_a, sin_a, loga=None):
        """Parallel (cumsum-of-angles) form of the recurrence. Q, K, V (B, H, T, dh); cos/sin (B, H, T, N)."""
        T = Q.shape[2]
        Qr, Kr = _apply_rope(Q, cos_a, sin_a), _apply_rope(K, cos_a, sin_a)
        S = torch.matmul(Qr, Kr.transpose(-1, -2))                     # (B, H, T, T)
        keep = torch.ones(T, T, device=Q.device, dtype=torch.bool).tril()
        if loga is not None:                                           # exp(L_t - L_s), L = cumsum log a
            L = torch.cumsum(loga, -1)
            S = S * torch.exp((L.unsqueeze(-1) - L.unsqueeze(-2)).masked_fill(~keep, 0.0))
        S = S.masked_fill(~keep, 0.0)
        return torch.matmul(S, V)                                      # (B, H, T, dh)

    def forward(self, x, cos_a, sin_a):
        if self.mixer == "none":
            out = torch.zeros_like(x)
        else:
            Q, K, V, loga = self.qkv(x)
            out = self.o_proj(self.y_norm(self.mix(Q, K, V, cos_a, sin_a, loga)))
        x = x + self.dropout(out)
        return x + self.ffn(self.norm2(x))


class SignSSM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=32,
                 bottleneck_r=4, mode="signed", steps="moves", action_ids=(0, 1), mixer="ssm", decay="none"):
        super().__init__()
        assert mode in MODES and steps in STEPS, (mode, steps)
        assert n_layers == 1, "G-3 is a one-layer model"
        self.mode, self.steps, self.mixer_kind, self.decay = mode, steps, mixer, decay
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.vocab_size, self.grid_size = vocab_size, grid_size
        # model_signroute's arms build MapFormerWM (whose plain ActionToLieAlgebra draws its weights second) and then
        # REPLACE action_to_lie with a SignRouteActionToLie drawn last; the same two draws are made here, so every
        # tensor, the step map included, starts from the attention arm's value at a seed (test_ssm_sign).
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, bottleneck_r)
        self.path_integrator = PathIntegrator(n_heads, self.n_blocks, grid_size)
        self.layers = nn.ModuleList([SSMLayer(d_model, n_heads, dropout, mixer, decay)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.action_to_lie = SignRouteActionToLie(d_model, n_heads, self.n_blocks, bottleneck_r,
                                                  mode="signed" if mode == "index" else mode)
        self.register_buffer("act", torch.tensor(action_ids, dtype=torch.long), persistent=False)

    # ---------------------------------------------------------------- step map
    def step_table(self):
        """(V, H, N) per-token step Delta before omega."""
        e = self.token_emb.weight
        V = e.shape[0]
        if self.mode == "index":
            return torch.ones(V, self.n_heads, self.n_blocks, device=e.device, dtype=e.dtype)
        a2l = self.action_to_lie
        if self.steps == "all":
            return a2l.from_latent(a2l.w_in(e)).view(V, self.n_heads, self.n_blocks)
        v = a2l.from_latent(a2l.w_in(e[self.act])).view(len(self.act), self.n_heads, self.n_blocks)
        out = torch.zeros(V, self.n_heads, self.n_blocks, device=e.device, dtype=v.dtype)
        out[self.act] = v
        return out

    def angle_table(self):
        """(V, H, N) per-token angle omega * Delta (what the model reads, mod 2 pi, through cos / sin)."""
        return self.step_table() * self.path_integrator.omega

    def latents(self):
        return self.action_to_lie.w_in(self.token_emb.weight[self.act])

    def clock(self):
        """(H, N) omega * (Delta(+1) + Delta(-1)) in angle units: 0 (mod 2 pi) on a channel = the pair cancels."""
        st = self.step_table()[self.act]
        return self.path_integrator.omega * st.sum(0)

    def forward(self, tokens):
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.step_table()[tokens])
        x = self.layers[0](x, cos_a, sin_a)
        return self.out_proj(self.out_norm(x))

    # ---------------------------------------------------------------- reference recurrence (rule 19)
    @torch.no_grad()
    def sequential_reference(self, tokens):
        """The recurrence H_t = a_t diag(exp(-i omega Delta_t)) H_{t-1} + k_t v_t^T, y_t = Re(conj(q_t)^T H_t), run step
        by step in complex128 from the model's own q, k, v, steps and omega (eval mode; dropout off). The linear layers
        run in the model's dtype: call it on model.double() for an exact comparison, on the float32 model for a
        tolerance one."""
        was = self.training
        self.eval()
        lay = self.layers[0]
        dt = self.token_emb.weight.dtype
        x = self.token_emb(tokens)
        B, T, _ = x.shape
        if lay.mixer == "none":
            y = torch.zeros_like(x)
        else:
            Q, K, V, loga = lay.qkv(x)
            dphi = (self.step_table()[tokens] * self.path_integrator.omega).double().permute(0, 2, 1, 3)  # (B,H,T,N)
            Q, K, V = Q.double(), K.double(), V.double()
            qc = torch.complex(Q[..., 0::2], Q[..., 1::2])                                         # (B, H, T, N)
            kc = torch.complex(K[..., 0::2], K[..., 1::2])
            Hs = torch.zeros(B, self.n_heads, self.n_blocks, self.d_head, dtype=torch.complex128,
                             device=tokens.device)
            ys = []
            for t in range(T):
                rot = torch.exp(-1j * dphi[:, :, t])                                               # (B, H, N)
                if loga is not None:
                    rot = rot * torch.exp(loga[:, :, t].double()).unsqueeze(-1)
                Hs = rot.unsqueeze(-1) * Hs + kc[:, :, t].unsqueeze(-1) * V[:, :, t].unsqueeze(-2)
                ys.append(torch.einsum("bhn,bhnp->bhp", qc[:, :, t].conj(), Hs).real)
            y = lay.o_proj(lay.y_norm(torch.stack(ys, 2).to(dt)))                                  # (B, T, d)
        x = x + y
        x = x + lay.ffn(lay.norm2(x))
        out = self.out_proj(self.out_norm(x))
        self.train(was)
        return out


def build(mode, steps, vocab_size, grid_size, action_ids=(0, 1), mixer="ssm", decay="none", **kw):
    return SignSSM(vocab_size, grid_size=grid_size, mode=mode, steps=steps, action_ids=action_ids, mixer=mixer,
                   decay=decay, **kw)
