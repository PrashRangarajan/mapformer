"""MapFormer-EM with PER-PAIR position origins -- the decisive test of the kernel-sharing claim.

`EM_WM_THEORY.md` P2. MapEM's position kernel is built from two LEARNED VECTORS, so it is one
kernel for every query-key pair; MapWM's is built from content-derived Q,K, so its amplitudes
and phases vary per pair (measured: phase spread across pairs 1.947 for WM, 0.000 for both EM
forms). Every account of EM's recency deficit that appeals to kernel SHARING predicts that
giving EM per-pair origins -- while keeping the Hadamard composition `softmax(A_X (*) A_P)` --
recovers WM's performance. The per-token-search account predicts it does not, because the model
still has to serve 64 query tokens, each seeing 1/64 of the queries.

This interpolates EM -> WM on the sharing axis ALONE. Composition, rank, depth and parameter
count are otherwise unchanged:

    q^p_t = p0 + W^q_out W^q_in x_t          k^p_s = p0 + W^k_out W^k_in x_s
    A_P[t,s] = (R_{theta_t} q^p_t) . (R_{theta_s} k^p_s) / sqrt(d_head)

`W_out` is ZERO-initialised on both sides, so at step 0 this is EXACTLY
`VanillaEM_P0_r4` -- same function, same RNG draws for every other parameter. That is checked
by `ckpt_guard.assert_same_function_at_init` in the batch, not assumed here: any difference
after training is what per-pair freedom buys, not what a different initialisation buys.

Cost: 2 * (d_model * r + r * n_heads * d_head) = 2,048 parameters at d=128, r=4, i.e. +0.9%.
Stated rather than hidden; it is far below anything measurable on this task, and the
alternative (a full per-token origin projection) would add 32k.
"""
import torch
import torch.nn as nn

from mapformer.model import _apply_rope
from mapformer.model_em_fixed import MapFormerEM_SingleP0_r4


class _PairOrigin(MapFormerEM_SingleP0_r4):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 grid_size=64, bottleneck_r=2, origin_r=4, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, bottleneck_r)
        H, dh = self.n_heads, self.d_head
        self.origin_r = origin_r
        self.q_origin_in = nn.Linear(d_model, origin_r, bias=False)
        self.q_origin_out = nn.Linear(origin_r, H * dh, bias=False)
        self.k_origin_in = nn.Linear(d_model, origin_r, bias=False)
        self.k_origin_out = nn.Linear(origin_r, H * dh, bias=False)
        with torch.no_grad():                      # zero OUT -> identical to single-p0 at init
            self.q_origin_out.weight.zero_()
            self.k_origin_out.weight.zero_()
        assert hasattr(self, "p0_pos") and not hasattr(self, "q0_pos"), "base origin not shared"

    def _origins(self, x):
        """(B, H, L, d_head) per-token query and key origins."""
        B, L, _ = x.shape
        H, dh = self.n_heads, self.d_head
        p0 = self.p0_pos.unsqueeze(0).unsqueeze(2)                       # (1, H, 1, dh)
        dq = self.q_origin_out(self.q_origin_in(x)).view(B, L, H, dh).transpose(1, 2)
        dk = self.k_origin_out(self.k_origin_in(x)).view(B, L, H, dh).transpose(1, 2)
        return p0 + dq, p0 + dk

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.action_to_lie(x))
        q0, k0 = self._origins(x)
        q_pos = _apply_rope(q0, cos_a, sin_a)
        k_pos = _apply_rope(k0, cos_a, sin_a)
        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, q_pos, k_pos, causal_mask)
        return self.out_proj(self.out_norm(x))


def _mk(name, origin_r):
    class _P(_PairOrigin):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                     grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size,
                             bottleneck_r, origin_r=origin_r)
    _P.__name__ = name
    _P.__doc__ = _PairOrigin.__doc__
    return _P


MapFormerEM_PairOrigin_r4 = _mk("MapFormerEM_PairOrigin_r4", 4)
