"""MapEM-NC: non-commutative MapFormer, per the paper's appendix B.2.2.

The paper specifies (verbatim):

    "any rotation in SO(n) must be expressed with a basis of n(n-1)/2
     skew-symmetric matrices {S_i}_{1<=i<=K}, matrices that do not commute
     together in general"

    R_theta^n = exp( sum_i theta_i S_i )  !=  prod_i exp(theta_i S_i)        (18)

    "non-commutativity implies that path integration cannot be performed via the
     exponential of a sum anymore and can only be achieved via a sequential matrix
     product scaling linearly with sequence length L ... our models cannot
     leverage the parallel processing abilities of Transformers anymore, making
     them analogous to TEM-t"

and gives two variants, because the group manifold is curved:

    "(1) MapEM-NC-L with a linear mapping: Delta_t := W_Delta x_t and
     (2) MapEM-NC-NL that uses an MPL to infer the rotation angles
     Delta_t := f_Delta(x_t)"

WHAT THIS COSTS. Path integration here is a genuine sequential matrix product,
so the O(log T) parallel scan that is MapFormer's efficiency claim is gone. The
paper says so itself. Expect training to be slow and sequences to be short --
they used length 16, testing at 32.

WHAT IT IS FOR. The paper motivates non-commutativity with a family tree
(mother/father do not commute) and then validates on synthetic 4D rotations. This
implementation exists to close that gap: run it on an actual relational hierarchy
(`environment_family_tree.py`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MapFormerEM, EMTransformerLayer


class NCPathIntegrator(nn.Module):
    """Sequential product of exp(sum_i theta_i S_i) over a learned generator basis."""

    def __init__(self, n_heads: int, n_blocks: int, block: int = 4):
        super().__init__()
        self.n_heads, self.n_blocks, self.block = n_heads, n_blocks, block
        self.K = block * (block - 1) // 2          # paper: K = n(n-1)/2
        # raw generators; the skew-symmetric part is what is used, so any
        # symmetric component is inert and cannot break orthogonality
        self.gen = nn.Parameter(torch.randn(self.K, block, block) * 0.05)

    def basis(self):
        return 0.5 * (self.gen - self.gen.transpose(-1, -2))   # skew(A)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """theta: (B, L, n_heads, n_blocks, K) -> cumulative R: (B, L, nh, nb, b, b).

        Sequential by necessity: R_t = R_step(t) @ R_{t-1}. Eq. 18 -- the
        exponential of a sum is NOT the product of exponentials here, so there is
        no scan to exploit.
        """
        B, L, H, NB, K = theta.shape
        S = self.basis()                                        # (K, b, b)
        A = torch.einsum("blhnk,kij->blhnij", theta, S)         # sum_i theta_i S_i
        R = torch.matrix_exp(A.reshape(-1, self.block, self.block)).reshape(
            B, L, H, NB, self.block, self.block)
        out = torch.empty_like(R)
        cur = R[:, 0]
        out[:, 0] = cur
        for t in range(1, L):                                   # the sequential cost
            cur = torch.matmul(R[:, t], cur)
            out[:, t] = cur
        return out


class MapFormerEM_NC(MapFormerEM):
    """MapEM-NC. `nonlinear=False` -> NC-L (linear Delta); True -> NC-NL (MLP)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2,
                 block: int = 4, nonlinear: bool = False):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        del self.action_to_lie, self.path_integrator
        self.block = block
        self.nb = self.d_head // block
        assert self.d_head % block == 0, "d_head must divide into blocks"
        self.pi = NCPathIntegrator(n_heads, self.nb, block)
        out_dim = n_heads * self.nb * self.pi.K
        if nonlinear:                                            # NC-NL
            self.to_theta = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, out_dim))
        else:                                                    # NC-L
            self.to_theta = nn.Linear(d_model, out_dim)
        self.nonlinear = nonlinear
        # one origin per head, shaped for block-wise rotation
        self.p0_nc = nn.Parameter(torch.randn(n_heads, self.nb, block) * 0.02)
        del self.q0_pos, self.k0_pos

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)
        theta = self.to_theta(x).view(B, L, self.n_heads, self.nb, self.pi.K)
        R = self.pi(theta)                                       # (B,L,H,NB,b,b)
        p = self.p0_nc.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1, -1)
        pos = torch.einsum("blhnij,blhnj->blhni", R, p)          # R_t applied to p0
        pos = pos.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        causal = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), 1)
        for layer in self.layers:
            x = layer(x, pos, pos, causal)                       # A_P = P.P^T
        return self.out_proj(self.out_norm(x))


class MapFormerEM_NC_L(MapFormerEM_NC):
    def __init__(self, *a, **kw):
        kw.pop("nonlinear", None); super().__init__(*a, nonlinear=False, **kw)


class MapFormerEM_NC_NL(MapFormerEM_NC):
    def __init__(self, *a, **kw):
        kw.pop("nonlinear", None); super().__init__(*a, nonlinear=True, **kw)
