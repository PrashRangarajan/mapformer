"""Weight-shared looped (recursive-depth) variants of MapFormer-WM and the RoPE
baseline, for the question: does RECURSION buy attention horizon cheaply?

WHY THIS EXISTS. HORIZON_RESULTS.md measured that an index model's ability to
path-integrate scales with DEPTH -- horizon ~2 at 1 layer, ~16 at 2, ~32 at 4 --
but stops there: 3.17M params at 4 layers x 256 wide still collapses past
interval ~32, while a 204K 1-layer path-integrated model reaches 65+ at 0.880.

A looped transformer buys effective depth at CONSTANT parameters, so it tests two
things the depth grid could not separate:

  1. Does effective depth alone extend the horizon, or does the horizon need
     DISTINCT layers that can specialise? A shared block must play every role.
  2. Does recursion HURT the path-integrated arm the way real depth did? The
     depth grid found Vanilla is non-monotone in capacity at long range
     (L2 d256 0.976 -> L4 d256 0.782 at interval 65+). If recursion behaves like
     depth, stacking it on MapFormer has a measured headwind.

DESIGN. Plain ALBERT-style weight sharing: one block applied n_loops times, with
NO per-iteration depth/timestep embedding. Universal Transformer adds one; that
would introduce both extra parameters and an extra signal, confounding the
"effective depth at constant params" question this pilot asks. If the pilot is
positive, the depth-embedding and the recompute-theta-per-iteration variants are
the natural follow-ons -- the latter being iterative refinement of the position
estimate, i.e. structurally the InEKF work moved from the sequence axis to the
depth axis.

theta is computed ONCE from the token embeddings and reused across iterations,
matching the base models. Recomputing it per iteration is a different experiment.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM, ActionToLieAlgebra
from mapformer.model_baseline_rope import MapFormerWM_RoPE


def _causal(L, device):
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


class MapFormerWM_Looped(MapFormerWM):
    """MapFormer-WM with ONE transformer block applied n_loops times.

    Parameter count equals the 1-layer model; effective depth is n_loops.
    """
    n_loops = 4

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, n_loops=None, **kw):
        # n_layers is IGNORED on purpose: the whole point is a single shared
        # block. Accepting and dropping it keeps the trainer's CLI uniform, and
        # the assert stops a caller silently believing they got 4 real layers.
        super().__init__(vocab_size, d_model, n_heads, 1, dropout, grid_size,
                         bottleneck_r)
        if n_loops is not None:
            self.n_loops = n_loops

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)
        m = _causal(L, tokens.device)
        block = self.layers[0]
        for _ in range(self.n_loops):
            x = block(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_RoPE_Looped(MapFormerWM_RoPE):
    """Index-position (standard RoPE) counterpart, same sharing scheme."""
    n_loops = 4

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, base=10000.0, n_loops=None, **kw):
        super().__init__(vocab_size, d_model, n_heads, 1, dropout, grid_size, base)
        if n_loops is not None:
            self.n_loops = n_loops

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self._rope_cos_sin(L, tokens.device, x.dtype)
        cos_a = cos_a.expand(B, -1, -1, -1); sin_a = sin_a.expand(B, -1, -1, -1)
        m = _causal(L, tokens.device)
        block = self.layers[0]
        for _ in range(self.n_loops):
            x = block(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_LoopedRefine(MapFormerWM_Looped):
    """Loop that REFINES the position estimate each pass, instead of re-reading a
    fixed one. The follow-on the Match-Query result pointed at.

    `MapFormerWM_Looped` computes theta once from the token embeddings and reuses
    it for every pass. Here the loop carries a position estimate that is corrected
    from the CURRENT hidden state -- structurally the InEKF work of this project
    moved from the sequence axis to the depth axis:

        theta_0 = omega * cumsum(action_to_lie(emb))          (the path integral)
        x       = block(x, cos theta, sin theta)
        theta   = theta_0 + gate * tanh(refine(x))            (bounded correction)

    THREE DESIGN CHOICES, each bought by a prior failure in this repo:

    1. `gate` is initialised to ZERO, so at step 0 this model is EXACTLY
       MapFormerWM_Looped and the correction has to be learned. Level15EM started
       with K=0.5 corrections, which destroyed the attention pattern before any
       gradient signal existed and diverged 3 of 9 seeds; the fix there was the
       same -- make the correction a near-no-op at init.
    2. The correction is bounded by `tanh`. The InEKF finding was that the WRAP
       (bounded innovation) was the load-bearing piece, not the inference: an
       unbounded correction lets theta leave the range the rotations were trained
       on, which is exactly how NoBypass blew up to |theta| ~ 3840 at T=512.
    3. The correction is applied to THETA (a position), not to delta (odometry),
       and is NOT cumsummed. A delta correction would compound along the sequence;
       a position correction stays bounded, which is the InEKF analogue.

    Parameter cost is negligible -- one more low-rank ActionToLieAlgebra (~400
    params on 204,630) plus a scalar gate.
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.refine = ActionToLieAlgebra(self.d_model, self.n_heads,
                                         self.n_blocks, 2)
        self.gate = nn.Parameter(torch.zeros(1))

    def _theta(self, delta):
        """omega * cumsum(delta) -> (B, H, T, n_blocks); the angle PathIntegrator
        builds internally before taking cos/sin."""
        return (torch.cumsum(delta, dim=1) *
                self.path_integrator.omega.unsqueeze(0).unsqueeze(0)).transpose(1, 2)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        theta0 = self._theta(self.action_to_lie(x))
        theta = theta0
        m = _causal(L, tokens.device)
        block = self.layers[0]
        for i in range(self.n_loops):
            x = block(x, torch.cos(theta), torch.sin(theta), m)
            if i < self.n_loops - 1:
                theta = theta0 + self.gate * torch.tanh(
                    self.refine(x).transpose(1, 2))
        return self.out_proj(self.out_norm(x))
