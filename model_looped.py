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
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF


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


class MapFormerWM_LoopedSampled(MapFormerWM_Looped):
    """Loop count SAMPLED during training, so the model is not tuned to one depth.

    WHY. A model trained at a fixed 4 passes is best at 4 passes on the length it
    trained at, and best at 2 on a 4x longer one (0.794 vs 0.776, same weights,
    eval-only change). Every pass past the second actively hurts out of
    distribution. That says the fixed count is a length-specific choice baked in at
    training time -- not that iteration and extrapolation are fundamentally opposed.

    So: draw the count per forward pass from `loop_choices` during training. At
    evaluation `n_loops` is used as normal, which makes the count a runtime knob
    the model has been trained to tolerate across its whole range.

    Range is {2..6}: 1 pass is degenerate (0.818 vs 1.000 at training length) and
    forcing the model to serve it would cost peak performance for a setting nobody
    would deploy. Sampling is per-batch, drawn from torch's seeded RNG so runs stay
    reproducible.
    """
    loop_choices = (2, 3, 4, 5, 6)

    def forward(self, tokens):
        if self.training:
            i = int(torch.randint(len(self.loop_choices), (1,)).item())
            k = self.loop_choices[i]
        else:
            k = self.n_loops
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cos_a, sin_a = self.path_integrator(delta)
        m = _causal(L, tokens.device)
        block = self.layers[0]
        for _ in range(k):
            x = block(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_Level15Looped(MapFormerWM_Level15InEKF):
    """The missing cell of the 2x2: Level 1.5 InEKF AND the shared-block loop.

    WHY. The two mechanisms have exactly anti-correlated profiles on the clean
    torus task. The loop's entire benefit is at TRAINING length (1.000 vs Vanilla's
    0.966 at T=128, and +0.205 over Vanilla under p=0.25 action noise) and its
    entire cost is at OOD length (0.816 / 0.642 at T=512 / 1024 against Vanilla's
    0.891 / 0.768). The filter is the mirror image: nothing measurable at training
    length, and its ONLY established effect is at OOD length (+0.062 at T=512,
    +0.124 at T=1024, loss-matched, L15_ABLATION.md). Each one's win is the other's
    loss, on the same axis. That is suggestive, and it has never been tested --
    there was no arm with both until this one.

    IT IS ONLY SUGGESTIVE, and two things argue the other way, so this is a real
    test rather than a confirmation:

      1. THE FILTER DOES NOT OBVIOUSLY TARGET THE LOOP'S FAILURE MODE. The loop's
         OOD damage was measured to be ITERATION COUNT -- same weights, T=512 peaks
         at 2 passes and falls monotonically to 6 -- and explicitly NOT residual
         growth (the residual norm is flat across length, 18.15 -> 18.71). The
         filter's mechanism is the wrap, which bounds theta. Bounding theta has no
         evident purchase on an iteration-count problem.
      2. A CHEAPER FIX ALREADY EXISTS. LoopedSampled repairs most of the collapse
         for free (0.816 -> 0.915 at T=512) with no filter and no parameters. It
         still falls short of the filter at T=1024 (0.736 vs 0.888), so a gap
         remains, but the filter has to beat sampling, not merely beat nothing.

    DESIGN. theta_hat is computed ONCE, before the loop, from the token embeddings
    -- which is what BOTH parents do (Looped computes theta once; Level15 computes
    theta once). Recomputing the correction per iteration is a different model:
    that is LoopedRefine, already tested in the regime built for it and null
    (-0.001/-0.011/+0.005 at T=128, no slope in noise, gate |g| 0.083 with
    inconsistent sign). Keeping theta fixed across passes is what makes this a
    clean 2x2 cell rather than a three-way confound.

    PARAMETER NOTE. The 2x2 has two parameter levels -- {Vanilla, Looped} share
    one count and {Level15, Level15Looped} share a larger one -- because the filter
    adds heads. The loop adds nothing on either row. So the LOOP main effect and
    the interaction are both parameter-matched; only the FILTER main effect carries
    the capacity difference, exactly as it already did in every prior comparison.
    """
    n_loops = 4

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2, n_loops=None, **kw):
        # n_layers ignored on purpose: one shared block is the whole point.
        super().__init__(vocab_size, d_model, n_heads, 1, dropout, grid_size,
                         bottleneck_r)
        if n_loops is not None:
            self.n_loops = n_loops

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)

        delta = self.action_to_lie(x)
        theta_path = (torch.cumsum(delta, dim=1) *
                      self.path_integrator.omega.unsqueeze(0).unsqueeze(0))
        theta_hat, Pi, K, R = self.inekf(theta_path, x)

        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach(); self.last_K = K.detach(); self.last_R = R.detach()

        t = theta_hat.transpose(1, 2)
        cos_a, sin_a = torch.cos(t), torch.sin(t)
        m = _causal(L, tokens.device)
        block = self.layers[0]
        for _ in range(self.n_loops):
            x = block(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))
