"""G-1 (RANKCLOCK): the rank clock-route batch, torus T = 1024 (`docs/paper/expansion/RANK_THEORY.md` Sec. 6 G1/G1+,
`SEARCH_THEORY.md` 6.1/6.3, `SYNTHESIS_DRAFT2_PLAN.md` row G-1).

THE ACCOUNT UNDER TEST (RANK_THEORY Summary, THEORY not yet registered). Every move adds a step common to all moves -- a
clock c -- to the latent. A channel stops seeing it either through the latent (shrink c: one global coordinate) or through
its own readout. At per-head rank r > D the readout can turn away from the part of c outside the map plane without touching
the map ("parking"); at r = D the clock lies in the map plane and a readout that stops seeing it must stop resolving the
drift direction ("hiding"). On the 2D torus D = 2, and our SHARED rank 2 solves 0/8 at T = 1024 while matched-init shared
rank 4 solves 8/8 (`RANK_MI_RESULTS.md`; four of the eight r2 runs were still DESCENDING at 900 epochs).

The rival: over-parameterisation (Burer-Monteiro style) -- spare rank smooths the landscape of the map factorisation, and
the clock has nothing to do with it.

ARMS added here (all built from `Vanilla`'s base at the seed, so every weight shared with Vanilla -- including W_in and W_out
-- has the SAME initial value; none of these classes draws a random number after the base is built):

  r2Clock  shared r2 + a per-channel CLOCK SLOT: Delta_c(x) = (W_out W_in x)_c + clock_c for EVERY token, clock initialised
           at 0 (+64 parameters = 2 heads x 32 channels, no RNG draw). Equivalent to a third latent coordinate with a fixed
           step of 1 per token whose readout column is learned: the map stays rank 2, but each channel gets a private,
           map-free way to cancel its own drift. At init the function equals Vanilla's exactly. Account: solves like r3/r4
           (>= 6/8). Over-parameterisation: like r2 (<= 2/8). Ported from the CPU pilot's
           `docs/paper/expansion/scripts/rank/rank_geom.MapFormerWM_r2Clock`, with the slot moved INSIDE the step module so
           the state-dict keys of W_in / W_out are Vanilla's (`action_to_lie.w_in.weight`), which the evaluators read.

  r2DF     shared r2, drift FORBIDDEN by construction (RANK_TESTS' R1DF on the torus): each inverse pair shares one
           direction, a_x = (e(+x) - e(-x)) / sqrt(2), and the two moves get +W_out W_in a_x and its exact negative; every
           other token's step is 0. So the clock is 0 and observation steps are 0 at every point of training. If r2 fails
           because training cannot remove the clock, r2DF solves like r4; if r2DF also fails, the rank-2 obstacle is not
           the clock (or not only).

  r2AO     shared r2, observation and blank steps forced to 0, all four move steps FREE (RANK_TESTS' R1AO on the torus).
           Removes the observation steps (the LEAK_RESULTS effect) but leaves the move-common component free, so r2DF vs
           r2AO isolates the clock and r2AO vs r2 isolates the observation steps.

The other arms are existing variants: `Vanilla` (shared r2) and `Vanilla_r4mi` (shared r4 built from r2's base).

Torus moves (environment.GridWorld, action_offset 0): 0 = (-1, 0), 1 = (+1, 0), 2 = (0, -1), 3 = (0, +1); inverse pairs
(0, 1) and (2, 3). The classes assert nothing about the env; `test_rankclock.py` checks the ids against the env.
"""
import torch
import torch.nn as nn

from mapformer.model import ActionToLieAlgebra, MapFormerWM

TORUS_ACTIONS = (0, 1, 2, 3)          # (-x, +x, -y, +y)
TORUS_PAIRS = ((1, 0), (3, 2))        # (+move, -move) token ids per inverse pair


class ActionToLieWithClock(ActionToLieAlgebra):
    """ActionToLieAlgebra plus a per-channel additive step `clock` (n_heads, n_blocks), applied to every token."""

    def forward(self, x):
        return super().forward(x) + self.clock[None, None]


class MapFormerWM_r2Clock(MapFormerWM):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64,
                 bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        # keep Vanilla's W_in / W_out (same draws, same keys); add the slot without any RNG draw
        self.action_to_lie.__class__ = ActionToLieWithClock
        self.action_to_lie.clock = nn.Parameter(torch.zeros(n_heads, self.n_blocks))


class ActionToLieWithAxisClock(ActionToLieAlgebra):
    """ActionToLieAlgebra plus a per-channel additive step for each move AXIS: clock_x on the two x-moves, clock_y on the
    two y-moves, nothing on observation / blank tokens. Needs the token ids, so the model passes them in `_ids`."""

    def forward(self, x):
        d = super().forward(x)
        tok = self._ids                                     # set by the model for the current forward pass
        if tok is None:
            raise RuntimeError("ActionToLieWithAxisClock needs the token ids (call through the model)")
        ax = torch.zeros(*tok.shape, 2, device=d.device, dtype=d.dtype)
        ax[..., 0] = ((tok == TORUS_ACTIONS[0]) | (tok == TORUS_ACTIONS[1])).to(d.dtype)
        ax[..., 1] = ((tok == TORUS_ACTIONS[2]) | (tok == TORUS_ACTIONS[3])).to(d.dtype)
        return d + ax[..., 0, None, None] * self.clock_x[None, None] + ax[..., 1, None, None] * self.clock_y[None, None]


class MapFormerWM_r2ClockXY(MapFormerWM):
    """Shared r2 + one clock slot PER MOVE AXIS (Amendment 1, finding 1): Delta(x-move) += clock_x, Delta(y-move) +=
    clock_y, observation / blank steps unchanged. +128 parameters, zero init, no RNG draw: identical to Vanilla at init.

    Why: a walk's drift has a COMMON part, (t/2)(u_x + u_y)/4 per channel, and a DIFFERENTIAL part, (n_x - n_y)(u_x - u_y)/4.
    The single per-token slot can cancel only the common part; rank 4 has room for both (SEARCH_THEORY S6(b): two
    independent pair clocks need r >= 4). This slot can cancel both, so it is the arm with the exits rank 4 has while the
    map stays rank 2. Relation to the single slot: with clock_x = clock_y = 2c it equals the per-token slot c at every
    OBSERVATION position (k moves and k observations) and differs by c at ACTION positions (one observation short), so it
    does not strictly contain it."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64,
                 bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        self.action_to_lie.__class__ = ActionToLieWithAxisClock
        self.action_to_lie.clock_x = nn.Parameter(torch.zeros(n_heads, self.n_blocks))
        self.action_to_lie.clock_y = nn.Parameter(torch.zeros(n_heads, self.n_blocks))
        self.action_to_lie._ids = None

    def step_table(self):
        """(V, H, nb) per-token step: every vocabulary id passed through the step map with its own id."""
        V = self.token_emb.weight.shape[0]
        ids = torch.arange(V, device=self.token_emb.weight.device)[None]
        self.action_to_lie._ids = ids
        try:
            return self.action_to_lie(self.token_emb(ids))[0]
        finally:
            self.action_to_lie._ids = None

    def delta_of(self, tokens):
        return self.step_table()[tokens]

    def get_position_state(self, tokens):
        return self.path_integrator(self.step_table()[tokens])

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.step_table()[tokens])
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        return self.out_proj(self.out_norm(x))


class _StepTableR2(MapFormerWM):
    """Shared rank 2 whose per-token steps come from `step_table()`; W_in / W_out are Vanilla's own (same draws)."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64,
                 bottleneck_r=2, action_ids=TORUS_ACTIONS, pairs=TORUS_PAIRS, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        self.register_buffer("act", torch.tensor(action_ids, dtype=torch.long), persistent=False)
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long), persistent=False)
        self._pairs_list = [tuple(int(i) for i in p) for p in pairs]     # host copy: no device sync per forward

    def step_table(self):
        raise NotImplementedError

    def delta_of(self, tokens):
        return self.step_table()[tokens]

    def get_position_state(self, tokens):
        return self.path_integrator(self.step_table()[tokens])

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.step_table()[tokens])
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_r2DF(_StepTableR2):
    def step_table(self):
        e = self.token_emb.weight
        out = torch.zeros(e.shape[0], self.n_heads, self.n_blocks, device=e.device, dtype=e.dtype)
        for p, m in self._pairs_list:
            a = (e[p] - e[m]) / (2 ** 0.5)                       # variance of one embedding (RANK_TESTS Amendment 1)
            v = self.action_to_lie(a.view(1, 1, -1))[0, 0]       # (H, nb)
            out[p] = v
            out[m] = -v
        return out


class MapFormerWM_r2AO(_StepTableR2):
    def step_table(self):
        e = self.token_emb.weight
        v = self.action_to_lie(e[self.act].view(1, len(self.act), -1))[0]   # (A, H, nb), free
        out = torch.zeros(e.shape[0], *v.shape[1:], device=e.device, dtype=v.dtype)
        out[self.act] = v
        return out
