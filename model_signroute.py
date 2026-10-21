"""G-2 (SIGNROUTE): does the |.| class fail because the READOUT is non-negative, or
because it has NO COLLECTIVE ROUTE?

`docs/paper/expansion/SEARCH_THEORY.md` Sec. 6, item 4 (listed there as the cheapest decisive test; see
WHAT THIS CAN AND CANNOT DECIDE below and SIGNROUTE_PREREG.md Amendment 1 for why it decides less than that):

    "A |.| class with a collective route (tests S3 constructively). |.| applied after
     a learned sign-free reparameterisation that keeps an odd latent: e.g. steps
     |w . z_x| for x and the same with an inverse-pair tie z_y := -z_x (the clock is
     then omega (|w . z| + |w . z|) = 2 omega |w . z|, never zero): predicted to fail;
     vs a "signed-latent, |.|-readout" hybrid a(x) = omega |w| . z_x elementwise
     (readout non-negative, latent signed: collective route restored) predicted to
     succeed. Ring, n = 8 per arm, 16 runs, ~1.5 h. This separates "non-negative
     readout" from "no collective route"."

WHY THESE TWO ARMS DECIDE SOMETHING. Prop. S2 says the signed class reaches a zero
clock on every channel through r numbers: the clock vector is k = diag(omega) W u with
u = z_x + z_y, so {u = 0} is an r-dimensional subspace that every readout contains,
and gradient flow on the quadratic surrogate drives u -> 0 from every start while
leaving the map part delta = (z_y - z_x)/2 untouched. Prop. S3 says the |.| class has
no such route for an inverse pair: the latent move that would cancel ends at
a(y) = a(x) in every channel (S3(a), by evenness of |.|), so cancellation has to be
bought channel by channel in the readout, at per-channel integer wrap offsets
2 pi n_c / |omega_c| spread across an omega ladder of ratio 32 on the ring.

Those two propositions differ in TWO ways at once, and the existing arms cannot
separate them, because Abs changes both together. The table below is MEASURED at
initialisation over 8 seeds (`test_signroute.py::test_init_step_statistics` and the
sign census in SIGNROUTE_PREREG.md Sec. 3), not asserted:

    arm        increment     readout W_out   collective route?    S2/S3 prediction
    signed     48.4% neg     50.5% neg       yes (u = 0)          solves   (LRSWEEP 8/8 at 1e-3)
    abs         0.0% neg     50.5% neg       no  (S3(a))          fails    (LRSWEEP 2/8 at 3e-3)
    hybrid     40.2% neg      0.0% neg       YES                  SOLVES  <-- new
    abs_tied    0.0% neg     50.5% neg       NO, by construction  FAILS   <-- new

READ THE FIRST COLUMN BEFORE QUOTING A HYBRID RESULT. The sign line's constraint (A5,
`model_sign.py`) is on the INCREMENT, and hybrid's increment is SIGNED -- 40.2% of its
per-channel move steps are negative at init, against 0.0% for abs, abs_tied, exp and
softplus. What hybrid constrains is the READOUT MATRIX, to the non-negative orthant
(|W_out| has 0.0% negative entries vs 50.5% for W_out in every other arm; the
STORED hybrid W_out is 50.5% negative too and is irrelevant, since the class applies |.|. omega's sign is free, so the
angle-space readout diag(omega)|W| is sign-coherent per row, not non-negative). So hybrid is NOT a
non-negativity-constrained model and a hybrid win does NOT license "non-negative steps
are fine". It licenses exactly one claim: restricting the readout matrix to the
non-negative orthant costs nothing, so non-negativity per se is not the obstacle -- what
costs is non-negativity OF THE INCREMENT, which is what destroys the collective route.
That is the S2/S3 account, so the signed increment here is the mechanism under test, not
a confound; it is still the sentence most likely to be mis-stated downstream.

WHAT THIS CAN AND CANNOT DECIDE (Amendment 1, findings 2 and 10). Only the HYBRID is a test:
its solution exists (construct_signroute.py, frozen 1.000), so its failure would be a search
failure. abs_tied has NO solution in its class and fails under every account; it is a
tripwire control, not half of a decisive pair. And the hybrid does not isolate S3: its zero
clock sits at k = 0, so it removes both the missing collective route (S3) and the S1 + P3
wrap barrier (a |.| channel cancels only at k = 2 pi, across pi from a small start), which
SEARCH_THEORY's S3 "Reading" calls the larger obstruction. A hybrid win shows the
non-negative readout MATRIX is costless and the signed INCREMENT is what matters; it does
not say which of S3 and S1/P3 is the obstacle. The only rival it refutes is "non-negativity
anywhere in the readout costs", which is a weak rival.

  hybrid    a_c(x) = omega_c (|w_c| . z_x): the abs sits on the READOUT WEIGHTS, the latent
            stays signed. Each step is a non-negative-weighted combination of a signed
            latent, so the step itself can be either sign (40.2% negative at init) while
            the readout matrix is confined to the non-negative orthant -- a real
            restriction of the reachable readouts, not a reparameterisation. The clock is
            diag(omega) |W| u, so {u = 0} is still an r-dimensional subspace of latents:
            S2(a)-(c) apply verbatim with W replaced by |W| (generically still full column
            rank, so H stays positive definite and u -> 0 from every start). If
            non-negativity of the readout were the obstacle, this fails; if the obstacle is
            the missing collective route, this solves.
            Verified in code, not inferred: at z_y = -z_x the hybrid clock is 0.00e+00 on
            all 64 channels with steps of magnitude 0.936 (non-blind), while abs at the
            same latents has clock 1.872 (`test_signroute.py`).

  abs_tied  a_c(x) = omega_c |w_c . z_x| with the inverse pair TIED, z_y := -z_x. The
            collective route is removed by construction while the readout is exactly
            Abs's: the clock is omega_c (|w_c . z_x| + |w_c . (-z_x)|) = 2 omega_c |w_c . z_x|,
            which vanishes only when w_c . z_x = 0, i.e. only blindly (S1(b)). Worse, the
            two move steps are EQUAL, so the angle depends on the move COUNT only and the
            class cannot represent the ring map at all (construct_signroute.py). It fails
            under every account: a TRIPWIRE (any SOLVED run means a bug), not a test.

(An earlier draft of this docstring called hybrid and abs_tied a decisive pair crossing the
factor. That was wrong for both reasons above and is withdrawn.)

SECONDARY ARMS, and an init confound they carry that the primaries do not.

  exp        a = exp(W_out W_in x)        LRU-style, strictly positive, unbounded
  softplus   a = softplus(W_out W_in x)   GRAPE-AP's non-negative omega (= model_sign's "pos")

`model_sign.py` records the confound for softplus and it applies to exp as well:
softplus(0) = 0.693 and exp(0) = 1, so at initialisation EVERY token advances the clock
by the same constant and the model is RoPE with a rescaled omega, whereas signed and abs
start near Delta = 0, i.e. near NoPE. A deficit in these two arms is therefore not
attributable to the sign route alone, which is why they are secondary and the |.|-vs-
hybrid-vs-tied triple is the primary. exp is additionally unbounded, so it can diverge
where the others saturate; the trainer records final loss so that is visible.

OBSERVATION-STEP-FREE, as the spec requires. Every arm here is built on the RANK_TESTS
R1AO construction (`model_wrap_exist.py`): only the two move tokens step, every other
token's step is 0 BY CONSTRUCTION. That is what makes S2/S3's inverse-pair algebra apply
exactly rather than approximately -- the first existence attempt failed because trained
signed ring solutions lean on their observation steps, which a rank-4 bottleneck cannot
also wrap (`model_wrap_exist.py` docstring).

NOT A CONFOUND. w_in / w_out shapes and the parameter count are identical in signed, abs,
hybrid, exp and softplus (no arm adds or removes a parameter). abs_tied is the one arm
whose step path reads only one move embedding, so its SECOND move embedding can receive
gradient through the content path alone. `test_signroute.py` measures the parameter
parity; the second-embedding claim follows from step_table's code (it reads ea[:1]) and is
not separately measured.
"""
import torch
import torch.nn.functional as F

from mapformer.model import ActionToLieAlgebra, MapFormerWM

_MODES = ("signed", "abs", "hybrid", "exp", "softplus")


class SignRouteActionToLie(ActionToLieAlgebra):
    """ActionToLieAlgebra with the sign ROUTE varied, not only the sign.

    `mode`:
      "signed"    W_out W_in x                 the paper's step (passthrough)
      "abs"       |W_out W_in x|               non-negative readout, no collective route
      "hybrid"    |W_out| (W_in x)             non-negative readout, route RESTORED
      "exp"       exp(W_out W_in x)            LRU-style, strictly positive
      "softplus"  softplus(W_out W_in x)       GRAPE-AP style
    """

    def __init__(self, d_model, n_heads, n_blocks, bottleneck_r=2, mode="hybrid"):
        super().__init__(d_model, n_heads, n_blocks, bottleneck_r)
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode

    def from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Readout + constraint applied to an ALREADY-COMPUTED latent z (..., r).

        Exists so the tied-latent arm can supply z_y = -z_x without going through w_in,
        while still using THIS code path for the readout -- rule 11's "the gate must call
        the task code", applied to the arms themselves: there is one readout
        implementation, not one per call site.
        """
        if self.mode == "hybrid":
            # abs on the READOUT WEIGHTS; the latent stays signed, so the clock is
            # diag(omega) |W| u and {u = 0} survives as a route (S2(a) with W -> |W|).
            return F.linear(z, self.w_out.weight.abs())
        delta = self.w_out(z)
        if self.mode == "signed":
            return delta
        if self.mode == "abs":
            return delta.abs()
        if self.mode == "exp":
            return torch.exp(delta)
        return F.softplus(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        delta = self.from_latent(self.w_in(x))
        return delta.view(B, T, self.n_heads, self.n_blocks)


def _signroute_variant(mode, r=4, tie_latents=False):
    """Observation-step-free MapFormer-WM whose two move steps use `mode`.

    Mirrors `model_wrap_exist._ao_variant` (same step_table / forward / freeze hooks) so
    that the `signed` and `abs` arms here are wiring-identical to SignedAO_r4 / AbsAO_r4;
    `test_signroute.py` checks that bitwise at a shared seed rather than trusting it.
    """

    class _SR(MapFormerWM):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                     grid_size=64, bottleneck_r=r, action_ids=(0, 1), **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, r)
            self.action_to_lie = SignRouteActionToLie(d_model, n_heads, self.n_blocks, r, mode=mode)
            self.tie_latents = tie_latents
            self.freeze_latent = False
            self.register_buffer("act", torch.tensor(action_ids, dtype=torch.long), persistent=False)
            if tie_latents and len(action_ids) != 2:
                raise ValueError("tie_latents needs exactly 2 action_ids (an inverse pair)")

        def step_table(self):
            """(V, H, nb) per-token step before omega: the moves' steps, 0 for every other token."""
            e = self.token_emb.weight
            ea = self.frozen_act_emb if self.freeze_latent else e[self.act]
            a2l = self.action_to_lie
            if self.tie_latents:
                # z_y := -z_x BY CONSTRUCTION, so the clock 2 omega |w . z_x| can never
                # vanish non-blindly: the collective route is removed, the readout is not.
                z = a2l.w_in(ea[:1])                       # (1, r) from the FIRST move only
                v = a2l.from_latent(torch.cat([z, -z], 0))  # (2, H*nb)
            else:
                v = a2l.from_latent(a2l.w_in(ea))           # (A, H*nb)
            v = v.view(len(self.act), self.n_heads, self.n_blocks)
            out = torch.zeros(e.shape[0], *v.shape[1:], device=e.device, dtype=v.dtype)
            out[self.act] = v
            return out

        def freeze_move_latents(self):
            """ACCOUNTS T2 / G-2 torus part: the move latents become constants. A buffer copy
            of the move embeddings feeds the step path (the embeddings still train through the
            content path) and w_in gets no grad, so AdamW applies no decay to it either."""
            self.register_buffer("frozen_act_emb",
                                 self.token_emb.weight[self.act].detach().clone(), persistent=True)
            self.freeze_latent = True
            self.action_to_lie.w_in.weight.requires_grad_(False)

        def latents(self):
            """(A, r) the move latents actually fed to the readout -- what S2/S3 call z_x, z_y."""
            ea = self.frozen_act_emb if self.freeze_latent else self.token_emb.weight[self.act]
            z = self.action_to_lie.w_in(ea)
            return torch.cat([z[:1], -z[:1]], 0) if self.tie_latents else z

        def clock(self):
            """(H, nb) the per-channel clock k_c = omega_c * (a_c(x) + a_c(y)) of S2(a), in
            ANGLE units. Zero on every channel = the map's cancellation condition."""
            st = self.step_table()[self.act]                      # (A, H, nb)
            return self.path_integrator.omega * st.sum(0)

        def delta_of(self, tokens):
            return self.step_table()[tokens]

        def get_position_state(self, tokens):
            return self.path_integrator(self.step_table()[tokens])

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            B, L = tokens.shape
            x = self.token_emb(tokens)
            cos_a, sin_a = self.path_integrator(self.step_table()[tokens])
            causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
            for layer in self.layers:
                x = layer(x, cos_a, sin_a, causal_mask)
            return self.out_proj(self.out_norm(x))

    _SR.__name__ = f"MapFormerWM_{mode}{'Tied' if tie_latents else ''}AO_r{r}"
    _SR.__doc__ = (f"Observation-step-free MapFormer-WM, rank r={r}, step mode {mode!r}"
                   f"{', inverse-pair latents TIED' if tie_latents else ''}.")
    return _SR


# The G-2 ring arms. Rank 4 is the shared rank of LRSWEEP's SignedAO_r4 / AbsAO_r4, so
# these sit on the same recipe as the anchors they are read against.
MapFormerWM_SignedAO_r4   = _signroute_variant("signed")                      # anchor, solves
MapFormerWM_AbsAO_r4      = _signroute_variant("abs")                         # anchor, fails
MapFormerWM_HybridAO_r4   = _signroute_variant("hybrid")                      # PRIMARY (the only test)
MapFormerWM_AbsTiedAO_r4  = _signroute_variant("abs", tie_latents=True)       # CONTROL C: no solution exists
MapFormerWM_ExpAO_r4      = _signroute_variant("exp")                         # secondary
MapFormerWM_SoftplusAO_r4 = _signroute_variant("softplus")                    # secondary

ARMS = {"signed": MapFormerWM_SignedAO_r4, "abs": MapFormerWM_AbsAO_r4,
        "hybrid": MapFormerWM_HybridAO_r4, "abs_tied": MapFormerWM_AbsTiedAO_r4,
        "exp": MapFormerWM_ExpAO_r4, "softplus": MapFormerWM_SoftplusAO_r4}
