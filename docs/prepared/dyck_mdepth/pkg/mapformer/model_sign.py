"""THE SIGN ABLATION: may the phase increment be negative?

WHY THIS EXISTS. The relational map (mapformer_math.tex sec. 7) has eight axes.
Seven of them are varied deliberately somewhere in the literature. A5 -- the SIGN
of the accumulated increment -- is not. Every content-dependent-phase mechanism
outside MapFormer is non-negative, and in every case it is a side-effect of a
squashing function rather than a design choice that was argued for:

    CARoPE   f(x) = 1/(softplus(xW)+1)  in (0,1)   -- cannot reverse OR stay
    GRAPE-AP omega = g(x) >= 0                     -- required by the construction
    CoPE     g_ij = sigmoid(q_i . k_j) >= 0
    FoX      log sigmoid(.) < 0                    -- negative by construction
    MapFormer  Delta = W_out W_in x                -- SIGNED, because nothing squashes it

(All four verified first-hand 2026-09-06; sources in papers/, index in
papers/INDEX.md.)

THE MECHANISM CLAIM, stated so it can fail. Under a non-negativity constraint the
per-channel angle theta_t^(c) = omega_c * sum_{u<=t} Delta_u^(c) is MONOTONE in t.
Monotone is all a language clock needs. It is not enough for a cognitive map: on
the torus, east and west must cancel, and a monotone code cannot represent net
displacement. It can encode (n_east, n_west) but not n_east - n_west, and revisit
retrieval needs exactly the latter -- two paths reaching the same cell by
different step counts must produce the SAME phase. So:

    prediction: invisible on language, decisive on navigation.

WHERE THE CONSTRAINT IS APPLIED, and why it matters. AFTER w_out, on the 64
per-(head, block) deltas -- so every channel is monotone. This is what CARoPE and
GRAPE-AP do (they constrain the scalar increment itself). Applying it after w_in
instead, on the r-dimensional latent, would NOT test the claim: w_out could still
carry negative entries and the channels would stay signed. Because the constraint
sits after w_out it is also independent of r.

THE THREE ARMS, and which one is the isolation.

  Abs      Delta = |W_out W_in x|
           THE PRIMARY. Sign removed and NOTHING ELSE: same parameters, same map,
           no squashing (gradient magnitude preserved), Delta still reaches 0
           exactly, and at init |Delta| is half-normal with the same scale as
           Vanilla's Delta. It is the only arm that varies A5 alone.

  Pos      Delta = softplus(W_out W_in x)         -- GRAPE-AP's non-negative omega
  CARoPE   Delta = 1/(softplus(W_out W_in x)+1)   -- CARoPE verbatim, in (0,1)

           Literature-faithful, and both carry an INIT CONFOUND that Abs does not:
           softplus(0)=0.693 and 1/(softplus(0)+1)=0.591, so at initialisation
           every token advances the clock by the same constant and the model IS
           RoPE (with a rescaled omega), whereas Vanilla starts near Delta=0, i.e.
           near NoPE. For CARoPE this is faithful rather than accidental -- the
           paper initialises to RoPE on purpose and calls RoPE its special case --
           but it means a deficit in these two arms is not attributable to sign
           alone. That is precisely why Abs is the primary and these are secondary.

NOT A CONFOUND: parameter count is identical to Vanilla in all three arms (the
constraint adds no parameters), and the rank of the content-to-angle map is
unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import MapFormerWM, ActionToLieAlgebra


class SignConstrainedActionToLie(ActionToLieAlgebra):
    """ActionToLieAlgebra with the per-channel increment forced non-negative.

    `mode` is one of:
      "signed"  -- passthrough, identical to the parent (used as a wiring check)
      "abs"     -- |z|                    scale-preserving, reaches 0, no squash
      "pos"     -- softplus(z)            GRAPE-AP style, strictly positive
      "carope"  -- 1/(softplus(z)+1)      CARoPE verbatim, bounded in (0,1)
    """

    _MODES = ("signed", "abs", "pos", "carope")

    def __init__(self, d_model, n_heads, n_blocks, bottleneck_r=2, mode="abs"):
        super().__init__(d_model, n_heads, n_blocks, bottleneck_r)
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = super().forward(x)          # (B, T, H, n_blocks), signed
        if self.mode == "signed":
            return delta
        if self.mode == "abs":
            return delta.abs()
        if self.mode == "pos":
            return F.softplus(delta)
        return 1.0 / (F.softplus(delta) + 1.0)   # carope


def _sign_variant(mode, r):
    class _S(MapFormerWM):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, r)
            # replace in place; parameter count and shapes are unchanged
            self.action_to_lie = SignConstrainedActionToLie(
                d_model, n_heads, self.n_blocks, r, mode=mode)
        def delta_of(self, tokens):
            """Diagnostic hook for the gate: the raw per-channel increments."""
            return self.action_to_lie(self.token_emb(tokens))
    _S.__name__ = f"MapFormerWM_{mode}_r{r}"
    _S.__doc__ = f"MapFormer-WM, rank r={r}, phase increment constrained: {mode}."
    return _S


MapFormerWM_Abs_r4     = _sign_variant("abs",    4)
MapFormerWM_Pos_r4     = _sign_variant("pos",    4)
MapFormerWM_CARoPE_r4  = _sign_variant("carope", 4)
MapFormerWM_Signed_r4  = _sign_variant("signed", 4)   # wiring check only

MapFormerWM_Abs_r2     = _sign_variant("abs",    2)
