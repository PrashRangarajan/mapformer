---
name: reference-positional-landscape
description: The unified log-polar frame for positional mechanisms, and the prior art that already occupies most of it -- GRAPE and Mamba-3. Read before claiming any theory contribution.
metadata:
  type: reference
---

**The frame** (`mapformer_math.tex`, ~24pp, the document of record). Every rotary
logit is a sum of magnitude x cos(phase). In log-polar coordinates with an
instantaneous complex log `c` and an accumulated one `G`:

    q_t = exp(c(x_t) - conj(S_t)),  k_s = exp(c(x_s) + S_s),  S_t = sum_{u<=t} G(x_u)

Real parts are magnitude, imaginary parts are phase. The `conj` is load-bearing:
`-S_t` makes the phase terms ADD rather than cancel and destroys relative
position (verified, off by 4 orders of magnitude).

**PRIOR ART -- most of this is published. Verified first-hand, not from a search
summary.**

- **GRAPE** (arXiv:2512.07805, ICLR 2026) is the two-slot unification: multiplicative
  rotations in SO(d) + additive logit biases from UNIPOTENT actions in GL,
  recovering RoPE and ALiBi exactly and PROVING FoX is an exact instance
  (`omega_t := log f_t`, bias = sum over the interval). Its Appendix D calls its
  own multiplicative extension "NON-CONTEXTUAL"; Appendix E composes both as one
  subgroup indexed by the OFFSET. So its content-dependence is entirely on the
  additive side, and there it needs `omega = g(x) >= 0` -- a monotone clock.
- **Mamba-3** (arXiv:2603.15569, Mar 2026, Dao/Gu) closes the rest. Complex SSM
  `Diag(A(t) + i theta(t))` with BOTH parts data-dependent; Prop 3 is titled
  "Complex SSM, Data-Dependent RoPE Equivalence" and accumulates the rotations
  onto B and C, "which correspond to the query and key components of attention".
  Signed, abelian, accumulated, on Q/K.

**So the taxonomy, the closure argument and the content-dependent rotation are all
published.** Do NOT claim them.

**What survives, and it is empirical not structural:**
1. **The rank of the content-to-angle map.** Nobody constrains it -- Mamba-3's
   theta is a per-channel projection. r=2 is the WORST setting; r=4 buys +0.085 at
   4x training length for 384 params, 8/8 seeds; the cause is a skewed basis.
2. **The navigation regime.** All of this literature evaluates on language, recall
   or state tracking. None runs grid navigation.

**Also corrected:** my "there is no third slot" closure claim is FALSE. It assumed
SEMISIMPLICITY; the unipotent/dual-number shear commutes, is interval-relative to
7e-16, and gives a term LINEAR in the accumulator that exp(Re)cos(Im) cannot
express. That slot is GRAPE's additive family. The claim that survives is
restricted to a fixed maximal torus of GL(d).

**The untested axis worth having:** SIGN. MapFormer's Delta is signed; GRAPE-AP
and CARoPE are constrained non-negative, as a side-effect of using a softplus or
gate rather than by design. A monotone clock cannot represent a -1 action, so the
prediction is: invisible on language, decisive on navigation. Two-arm ablation on
existing machinery. See [[project_rank_and_selective_rope]].


## Verified 2026-09-06 — the frame has a published proof, and three claims changed

All 28 sources are now read first-hand and stored (see [[reference-paper-corpus]]).

**The frame is Puranik's** (Jane Street Blog, 22 Apr 2026, `papers/txt/puranik_janestreet.txt`):
linear + translation-invariant + continuous ⇒ `A(d)=exp(dM)` a one-parameter group,
so classifying positional encodings IS classifying M up to conjugacy. Diagonalisable
real ⇒ decay/NoPE; conjugate pair ⇒ damped RoPE (this is the Re/Im split — there are
two slots because C has two); defective ⇒ polynomial, which he flags as unexplored.
He also states the reparameterised-time reading as an aside: gated models "learn how
far to advance time, as opposed to changing the rate of decay". Jordan-RoPE
(2605.04217) then fills the defective cell AND needs a contragredient query action —
the q/k sign asymmetry, published, in non-semisimple form.

**Three corrections to the relational map:**
- CARoPE differs on TWO axes, not one: `f = 1/(softplus(xW)+1) ∈ (0,1)` — cannot
  reverse *or stay* — and `W ∈ R^{d×h}` is one scalar per head, so its rank is
  1/head, BELOW MapFormer's r, not above.
- TAPE (2501.00712), not "TAPA", and it breaks the PATH STRUCTURE (layerwise update
  through attention+MLP), not the per-token factorisation. It belongs with
  MesaNet/Titans, not with CoPE.
- Jordan-RoPE is NON-SEMISIMPLE (complex eigenvalue coupled to a nilpotent in one
  block, modes d^r e^{iωd}), not unipotent, and its magnitude factor is polynomial
  in the lag, not exponential.

**LieRE (2406.10322, ICML 2025) is the rank axis's nearest neighbour and must be
cited.** `R(p) = exp(Σ p_i A_i)` with a learned skew-symmetric basis — structurally
MapFormer's ActionToLieAlgebra driven by a known POSITION. Its Fig 6 sweeps generator
tile size 2x2..48x48 and PEAKS AT 8x8, an interior optimum: same shape as our step at
r=2, on the other side of the map (output-side density vs input-side bottleneck).
The input-side bottleneck is what remains unstudied.

**The two HGRN claims are verified and are published negatives for content-dependent
phase ON LANGUAGE.** HGRN Table 11, in its own words: "the phase argument θ should
not be data-dependent". HGRN2 Table 1: real HGRN at 2x state size beats complex
HGRN — the gain was state expansion, not the phase. With our own underpowered enwik8
result that is three independent lines pointing the same way on text, and none of
them evaluates navigation. This STRENGTHENS the scoping of the surviving claim.

**PoPE is defined by DELETING RoPE's implicit content-phase interaction term**
(φ_k − φ_q), keeping only the content-dependent magnitude. The exact complement of
MapFormer, which amplifies that term into a learned accumulator. The polar frame
equation in the note is PoPE's eq. 2, and should be attributed as such.
