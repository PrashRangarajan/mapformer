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
