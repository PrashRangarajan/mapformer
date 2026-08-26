# MiniWorld fresh-map factorial — findings (2026-08-26)

The allocentric-flip test in the IN-CONTEXT (fresh-map) regime, the one where path
integration was expected to be load-bearing (like the torus, +0.461). Full
factorial: {Vanilla, MapPoPE-Flat}=path-int × {RoPE, PoPE-Flat}=index × {raw, allo}
× 3 seeds = 24 arms, d=256/4-layer/100 epochs, held-out on a NEW map. All guardrails
honoured (fresh-map n-gram gate PASS both encodings; context-destruction ablation
PASS all 24; solvable arm present; NLL-led; one batch).

## Verdict: NO FLIP — and path integration is a LIABILITY here

Position effect = (path-int mean) − (index mean), paired within seed:

| length | raw | allocentric |
|---|---|---|
| T=512  | **−0.086 ± 0.021** (all 3 seeds neg) | **−0.174 ± 0.035** (all 3 seeds neg) |
| T=1024 | **−0.051 ± 0.020** | **−0.184 ± 0.034** |

Per-arm nb-acc, T=512:

| model | position | raw | allocentric |
|---|---|---|---|
| RoPE | index | 0.398 | **0.501** |
| PoPE-Flat | index | 0.384 | 0.364 |
| MapPoPE-Flat | path-int | 0.308 | 0.232 |
| Vanilla | path-int | 0.303 | 0.284 |

Two results, both opposite the hypothesis and consistent across all 3 seeds:
1. **Index > path-int on fresh-map** (position effect NEGATIVE at both lengths) —
   the reverse of the torus. Path integration does not help in-context continuous-3D
   navigation; it hurts.
2. **Allocentric does NOT rescue path-int — it widens the index lead** (−0.086 →
   −0.174). The richer displacement-direction input helps the flexible attention
   (index) arm most (RoPE-allo 0.501, the best arm); the rigid cumsum benefits less.

## The probe was an undertraining artifact (F5 confirmed)

The 40-epoch probe had Vanilla-raw at chance (0.093) and I read it as "path-int
collapses under raw rotation actions → the flip setup." At the full 100-epoch
budget Vanilla-raw is 0.356 — nowhere near chance. The dramatic probe signal was
the F5 undertraining the validity review warned about, not mechanism. Lesson
re-learned: a weak number at a fixed budget is not a result (standing rule 5).

## Validation (why the negative is trustworthy)

- **Learnable in-context:** RoPE-allo 0.501, NLL 1.36 ≪ chance NLL 2.77 — a
  solvable arm exists, so this is not an under-data null (F1).
- **Context-destruction PASS on all 24 arms:** intact 0.17–0.51 → obs-shuffle
  0.01–0.09, action-shuffle 0.02–0.06, all near the 0.076 marginal / 0.0625 chance
  (MINIWORLD_FRESH_ABLATION.md). Every model, including the winning index arms,
  uses a genuine in-context map — the comparison is not a shortcut artifact.
- **Flagged arms are converged, not undertrained:** the 2 arms the aggregator
  flagged (Vanilla_raw_s2 1.53, MapPoPE_allo_s0 1.60) plateaued at those losses
  (end-slope ≈ 0) — the high loss IS the result (path-int handles this task poorly),
  not a convergence failure. The effect is negative across all seeds regardless.

## Mechanism (honest read)

On continuous-3D in-context navigation, **learned attention-based position (index)
beats the hardwired SO(2) cumsum (path-int)**, and richer allocentric input helps
the flexible attention arm more than the rigid cumsum. This matches the project's
standing finding — "a plain/index transformer path-integrates via attention;
MapFormer's SO(2) code is an inductive bias, not privileged info." On the torus
(clean discrete ±1 translations, exactly integrable) that inductive bias wins big;
in continuous 3D the quantized displacement is an imperfect integrand and the fixed
cumsum is worse than free attention.

## Why the MiniGrid flip does NOT transfer

MiniGrid's allocentric recode worked because on a discrete grid the displacement is
EXACTLY integrable (±x/±y). In continuous 3D the recode is a quantized direction
(24 bins) over variable geometry, so the cumsum accumulates error while attention
does not. **The allocentric-recoding result is scoped to environments where
displacement is exactly discrete; it does not extend to continuous geometry.**

## WHY the flip fails — reconstruction fidelity (two-agent forensic, 2026-08-26)

MapFormer position = COMMUTATIVE cumsum of a FIXED per-token-id displacement
(ActionToLieAlgebra rank-2 bottleneck -> per-id Δ; PathIntegrator cumsum + RoPE).
So the best position it can represent is a fixed 2D vector per action id summed
along the path. The metric that sets the sign: R² = fraction of each step's true
displacement determined by the action token id (1.0 = exactly integrable). It
tracks the position effect MONOTONICALLY (200 trajs, oracle least-squares fit =
upper bound the trained model cannot exceed):

| env × encoding | R² | drift@512 (cells) | position effect |
|---|---|---|---|
| torus (±1 translate) | 1.0000 | 0.00 | +0.461 |
| MiniGrid allo (±x/±y) | 0.9994 | 0.13 | +0.02 |
| MiniWorld allo (24-bin dir) | 0.5506 | 2.62 | −0.174 |
| MiniWorld raw (turn/fwd) | 0.0000 | 4.30 | −0.086 |

The cumsum wins when it is an EXACT position estimator, loses when noisy/biased.
MiniWorld allo leaves 45% of each step un-integrable -> 2.6-cell drift -> worse
than attention-learned position.

Why allo (−0.174) is WORSE than raw (−0.086): raw `forward` is isotropic so its
best fixed delta ≈ 0 -> cumsum barely moves -> position code degenerates to near-
static -> falls back to index-like -> only mildly worse. allo gives every bin a
definite non-zero vector -> cumsum integrates a CONFIDENTLY-DRIFTING wrong position
-> misleads attention MORE. A wrong-but-confident code beats a degenerate one at
being harmful.

Root cause within allo: NOT the 24-bin angle quantization (3.46°) but MAGNITUDE
VARIANCE — forward-step CV = 0.49 (the macro runs a variable # of 0.15m substeps
until the cell changes, so displacement magnitude swings ±50%), unrepresentable by
a fixed direction vector.

One-axis synthesis: sign = reconstruction fidelity; magnitude = fidelity −
attention's own localization ability (torus index arms at chance floor -> exact
code supplies everything -> +0.461; small MiniGrid attention localizes -> exact
code worth only +0.02). Refuted: ω-scale (fixed-map same geometry, no disadvantage),
drift-with-length (effect shrinks with T), in-context-demand-alone (torus is also
fresh-map yet +0.461).

## Decisive confirmatory experiment (designed, not yet run)

Oracle exact-cell-displacement recode on fresh-map MiniWorld: emit the exact
integer cell transition (Δgx, Δgz) ∈ {−1,0,+1} (a ≤10-class token) instead of the
24-bin direction. This makes cumsum reconstruct the obs-map cell EXACTLY (R²→1,
like MiniGrid) while holding env/in-context-demand/models/budget fixed — only
fidelity changes. Within-batch: {Vanilla,MapPoPE}×{RoPE,PoPE}×{24-bin control,
oracle-exact}×3 seeds, fresh-map, 100ep, + n-gram gate + context-destruction.
Predicted: path-int FLIPS positive with the oracle recode (24-bin control stays
negative in the same batch) -> H1 confirmed causally, and the negative becomes a
mechanistic result WITH A FIX. If it does NOT flip -> fidelity is not the whole
story and the residual is genuine in-context interference. Report the multi-cell-
jump clamp rate as a caveat.

## ORACLE EXPERIMENT RESULT (2026-08-26) — H1 REFUTED, mechanism corrected

The decisive test ran. Exact-cell recode (R²→1, clamp rate 0) vs 24-bin allo
(R²=0.55), same env/demand/budget, one batch, n=3, gated + ablated:

| model | position | allo (24-bin) | oracle (exact cell Δ) |
|---|---|---|---|
| RoPE | index | 0.501 | **0.977** |
| PoPE-Flat | index | 0.364 | **0.938** |
| Vanilla | path-int | 0.284 | 0.448 |
| MapPoPE-Flat | path-int | 0.232 | 0.324 |

Position effect (path-int − index): allo −0.174 → **oracle −0.571** (T=512),
−0.184 → −0.409 (T=1024), all seeds. Ablation PASS all arms (RoPE-oracle 0.977 →
shuffled 0.07 = genuine map).

**H1 (reconstruction fidelity) is REFUTED.** The prediction was "exact recode →
path-int flips positive". Instead the exact displacement let the INDEX arm's
ATTENTION reconstruct position near-perfectly (0.94–0.98), while the rigid cumsum
lagged — the effect went MORE negative. So the forensics R²-correlation
(1.0/+0.46, 0.999/+0.02, 0.55/−0.17, 0.00/−0.09) was CONFOUNDED: fixing R²→1 does
not flip the sign. Higher fidelity helps whoever can USE it; on MiniWorld's small
grid that is attention, not the fixed cumsum.

**Corrected mechanism — attention substitutability (the driver of the SIGN):**
whether attention can integrate position from the action tokens within its horizon,
set by grid size / revisit distance vs the ~2–32-step attention horizon.
- Torus 64×64, long revisits: attention CANNOT span them (index arms at chance
  floor) → the hardwired cumsum is the ONLY integrator → path-int +0.46.
- MiniWorld 8×8, short revisits: attention CAN integrate → and given good tokens
  beats the rigid cumsum → path-int loses; better tokens (oracle) widen the index
  lead (index near-solves 0.94–0.98).
Token fidelity MODULATES magnitude (helps whoever can use it); it does not set the
sign. This is the mechanism agent's "second factor" (attention substitutability),
promoted from magnitude-only to the sign-setting driver by the oracle result.

**Falsifiable follow-up prediction:** on a LARGER MiniWorld grid (revisit distances
exceeding the attention horizon), path-int should start winning again even with the
oracle recode — because attention can no longer substitute. Untested; the clean
next experiment if this line continues.

**Honest correction:** the "reconstruction fidelity explains the flip" conclusion
(reported confidently after the two-agent forensic) was REFUTED by the very
experiment designed to confirm it. The forensics measurement (R² values) is real;
the causal story built on it was wrong. This is why the decisive experiment was
worth running.

## Implication for Habitat

The premise of the MiniWorld→Habitat path — that allocentric recoding rescues path
integration in continuous 3D — is FALSIFIED in the in-context regime. Both MiniWorld
regimes (fixed-map §earlier, fresh-map here) show no flip; fresh-map is a stronger
negative (path-int is a liability). A discrete direction recode is not enough for
continuous geometry; Habitat would need a continuous-magnitude displacement recode
(full vector), and even then attention may match or beat it. This is the honest
place to STOP the "allocentric flip in 3D" line, or pivot to the continuous-vector
recode as a distinct, separately-motivated experiment — not a foregone win.

## GRID SWEEP (2026-08-26) — mechanism CONFIRMED, crossover measured

Fresh-map oracle recode, {Vanilla=path-int, RoPE=index}, grids 8/16/24/32, n=3.
Position effect (path-int − index):

| grid | T=512 | T=1024 |
|---|---|---|
| 8  | −0.529 | −0.253 |
| 16 | +0.148 | +0.333 |
| 24 | +0.076 | +0.208 |
| 32 | +0.087 | +0.172 |

The sign FLIPS between grid 8 and 16 (all 3 seeds) and stays positive. Confirms
attention substitutability as THE mechanism: RoPE(index) near-solves at grid 8
(0.977) — attention integrates the exact tokens on a small grid — then collapses
to ~0.62 at grids ≥16 as revisit distances exceed its ~2-32-step horizon, while
Vanilla(path-int) holds (0.89→0.69→0.70) via the bounded cumsum. So the fresh-map
"path-int is a liability" negative was a SMALL-GRID artifact; path integration WINS
once the map outgrows attention's horizon (grid ≥16). T=1024 effect > T=512 at
every grid (longer seq → more revisits past the horizon → bigger path-int win).
Nuance: effect peaks at grid 16, dips slightly at 24/32 (both arms degrade as the
grid grows) but stays clearly positive. Source: MINIWORLD_GRID_SWEEP.md.
