# Does a non-negative READOUT MATRIX carry the |.| cost, or the non-negative INCREMENT? -- results (registered, `SIGNROUTE_PREREG.md` + Amendment 1)

Batch 2026-10-20 23:36 -> 10-21 07:43 on GPU 1 only, one job at a time, code `4adfe26` (clean tree, md5 guard over 100 files),
torch 2.10.0+cu128, 96 runs: six observation-step-free rank-4 arms x lr 1e-3, 3e-3 x fresh seeds 1800-1807, ring 32 cells,
moves +1/-1, LRSWEEP recipe verbatim (T = 128, 300 epochs, cosine, wd 0.05). Independent results-blind check before launch
(Amendment 1: 2 MAJOR, 8 MINOR, 5 NOTE, all folded in). Analysis `analyze_signroute.py` -> `SIGNROUTE_ANALYSIS.txt`,
`SIGNROUTE_VERDICTS.json`. Pilots (seeds 1880-1883) are in the prereg Sec. 7 and are not pooled.

**Reading (registered): SUPPORTED -- at lr 1e-3 and 3e-3, a non-negative readout matrix is costless and the signed increment
is what matters.** The hybrid (readout |W_out| in the non-negative orthant, latent and increment signed) solves 8/8 at both
rates with accuracy 1.000 on every seed, identical to signed; |.| solves 0/8 and 2/8. **Scope (Amendment 1, finding 2): this
does NOT separate the missing collective route (S3) from the S1/P3 wrap barrier** -- the hybrid's zero clock sits at k = 0, so
it removes both, and both accounts predicted this outcome. It refutes only "non-negativity anywhere in the readout costs", which
nothing in `SEARCH_THEORY.md` predicted. It is a positive control for the S2/S3 framing plus a two-rate in-batch replication of
the sign cost, not the decisive S3 test the runbook listed.

## Primary (registered): hybrid H vs |.| A and vs signed S, gated on the in-batch anchors

| lr | signed | \|.\| | **hybrid** | abs_tied (control) | G0 / G1 | primary |
|---|---|---|---|---|---|---|
| 1e-3 | 8/8, 1.000 | 0/8, 0.918 | **8/8, 1.000** | 0/8, 0.739 | pass / pass (S vs A Fisher p 0.0002) | SIGNED INCREMENT RESTORES |
| 3e-3 | 8/8, 1.000 | 2/8, 0.958 | **8/8, 1.000** | 0/8, 0.748 | pass / pass (S vs A Fisher p 0.007) | SIGNED INCREMENT RESTORES |

SOLVED, mean held-out accuracy at T = 128 (fresh maps, 400 trials). H vs A: Fisher p 0.0002 at 1e-3, 0.007 at 3e-3 (permutation
p identical, d +0.082 / +0.042). H vs S: p 1.0, d 0.000 at both rates (full = true; likeA = false). The |.| anchor replicates
LRSWEEP (1/8 at 1e-3, 2/8 at 3e-3 there; 0/8 and 2/8 here).

**Control C held:** abs_tied 0/16 SOLVED, as the proof requires (its two move steps are equal, so its code depends on the move
count only). Its 0.739 / 0.748 sits between the n-gram floor (0.554) and retrace (0.860), consistent with a code that carries
displacement parity and nothing more.

## Mechanism (descriptive, registered as such): the hybrid ends where S2 says, |.| does not

Medians per arm x rate, from the trained models:

| arm | lr | u_ratio = \|z_x + z_y\| / (\|z_x\| + \|z_y\|) | cos(z_x, z_y) | max clock (rad) | increment < 0 |
|---|---|---|---|---|---|
| signed | 1e-3 / 3e-3 | 0.055 / 0.001 | -0.994 / -1.000 | 0.418 / 0.002 | 0.50 / 0.50 |
| **hybrid** | 1e-3 / 3e-3 | **0.043 / 0.002** | **-0.996 / -1.000** | **0.261 / 0.005** | 0.50 / 0.50 |
| \|.\| | 1e-3 / 3e-3 | 0.661 / 0.664 | -0.133 / -0.165 | 5.97 / 6.28 | 0.00 / 0.00 |
| abs_tied | 1e-3 / 3e-3 | 0.000 / 0.000 | -1.000 / -1.000 | 20.6 / 11.5 | 0.00 / 0.00 |

The hybrid reaches the antipodal end state of S2 (u -> 0) exactly as signed does -- the same values as trained signed ring
models elsewhere (0.044-0.053, -0.994; `SEARCH_THEORY.md` Sec. 2) -- with a readout confined to the non-negative orthant (its
effective readout is 0% negative and 100% row-sign-coherent, against 46-52% and 0-11% for every other arm). |.| never approaches it.
abs_tied sits at u = 0 by construction and still carries a large clock: the tie gives it the antipodal latents but not
cancellation, which is the whole point of the control. Descriptive only; no contrast was registered on these numbers.

## Breadth (registered, fixed rates; NOT best-rate claims): exp and softplus vs signed

| lr | arm | SOLVED | acc | d vs signed | label (Holm per test, 4 cells) | qualifier |
|---|---|---|---|---|---|---|
| 1e-3 | exp | 1/8 | 0.935 | -0.065 | SIGNED LEARNS BETTER (Fisher 0.0056, perm 0.0006) | fires on both tests |
| 1e-3 | softplus | 1/8 | 0.879 | -0.122 | SIGNED LEARNS BETTER (Fisher 0.0056, perm 0.0042) | fires on both tests |
| 3e-3 | exp | 1/8 | 0.958 | -0.042 | SIGNED LEARNS BETTER (Fisher 0.0056) | **SOLVED test only** (d above -0.05); 5/7 unsolved runs still DESCENDING, one at held-out 1.000 -- budget-scoped |
| 3e-3 | softplus | 3/8 | 0.966 | -0.034 | SIGNED LEARNS BETTER (Fisher 0.0256) | **SOLVED test only**; 2/5 unsolved runs still DESCENDING -- budget-scoped |

Both arms carry the measured init confound (mean move step at init exp +1.051, softplus +0.706 vs signed +0.003: they start as
RoPE with a rescaled omega, not near NoPE), so a deficit is not attributable to sign alone. Worded "at the two reference rates";
no best rate exists for them (LRSWEEP swept only signed and |.|).

## Checks on the reading

- **Rule 2:** r(log10 final loss, acc) = -0.702 at T = 128 and -0.709 at T = 512 over 96 runs (0 diverged). The primary does not
  depend on it: signed and hybrid are at 1.000 on every seed, so the contrast is SOLVED-vs-not with matching accuracy.
- **Rule 3 / convergence:** `experiment_audit` windowed classes (both rates pooled, n = 16 per arm) agree with the registered
  ones: signed 16 SOLVED, hybrid 16 SOLVED, |.| 2 SOLVED + 14 STALLED (16/16 flat), abs_tied 16 STALLED, exp 2 SOLVED + 8 STALLED
  + 6 DESCENDING, softplus 4 + 9 + 3. The audit's "not all arms converged" flag concerns abs_tied and the breadth arms; the primary
  arms are converged (signed, hybrid) or flat-stalled (|.|). As in LRSWEEP, |.|'s stall is read at this budget.
- **Ceiling (rule 5):** signed and hybrid are both at 1.000 +/- 0.000, so S1-type "no difference" between them cannot be
  expressed as a gap; the registered primary rests on hybrid vs |.|, where it fired on both tests.

## Per seed (accuracy at T = 128 and run class: S SOLVED, T STALLED, D DESCENDING; none diverged)

```
lr     arm       1800   1801   1802   1803   1804   1805   1806   1807
1e-3   signed    1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S
1e-3   abs       0.912T 0.874T 0.902T 0.926T 0.918T 0.968T 0.939T 0.902T
1e-3   hybrid    1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S
1e-3   abs_tied  0.748T 0.748T 0.672T 0.723T 0.749T 0.751T 0.748T 0.770T
1e-3   exp       0.904T 0.973T 0.929T 0.957T 0.852T 0.958T 0.936D 0.972S
1e-3   softplus  0.841T 0.794T 1.000S 0.984T 0.734T 0.861T 0.909T 0.906D
3e-3   signed    1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S
3e-3   abs       0.921T 0.944T 0.935T 0.968T 0.949T 1.000S 0.945T 1.000S
3e-3   hybrid    1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S
3e-3   abs_tied  0.743T 0.748T 0.652T 0.778T 0.775T 0.748T 0.774T 0.769T
3e-3   exp       0.897T 0.959D 1.000S 1.000D 0.902T 0.967D 0.960D 0.977D
3e-3   softplus  0.954T 0.959D 0.999S 1.000S 0.964T 0.975D 1.000S 0.880T
```

**Display correction (post-batch, does not touch the analysis).** The registered printout labelled each run by the first letter
of its class, which collides (SOLVED and STALLED both "S"; DESCENDING printed "D" beside a legend that said "D = DIVERGED"). Fixed
in `analyze_signroute.py` to one distinct letter per class and re-run: `SIGNROUTE_VERDICTS.json` is byte-identical to the
registered output (md5 481d03b7...) and the diff of `SIGNROUTE_ANALYSIS.txt` is confined to the per-seed block. The SOLVED
counts the verdicts use come from a separate field and were never affected.

## What this changes

- Adds to the sign line: the |.| cost is not a cost of non-negativity in the readout matrix. A model whose readout is confined to
  the non-negative orthant but whose increment can be negative learns exactly like signed (16/16), and it does so through the
  collective route S2 describes (antipodal latents, u -> 0).
- Does NOT decide between S3 (no collective route) and S1/P3 (the wrap barrier) as the obstacle for |.|; the hybrid removes both.
  For the clock theory, G-1 (rank) now carries more of the weight; a test that separates S3 from S1/P3 would need a
  non-negative-increment class with a route and no wrap barrier, which S3 rules out for an inverse pair.
- The runbook's "decisive" label for G-2 is withdrawn (Amendment 1).

Not run: the G-2 torus part (frozen move latents, `SEARCH_THEORY.md` 6.2) -- held until this read, per the plan; and a best-rate
sweep for exp / softplus (deferred).
