# SIGNROUTE (G-2 Part A): is the |.| cost carried by a non-negative READOUT MATRIX, or by the non-negative INCREMENT? -- pre-registration

Registered 2026-10-20, before any batch run; amended the same day by Amendment 1 (the independent results-blind code check,
rule 29), whose fixes are folded into the sections below and listed at the end. Source of the design:
`docs/paper/expansion/SEARCH_THEORY.md` Sec. 6 item 4 (Props. S1, S2, S3 in Sec. 2), scoped by the user on 2026-10-20 (rates,
arms, claim wording; Sec. 1). Code: `model_signroute.py`, `train_signroute.py`, `analyze_signroute.py`, `run_signroute.sh`; unit
checks `test_signroute.py` (12/12), existence `construct_signroute.py` (3/3 + one report), analysis selftest (12 readings + gates,
divergence, breadth equivalence, on-disk loading).

## 1. Question, and what this batch can and cannot decide

LRSWEEP (registered, `LRSWEEP_RESULTS.md`): on the ring the |.| step solves 2/8 against the signed step's 8/8, each at its best
rate. The |.| step makes BOTH the increment and (in angle space) the readout non-negative. The **hybrid** arm separates the two:
a_c(x) = omega_c (|w_c| . z_x) -- readout matrix |W_out| confined to the non-negative orthant, latent signed, so the increment
is signed. Its clock is diag(omega) |W| u with u = z_x + z_y, so the zero-clock subspace {u = 0} of S2 survives with W -> |W|.

- If a non-negative readout matrix were itself costly, the hybrid fails like |.|.
- If what costs is the non-negative INCREMENT, the hybrid learns like signed.

**What it cannot decide (Amendment 1, finding 2).** The hybrid's zero clock sits at k = 0, so it removes both the missing
collective route of S3 and the S1 + P3 wrap barrier (a |.| channel can cancel only at k = 2 pi, across pi from a small start),
which SEARCH_THEORY's own S3 "Reading" calls the LARGER obstruction. Both of those accounts predict a hybrid win. This batch does
not separate S3 from S1/P3; that would need a non-negative-increment class with a route and no wrap barrier, which S3 rules out
for an inverse pair. The rival the hybrid does test -- "non-negativity anywhere in the readout costs" -- is weak: nothing in the
theory predicts a signed-increment model failing like |.|. So this is a **positive control for the S2/S3 framing plus an
in-batch two-rate replication of the sign cost, not a decisive test of S3**. The runbook's "decisive" label is withdrawn for G-2.

User's scoping (2026-10-20), registered verbatim: arms signed, |.|, hybrid, |.|-tied at **both lr 1e-3 and 3e-3**, n = 8, same
seeds (1e-3 = signed's best rate and the default; 3e-3 = |.|'s best rate, pre-empting a wrong-rate objection). **Primary
readout: hybrid vs |.| at each rate** (and, Amendment 1 finding 1, vs signed, so a half-working hybrid cannot read as a
restoration). Claim scoped **"at lr 1e-3 and 3e-3"**, never "at best rates". exp and softplus inside the batch at the same two
rates, reported **"at the two reference rates"** with no best-rate claim; their best-rate sweep is deferred.

## 2. Arms (measured at init at the batch's vocabulary 19, 8 seeds; `test_signroute.py::test_sign_census_at_batch_vocab`)

| arm | step | increment < 0 | effective readout < 0 | rows of diag(omega) W_eff single-signed | role |
|---|---|---|---|---|---|
| signed | W_out W_in x | 50.5% | 50.3% | 11.9% | anchor (G0, G1) |
| abs | \|W_out W_in x\| | 0.0% | 50.3% | 11.9% | anchor (G1) |
| **hybrid** | \|W_out\| (W_in x) | **56.2%** | **0.0%** | **100.0%** | **primary, the only test** |
| abs_tied | \|W_out W_in x\|, z_y := -z_x | 0.0% | 50.3% | 11.9% | control C (tripwire) |
| exp | exp(W_out W_in x) | 0.0% | 50.3% | 11.9% | breadth |
| softplus | softplus(W_out W_in x) | 0.0% | 50.3% | 11.9% | breadth |

(An earlier census at vocab 40 read signed 48.4% / hybrid 40.2% / readout 50.5%; corrected here, finding 12. "Effective
readout" is |W_out| for the hybrid -- its stored W_out is ~50% negative and irrelevant.) omega's sign is free, so the hybrid's
angle-space readout is **row-sign-coherent, not non-negative**; say so wherever the hybrid is described.

All six: observation-step-free (non-move steps exactly 0), rank 4, **209,640 parameters, identical shapes**, identical initial
weights at a given seed (same RNG order; reviewer re-checked). signed and abs are **bitwise-identical** to
`model_wrap_exist.SignedAO_r4` / `AbsAO_r4`. **Do not misquote the hybrid:** a hybrid win licenses "a non-negative readout
matrix is costless; the non-negative increment is what costs". It does NOT license "non-negative steps are fine".

**exp / softplus init confound (measured).** Mean move step at init: exp +1.051, softplus +0.706, signed +0.003, abs +0.236: both
start as RoPE with a rescaled omega rather than near NoPE. A deficit in either is not attributable to the sign alone.

**|W| conditioning (finding 13, NOTE).** At init |W_out|'s normalised singular values are 1 : .31 : .27 : .24 against signed
W_out's 1 : .92 : .85 : .74 (measured, vocab 19, mean over 8 seeds; the reviewer's vocab-40 figures were 1 : .31 : .30 : .26 and
1 : .90 : .88 : .72), which slows u -> 0 along three directions (rule 15). A hybrid failure is read "under this
parameterisation".

## 3. Existence before mechanism (rule 14) -- run before registration

Source: a TRAINED SignedAO_r4, pilot seed 1880 (outside the batch), lr 1e-3, this host, torch 2.10.0+cu128: **acc 1.0000 at
T = 128, 0.9999 at T = 512, final loss 0.0000** (reproduces LRSWEEP's signed anchor).

- **hybrid: a solution EXISTS.** Construction with the source's content and omega, each move's step split into non-negative
  readout columns (orthogonal latents; the construction does not use the route and need not): angles equal to 1.2e-7 rad,
  readout 0% negative (source 52%), weight scale max|w_in| 0.269 (source 0.162), max|w_out| 0.401 (source 0.300).
  **Frozen 1.0000 = source 1.0000.** **Trainable** warm start (Sec. 7): holds 1.0000. The committed
  `construct_signroute.py --save` reproduces the warm-start checkpoint bitwise. A hybrid failure in the batch is therefore a
  SEARCH failure.
- **abs_tied: NO solution exists.** Under the tie the two move steps are equal (evenness), so theta_c =
  omega_c a_c (n_x + n_y): a function of the move count only (it carries displacement PARITY, which cannot bring training loss
  below 0.05 because maps are redrawn per trajectory; reviewer's check). Measured: east and west angles identical to 0.00e+00 rad
  over 8 seeds. **Its failure is representability, not search.** It fails under every account, so it is not half of a decisive
  pair (finding 10); it is a tripwire: any abs_tied SOLVED run means a bug or a proof error.

## 4. Batch

Ring 32 cells, moves +1/-1 (`GridWorldSteps`, p_first 0.5), the LRSWEEP recipe verbatim (`train_gauge`; reviewer found no
difference): T = 128, batch 128, 98 batches, 300 epochs, cosine, AdamW wd 0.05, 1 layer, 2 heads, d 128, rank 4, 3 data workers.
Fresh seeds **1800-1807** (every other "1800" in the repo is an epoch count). 6 arms x 2 rates x 8 seeds = **96 runs**. Pilots on
seeds **1880-1883**, never pooled. Evaluation: `train_cancel.evaluate`, fresh stream (np seed 10^6, env seed 10000, maps
redrawn), 400 trials, T = 128 and 512. Host: 2x RTX 4090, driver 550.144.03 (CUDA 12.4) with torch 2.10.0+cu128 via CUDA
minor-version compatibility (cuBLAS, cuFFT, SDPA, autograd verified on both GPUs; `test_guards` byte-identical to torch 2.6 on its
6 runnable guards). **One GPU** (GPU 1) by the user's instruction, **one job at a time** (measured: 3 concurrent runs 3.1 s/epoch vs
0.8 s alone; ~4.5 min per run, ~7.2 h). The driver refuses modified tracked files AND any uncommitted SIGNROUTE file, checks its
directory tags against the analysis, and md5-guards all 97 imported mapformer modules plus driver, lib and this file, before
training and before the analysis.

## 5. Registered analysis (`analyze_signroute.py`, as amended)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); a run with any non-finite loss is DIVERGED, never SOLVED,
non-finite accuracy counted 0 (declared, as LRSWEEP); every run's recipe and len(losses) = 300 are asserted.
`better(a, b)`: (Fisher < .05 and k_a > k_b) or (permutation p < .05 and acc_a - acc_b >= +0.05) -- the house
SIGNED-LEARNS-BETTER thresholds, a in the reference slot. Per rate, n = 8, signed S, abs A, hybrid H:

- **G0** k_S >= 6, else UNREADABLE at that rate.
- **G1** better(S, A), else NO COST TO EXPLAIN at that rate: the sign cost must be measured in this batch (finding 8).
- **P (primary)**, with full = (k_H >= k_S - 1 and acc_H >= acc_S - 0.01) and likeA = (k_H <= k_A + 1 and acc_H <= acc_A + 0.01):
  **SIGNED INCREMENT RESTORES** if better(H, A) and full; **NON-NEGATIVE READOUT COSTS** if better(S, H) and likeA; **PARTIAL**
  if better(H, A) or better(S, H) but neither of those; **UNMEASURED** otherwise.
- **C (control)** any abs_tied SOLVED run -> every reading prefixed CONTROL BROKEN.

**Overall reading:** UNREADABLE (no rate passes G0 and G1); MIXED (RESTORES at one rate, COSTS at the other); PARTIAL (PARTIAL at
any readable rate, otherwise); SUPPORTED (RESTORES at both rates); SUPPORTED AT ONE RATE (RESTORES at one, the other UNMEASURED,
UNREADABLE or NO COST TO EXPLAIN); REFUTED (COSTS at both); REFUTED AT ONE RATE (COSTS at one, the other UNMEASURED or
unreadable); UNMEASURED. "Supported" means supported for the claim in Sec. 1 (non-negative readout matrix costless), nothing more.

**Breadth:** exp and softplus vs signed per rate with cell_label's thresholds; Holm applied to EACH test separately over the 4
cells (Fisher p's together, permutation p's together), a label kept only if the test that fired survives (finding 6); with raw p's
the rule equals `analyze_robust.cell_label` exactly (selftest). Worded "at the two reference rates".

**Also printed (descriptive):** r(log10 final loss, acc) at T and 4T over finite runs (rule 2); per-seed accuracy and run class
for every arm x rate (rule 6); medians per arm x rate of u_ratio, cos(z_x, z_y), max clock, increment < 0, effective readout < 0,
and row-sign-coherence (finding 7); floors blank ~0.50, n-gram 0.554, retrace 0.860.

**Power (rule 5; measured, `stats_core.fisher_solved`).** With S at 8/8: RESTORES needs H >= 7/8 (full) and, on SOLVED, A <= 2/8
(H 7 vs A 2 p .041; vs A 3 p .119) -- or the accuracy test. COSTS needs H <= 3/8 (S 8 vs H 3 p .026; vs H 4 p .077) and H <= A + 1.
G1 needs A <= 3/8 on SOLVED (8 vs 3 p .026; 8 vs 4 p .077) or the accuracy test. A hybrid between these reads PARTIAL or
UNMEASURED, not SUPPORTED: that is by design. (The pre-amendment claim that 6/8 vs 2/8 "reads UNMEASURED" was wrong under the old
rule, where the accuracy arm could fire alone; finding 1.) Seeds are paired by design (same init and data per seed across arms);
the tests are unpaired, which is conservative (finding 15).

## 6. Predictions

- S2/S3 framing: hybrid ~ signed at both rates (>= 7/8) -> **SUPPORTED**; abs_tied 0/16; abs replicates LRSWEEP (<= 2/8).
- "Readout non-negativity costs": hybrid ~ abs -> **REFUTED**. Not expected by any account in SEARCH_THEORY.
- Breadth: no registered prior beyond "non-negative-increment arms without a route pay a cost like |.|", confounded by their init.

## 7. Pilots (seeds 1880-1883, outside the batch; never pooled)

| pilot | arm, lr | acc T = 128 | acc T = 512 | final loss | u_ratio | cos(z_x, z_y) | note |
|---|---|---|---|---|---|---|---|
| 1880 | signed 1e-3 | 1.0000 | 0.9999 | 0.0000 | -- | -- | the existence source |
| 1881 | hybrid 1e-3, warm start from the construction | 1.0000 | 0.9994 | 0.0000 | 0.119 | -0.972 | rule 14 trainable: holds |
| 1882 | hybrid 1e-3, from scratch | 1.0000 | 0.9994 | 0.0000 | 0.023 | -0.999 | ends antipodal, like trained signed models (0.044-0.053, -0.994) |
| 1883 | abs_tied 3e-3 | 0.7866 | 0.4457 | 0.8512 | 0.000 | -1.000 | not SOLVED; tripwire holds |

One pilot seed per condition: anecdote, not evidence. Pilots used the pre-amendment trainer, which differs only in diagnostics
written to eval.json (no change to training).

## Amendment 1 -- independent results-blind code check (rule 29), 2026-10-20

Reviewer read the prereg, SEARCH_THEORY Sec. 2/6, all batch code and its imports, blind to `runs/`; ran `test_signroute`,
the selftest, and scratch scripts reproducing the Fisher p's and the init census. Found **no difference** between the arms
beyond the intended ones (init RNG order, registration, buffers, requires_grad, dropout, step_table, forward), **no difference**
between `train_signroute` and `train_gauge`'s recipe, and the driver's artifact count, lock, one-GPU dispatch, rules 23 and 27
correct. Findings and dispositions:

| # | sev | finding | disposition |
|---|---|---|---|
| 1 | MAJOR | primary returned ROUTE RESTORES before checking hybrid vs signed; the accuracy arm alone could read a 4/8 hybrid as SUPPORTED, and 3/8 vs 0/8 as REFUTED | FIXED: P now requires better(H, A) AND full for RESTORES, better(S, H) AND likeA for COSTS, else PARTIAL; the reviewer's three cases are selftest cases and read PARTIAL |
| 2 | MAJOR | the hybrid also removes the S1/P3 wrap barrier, so a win does not isolate S3; the "readout" rival is weak | ACCEPTED: label renamed SIGNED INCREMENT RESTORES; Sec. 1 rescoped, "decisive" withdrawn; printed beside the reading |
| 3 | MINOR | clean-tree check passed untracked batch code | FIXED: `git ls-files --error-unmatch` on every SIGNROUTE file and this prereg |
| 4 | MINOR | md5 guard missed environment_cancel / environment_nd / train_variant and 84 imports | FIXED: guard built from sys.modules (97 modules) + driver, lib, prereg |
| 5 | MINOR | diverged runs unhandled (exp is unbounded) | FIXED: DIVERGED class, never SOLVED, acc 0, excluded from rule 2; recipe and len(losses) asserted; selftest case |
| 6 | MINOR | breadth Holm on min(perm, Fisher), not a valid p | FIXED: Holm per test; equality with cell_label on raw p's checked in the selftest |
| 7 | MINOR | S2 diagnostics promised but not printed; SUPPORTED AT ONE RATE broader in code than text; "exactly that" power sentence wrong | FIXED: medians printed; Sec. 5 text matches the code; power sentence rewritten from measured p's |
| 8 | MINOR | G1 (k_A <= 4) passed with no measurable cost | FIXED: G1 = better(S, A) |
| 9 | MINOR | construction docstring stale; an unfailable test counted; no weight-scale report; warm start not reproducible from committed code | FIXED: docstring rewritten; test renamed to a report (3/3 counted); scales reported and bounded; `--save` committed, reproduces the pilot checkpoint bitwise |
| 10 | MINOR | model docstring called abs_tied PRIMARY and the pair decisive | FIXED: abs_tied labelled CONTROL C; "decisive pair" withdrawn in the docstring |
| 11 | NOTE | abs_tied proof correct (parity cannot reach loss 0.05 with redrawn maps); hybrid existence sound | recorded in Sec. 3 |
| 12 | NOTE | "0% negative readout" is about \|W\|; omega's sign is free (row-sign-coherent); census at wrong vocab | FIXED: census at vocab 19 with the effective readout and row-coherence; trainer reports the effective readout |
| 13 | NOTE | \|W\| badly conditioned at init (measured here 1 : .31 : .27 : .24 vs signed 1 : .92 : .85 : .74) | recorded in Sec. 2; a hybrid failure read "under this parameterisation" |
| 14 | NOTE | grad test docstring overclaimed; "final 20%" comment wrong | FIXED both |
| 15 | NOTE | paired design, unpaired tests (conservative); selftest skipped load() and classify_run | recorded; selftest now writes eval.json files in the driver's naming and goes through load() |
