# Pre-registration: does MapPoPE's Dyck-2 win survive on real code?

Written before any training run. Gates: `CODE_GATES.md` (all pass).
Corpus: `build_code_corpus.py`. Metric: `eval_code_delims.py`. Analysis: `analyze_code.py`.

## The claim under test

MapPoPE is the best arm on Dyck-2 by a wide, detectable margin -- **+0.058 over MapWM
(8/8 seeds, MDE 0.039)** and **+0.312 over PoPE-1L (8/8)** -- and it is the only place in
the project where MapPoPE beats both of its components. Everywhere else on language-like
data it ties (Indirect Indexing at adequate budget) or loses (Bach in distribution,
+0.011 unaugmented and +0.028 augmented, 0/5 seeds).

The proposed reason is that Dyck's positional variable is **signed**: `(` is +1, `)` is
-1, and what matters is net depth. `SIGN_ABLATION.md` measured that a monotone increment
beats an index code NOWHERE, with opposition score 0.11 signed against 1.85-1.98 monotone.
Bach and running text are clock-like -- the accumulator degenerates to a token counter --
which is the standing explanation for why path integration adds nothing there.

**Real code nests.** If the Dyck win is about bracket structure it should survive here.
If it was a property of the synthetic sampler, it should not.

## Why this is a real test and not a rerun of Dyck

Dyck sequences are balanced by construction, uniform in depth, and contentless. Python is
none of those: depth 1 alone is 60.2% of scored positions, closers are often predictable
from local text, and there is semantic content competing for the same capacity. The gates
quantify exactly that -- see the floor below.

## STANDING RISK, stated in advance

Four predictions failed in the 2026-09 line, **each one a generalisation from a
within-task intervention to a cross-task rule** (`T2_RESULTS.md`). This registration is
another such generalisation: Dyck -> code. It is therefore registered with a falsifier
that kills the account rather than qualifying it, and the primary readout is fixed below
before any checkpoint is read.

## The measured floor -- registered, not assumed

From `CODE_GATES.md`, val split, best no-stack predictor per cell (the better of an
order-8 backoff n-gram and the constant `)`):

| depth \ distance | 0-2 | 3-8 | 9-32 | 33-128 | 129+ |
|---|---|---|---|---|---|
| **1** | 1.000 | 0.962 | 0.909 | 0.938 | 0.851 |
| **2** | 1.000 | 0.953 | 0.777 | 0.500 | 0.534 |
| **3-4** | 1.000 | 0.953 | 0.717 | 0.602 | 0.662 |
| **5-8** | -- | 0.833 | **0.499** | **0.272** | -- |

Overall no-stack accuracy is **0.858**; majority class (always `)`) is **0.696**.

**Consequence, registered now: overall accuracy and overall bpc are floor-dominated and
will not be used as the headline.** This is the same defect as Dyck's F1 metric, where an
n-gram scored 0.857 at the hardest cell -- above every index model -- and two batches were
read below their floor before it was caught.

**Cells registered as UNINFORMATIVE in advance** (floor > 0.95, 6 of 18): the whole
distance 0-2 column, and d1/d2/d3-4 at distance 3-8. They will not be read either way.
**Depth 9+ is unmeasurable** (0.3% of positions, every cell under 200 samples).

## Primary readout, fixed before any run

**Closer-identity accuracy at depth 5-8, distance 33-128** (floor 0.272, n=958 scored
positions per val pass): given that the next byte is a closing bracket, does the model
put most mass on the correct one of `)` `]` `}`? Renormalised over the three closers, so
the model is not rewarded for knowing that a bracket closes -- only for knowing WHICH.
A closer is scored only if its matching opener lies inside the same crop; one it never
saw is not a memory test.

Secondary, reported together and never selectively: d5-8/x9-32 (floor 0.499),
d2/x33-128 (0.500), d3-4/x33-128 (0.602), d3-4/x9-32 (0.717), d2/x9-32 (0.777).

## Arms and recipe

The enwik8 2x2, unchanged, so the two corpora are directly comparable: **RoPE**,
**PoPE-Flat** (index), **Vanilla/MapWM r4**, **MapPoPE-Flat r4** (path-integrated).
36k iters, seq 512, batch 16, lr 2e-4, dim 512, 9 layers, r=4, deterministic val.
Every arm trained in one batch. Seed outer, variant inner, so a full low-confidence
table lands before any single arm is finished.

## Predictions

- **P1.** Overall val bpc separates the four arms by less than 0.01, i.e. the headline
  metric carries no signal. (Registering the expected failure of the obvious metric.)
- **P2, PRIMARY.** At d5-8/x33-128, **MapPoPE > PoPE**, sign-consistent across seeds and
  clearing its MDE.
- **P3.** The path-integration main effect (path minus index, averaged over encoding) is
  larger in the low-floor cells than in the high-floor ones.
- **P4.** The MapPoPE-over-PoPE gap is monotone increasing in depth and in distance.

## Falsifiers

- **F1, kills the account.** MapPoPE <= PoPE at *every* depth stratum. Then bracket
  nesting in real code does not recruit the signed accumulator, and the Dyck win is a
  property of the synthetic task. This is the outcome the standing risk above predicts.
- **F2, voids the batch.** No arm clears the measured floor in the informative cells.
  Then the verdict is "unmeasured at this scale" (rule 11), never "null".
- **F3, inverts the metric choice.** Overall bpc separates the arms while the stratified
  metric does not. Then the metric registered here was the wrong one and the bpc reading
  stands instead.
- **F4, rule 9.** If |r(final training loss, primary cell accuracy)| > 0.98 across runs,
  the effect is a convergence gap in disguise and must be reported as one. Loss-matching
  requires overlapping losses; if the arms' final losses do not overlap, no residual will
  be quoted.

## Power

MDE = 2.8 * sd / sqrt(n) on each paired contrast, computed from the realised seed sd and
reported beside every number. A contrast not clearing it is reported as **unmeasured**,
with the MDE stated. n is set after the 1-seed pilot establishes the seed sd; the pilot
itself establishes shape and convergence only and will not be read as a result.

## What would make this void

- Any gate regressing (`validate_code.py` exits non-zero).
- Arms not converged: loss still falling steeply at 36k, checked before reading.
- Any arm trained from different code than the others, or in a different batch.
