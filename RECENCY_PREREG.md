# Recency (k-back): pre-registration

**Status: pilot run FIRST. This is not a blind pre-registration and is not
labelled as one.** A 12-epoch CPU smoke test of three arms was run to check the
trainer worked, and it showed the direction before this file was written. What
follows pre-registers everything the pilot does NOT settle, with the pilot stated
in full so nobody later mistakes it for the result.

## The pilot, in full

n=1 seed, 12 epochs, batch 8, T=128, CPU, `--n-batches 12`. Chance 0.0625;
most-recent shortcut floor 0.1797.

| arm | acc | nll | per-offset k1..k8 |
|---|---|---|---|
| `Signed_r4` | 0.424 | 1.990 | 0.92 0.57 0.42 0.38 0.19 0.35 0.29 0.28 |
| `Abs_r4` (monotone) | 0.667 | 1.233 | 0.92 0.76 0.67 0.67 0.64 0.53 0.54 0.63 |
| `RoPE` (index) | 0.782 | 1.012 | 0.88 0.71 0.84 0.75 0.66 0.76 0.88 0.81 |

**What this establishes:** the task is learnable, all three arms sit off both the
floor and the ceiling, and the pipeline runs end to end.

**What it does NOT establish: anything numeric.** n=1, a budget far below any
used for a result here, CPU, and a training length a quarter of the intended one.
Standing rules 5 and 6 apply with full force -- this project has had three
successive claims off n<=3 dissolve in a single session, and a 16-epoch budget
produce three false negatives in a single day.

## Pre-registered, and not settled by the pilot

**H1 (the crossover interaction) -- the primary claim.**
`(monotone - signed)` on recency, against the same contrast on the torus, where
it is measured at **-0.215 / -0.280** at T=512/1024 (12/12 seeds, matched loss).
Predicted **positive on recency**, giving a sign-reversing interaction. n=8 in one
batch, arms `Signed_r4` / `Abs_r4` / `Pos_r4` / `CARoPE_r4`, loss-matched per
rule 9 and with r(final loss, acc) reported per length before any residual is
leant on.

**H2 (the more interesting one) -- alpha as a diagnostic.** The unconstrained
`Signed_r4` arm is FREE to learn a monotone code. Predicted: on recency it learns
opposition high and alpha near 0.94, on the torus opposition 0.11 and alpha 0.52
-- i.e. the accumulator exponent measured on a trained model READS OFF what the
task demanded. Measured with `probe_sign.py` / `probe_accumulator.py`. This is
the claim that would make alpha predictive rather than descriptive, and the pilot
says nothing about it.

**H3 (mechanism readout).** Per-offset accuracy should be markedly steeper in k
for signed than monotone, because signed collisions grow with reach. Pre-register
the SLOPE, not the mean: a mean gap with matched slopes would mean the arm is
worse for some other reason. The pilot is consistent with this and is n=1.

**H4 (the full position axis inverts).** Index (`RoPE`, `PlainFlat`) should be at
or above the path-integrated arms here, against +0.461 behind them on the torus.
The pilot's RoPE is the best arm, which if it holds is a stronger statement than
the sign result alone.

## Falsifiers, committed before the batch

- **If `signed >= monotone` on recency at n=8 (loss-matched), the clock/map
  dichotomy is WITHDRAWN from `positional_review.tex` and `mapformer_math.tex`.**
  It would also explain why alpha covers only 2 of the 4 mechanisms it was
  introduced for.
- If every arm lands within the noise floor (0.150), the task is uninformative
  and no reading is taken in either direction -- rule 11, "unmeasured", not
  "null".
- If the per-offset slopes match while the means differ, H3 fails and the mean
  gap is attributed to something else, named before it is explained.

## Checks required before any number is read

1. **Convergence** (rule 10): loss slope over the final 10%, per arm, both
   schedules if any arm is flat. `--schedule cosine` is the default here.
2. **Budget sensitivity** (rule 5): the headline re-run at 2x epochs. A weak
   number at one budget is not a result, and two budget points are not a trend.
3. **MDE** (rule 11) reported beside every contrast, and "unmeasured" used where
   the effect is under it.
4. **One batch** (rule 3): every arm retrained together, never against a stored
   checkpoint.
5. Gates already pass at the default `min_gap = k_max` for T=256..2048
   (`RECENCY_GATES.md`); re-run them if any task parameter changes.

## The ceiling trap, explicitly

The sign ablation's pre-registered discriminator was unmeasurable by construction
-- it asked for a deficit at a training length where the baseline is
1.000 +/- 0.000. Checked here: the pilot's arms span 0.42-0.78 at T=128 with the
floor at 0.18, so every verdict cell can go either way. If the real budget pushes
any arm to ceiling, the primary contrast moves to a longer T where headroom
remains, and that substitution is recorded here rather than chosen afterwards.

---

# AMENDMENT 2026-09-07, before any batch was launched

Four headroom pilots (n=1, 60 epochs) were run to pick a config, as the ceiling
clause above requires. They forced a redesign of the TASK and refuted two of the
four hypotheses above. Recorded here in full, before any multi-seed run, in the
same way the sign ablation's A5 claim was amended at 48/72 checkpoints.

## The task as first built was index retrieval in disguise

| config | Signed_r4 | Abs_r4 | RoPE (index) |
|---|---|---|---|
| k_max=8, T=256 | 1.000 | 1.000 | 1.000 |
| k_max=32, T=512 | 1.000 | 1.000 | 1.000 |
| k_max=64, T=512 | 0.946 | 0.946 | **1.000** |

Everything at ceiling, and where it was not, **RoPE was the best arm and signed
and monotone were EXACTLY TIED**. Diagnosis: `k` counted symbols in a stream
whose only non-symbols were the query pairs themselves, so the token distance
back to the answer was nearly constant and a fixed index offset addressed it
directly. The task never tested contextual counting at all.

**Fix (`p_filler`, default 0.5):** filler tokens are emitted into the stream but
not counted. The token distance to the k-th content symbol becomes a random
variable -- measured at `k=64`, **129.7 +/- 10.3 tokens** (G8) -- so `k` is a
CONTEXTUAL position in CoPE's sense (arXiv:2405.11582, verified in the local
corpus: relative PE's best is "a decaying attention"). Result:

| config (p_filler=0.5) | Signed_r4 | Abs_r4 | Pos_r4 | RoPE |
|---|---|---|---|---|
| k_max=16, T=512 | 1.000 | 1.000 | -- | **0.366** |
| k_max=32, T=512 | 1.000 | 0.966 | -- | **0.315** |
| **k_max=64, T=1024** | **0.610** | **0.531** | **0.309** | -- |

## H4 is REFUTED, with the sign inverted

H4 predicted index would be at or above the path-integrated arms here. With
contextual counting index **collapses** (0.31-0.37 against 1.000). The prediction
was wrong and the reason is instructive: I had reasoned "a clock task is what
index natively encodes", but a CONTEXTUAL clock is precisely what a fixed index
cannot encode. This task is therefore NOT an index-friendly regime and does not
test a position-axis inversion. Withdrawn.

## H1 is MALFORMED, and this is the important one

`Signed_r4` is UNCONSTRAINED: its Delta may take either sign, so the monotone
solution is a special case it contains. `Abs`/`Pos`/`CARoPE` are strict
restrictions of it. Therefore **signed >= monotone on ANY task, up to
optimisation**, and "monotone beats signed on a clock task" -- H1 as written --
can essentially never be true. Every pilot shows exactly this (0.610 vs 0.531;
1.000 vs 0.966; 1.000 vs 1.000). Running it as stated would have produced a null
that means nothing, which is the same defect as the sign ablation's unmeasurable
discriminator, in a different disguise.

**H1 is replaced by the COST OF CONSTRAINING**, which is well posed:

    cost(task) = acc(monotone) - acc(signed),  both trained in one batch

  torus (measured, 12 seeds, loss-matched): **-0.215 / -0.280** at T=512/1024
  recency (predicted): **~0**, i.e. the penalty for giving up cancellation
  disappears when the task does not need it.

The interaction is `cost(recency) - cost(torus)`, predicted POSITIVE. The pilot
gives -0.079 at k_max=64, so a value near zero is not guaranteed and the honest
prediction is "substantially smaller in magnitude than on the torus", tested
against the torus's own seed sd rather than against zero.

## H2 is now the primary claim, not the secondary one

Since the unconstrained arm can adopt either code, the substantive question is
which one it DOES adopt. Predicted: `Signed_r4` trained on recency shows a HIGH
opposition score and alpha near 0.94; the same architecture trained on the torus
shows opposition 0.11 and alpha 0.52. That is the claim that makes alpha a
diagnostic of what a task demanded rather than a description of an architecture,
and no pilot touches it. Measured with `probe_sign.py` / `probe_accumulator.py`
on the trained checkpoints.

H3 (per-offset slope) is unchanged and remains the mechanism readout.

## Falsifiers, restated for the amended hypotheses

- **If `cost(recency)` is statistically indistinguishable from `cost(torus)`,
  the clock/map dichotomy is WITHDRAWN from both documents.** The dichotomy's
  entire content is that the right code depends on the task.
- **If `Signed_r4` learns the SAME accumulator on both tasks** (opposition and
  alpha within their seed sds), H2 fails and alpha is descriptive only -- which
  would be a direct retraction of a claim currently in `positional_review.tex`.
- If every arm lands within the 0.150 noise floor, "unmeasured", not "null".

## Launched configuration

`k_max=64`, `p_filler=0.5`, `n_symbols=16`, `min_gap=64` (the structural
guarantee, kept over the empirically-passing `min_gap=16` because this project
has a long record of empirical passes that later failed), train `T=1024`, eval
`T` in {1024, 2048}, 300 epochs, cosine, lr 1e-3, 1 layer, d=128, 2 heads,
fast-attn, **8 seeds, 6 arms, one batch**. Gates at this exact config:
`RECENCY_GATES_K64.md`, all rows PASS at 800 episodes.
