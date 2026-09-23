# Indirect Indexing CAN be solved by arithmetic -- existence shown by hand

`hand_indirect.py`. No training: a two-hop attention construction with hand-set weights
and **no per-shift parameters anywhere**, run on examples from the task's own
generator (`environment_indirect.IndirectWorld`, block widened only so long strings fit).

## Why this was needed

Trained PoPE and MapPoPE both memorise the task -- 31 separate lookups, one per shift --
and fall to chance on any shift outside the trained +/-15 (`INDIRECT_OOD.md`). Before
building a model that might learn it properly, check that a lookup-free solution exists
at all. Project rule: existence before mechanism.

## The construction

- **hop 1 (content)**: the answer query attends to the string letter matching the source
  letter; the VALUE returns that letter's POSITION as features `e(p) = (cos p*th, sin p*th)`.
- **shift**: each digit contributes an angle scaled by its place value, and angles ADD --
  rotating by a then by b equals rotating by a+b, so "16" is 16*th exactly.
- **hop 2 (position)**: the query is `e(p)` rotated by the shift angle, i.e. `e(p+k)`;
  string keys are `e(j)`; `score(j) = sum_i cos((p+k-j) th_i)`, peaking exactly at `j = p+k`.

## Result (chance 1/52 = 0.019)

| condition | accuracy |
|---|---|
| shifts +/-15, strings 20-40 (training distribution) | **1.000** (n=4000) |
| shifts \|k\| 16-30, never used anywhere, strings 40-52 | **1.000** (n=1489) |

## Ablations -- which ingredient buys what

| | accuracy |
|---|---|
| **A removed** -- hop 1 returns no position (standard RoPE / MapFormer) | 0.024 -- chance |
| **C replaced** by a lookup table of EVEN shifts | even **1.000**, odd **0.000** -- memorisation's signature |
| **C kept** (additive digit angles), same split | even 1.000, odd 1.000 |
| **B**: best single position offset anchored at the query | at most **0.048** |

B in detail: under MapFormer's placement the query's position is fixed by the prefix, so
a single position kernel can only point a FIXED distance back from the query. The distance
from the end of the string to the target varies 0-39 (sd 8.5 positions), so one fixed
offset is right at most 4.8% of the time -- even with a perfect clock.

## What this establishes, and what it does not

- **A lookup-free solution EXISTS** in a function class with three ingredients:
  **(A)** position readable as a VALUE, **(B)** the second hop's rotation set from what
  the first hop RETRIEVED, **(C)** the shift as additive angles.
- **Neither standard RoPE nor MapFormer has (A) or (B).** Both put position only in the
  query-key rotation, never in values; and MapFormer computes its angle once from token
  identity before any attention runs (`mapformer_math.tex`: "the one axis on which the two
  designs remain uncompared"). So MapWM should fail this task by construction EVEN WITH
  perfect digit arithmetic -- path integration alone is not enough; placement is.
- **(C) is exactly what path integration provides** -- digit tokens as additive angles.
  It is the one ingredient here that MapFormer's mechanism supplies natively.
- **It does NOT show a trained model will FIND this solution.** This project has seen
  existence without learnability (EM recency: 1.000 installed by hand, mostly not found
  by training). That is the next test.

## The next test, designed to avoid the extrapolation trap

Train on EVEN shifts only, test on ODD shifts: held-out values INSIDE the trained range,
so it is interpolation and changes no lengths. A memorising model scores chance on odd
shifts (ablation C above shows the exact signature); an arithmetic one does not. Needs a
model with ingredients A and B -- e.g. per-layer rotation angles computed from the
residual stream (Selective-RoPE placement) plus position features in the values.
