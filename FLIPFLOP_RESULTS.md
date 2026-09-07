# Flip-Flop LM: the external check does NOT reproduce the recency result

6 arms x 8 seeds, one batch. 4 layers, d=256, 4 heads (CoPE sec 5.1's config for
this task). Chance 0.500; eval scores the FINAL read only, which is what puts every
n-gram order back at chance (`FLIPFLOP_GATES.md`). Error %, lower is better.

| arm | in-dist | OOD dense | OOD sparse |
|---|---|---|---|
| `Signed_r4` | 0.00 | **6.69 ± 3.12** | 26.06 ± 16.43 |
| `Pos_r4` | 0.00 | 7.00 ± 5.83 | 8.84 ± 16.26 |
| `Abs_r4` | 0.00 | 8.09 ± 3.20 | 7.69 ± 14.17 |
| `RoPE` (index) | 0.00 | **8.59 ± 4.54** | 0.13 ± 0.35 |
| `PlainFlat` (index) | 0.00 | 10.84 ± 5.41 | 0.00 ± 0.00 |
| `CARoPE_r4` | 0.03 | 12.16 ± 5.76 | 3.12 ± 7.34 |

| contrast (OOD dense) | delta | MDE | verdict |
|---|---|---|---|
| path-integrated - index | -1.23pp | 3.99 | **unmeasured** |
| monotone - signed | +2.40pp | 3.97 | **unmeasured** |
| every pairwise contrast | -- | 4.2-7.0 | **unmeasured** |

## This was predicted by our own data, before the run

`RECENCY_RESULTS.md` reports index accuracy **by offset**: `k=1` is **0.99**, and
the collapse to 0.11-0.33 begins at `k=8`. **Flip-Flop only ever asks `k=1`** --
"the bit of the most recent write" -- so it is precisely the offset at which an
index code already works. An index code should therefore do fine here, and it does.

So the honest reading is not "the recency result failed to replicate". It is that
**Flip-Flop does not test the property the recency result is about.** The published
benchmark varies the DISTANCE to the governing write (via ignore density) but never
the ORDINAL depth of the query, and it is ordinal depth that separates a
content-gated counter from an index. That is a limitation of the benchmark as an
external check, and it is the reason the graded-k task was built rather than
inherited.

## Two further honest points

**Underpowered, so "unmeasured" and not "null" (rule 11).** Seed sds are 3-6pp
against differences of 1-5pp, giving an MDE around 4pp at n=8. A real effect of
2-3pp would not be visible here.

**OOD-sparse is not a result, it is the shortcut.** The index arms score 0.00-0.13%
there against the signed arm's 26%. The gates already recorded why: at `p_i = 0.10`
the last bit in the stream IS the governing write's bit 95-98% of the time, so
"copy the previous bit" nearly solves the split. The index arms exploit that and
the path-integrated arms do not. Reading it as an index win would be reading a
shortcut. It is reported because it was pre-gated, not because it means anything.

## What it does establish

All six arms reach 0.00% in-distribution error, so **every mechanism here can learn
the write/read relation**; they differ only under distribution shift, and there not
measurably at this power. Flip-Flop is a weaker discriminator than its reputation
suggests for this particular question.
