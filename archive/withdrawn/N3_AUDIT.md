# Numbers still cited at n=3

Audit 2026-09-07, prompted by the +0.011 correction (which was itself an n=3
half-table quantity printed as an n=8 main effect). Scope: numbers reachable from
a **live surface** -- `RESULTS_INDEX.md`, `BASELINE_TABLE.md`, `CLAUDE.md`,
`positional_review.tex` -- that were measured at n=3 and never re-measured higher.
Files in `archive/void/` and `archive_stale/` are out of scope.

Rule 6 says three seeds is not a point estimate. This is the list of places that
still depend on one.

## The base rate: what happened every time an n=3 number here was extended

| quantity | at n=3 | extended | change |
|---|---|---|---|
| Match-Query MapWM-Flat | 0.888 +/- 0.140 | **0.730 +/- 0.247** (n=5) | **-0.158** |
| encoding effect, torus (path-int row) | +0.011 | **+0.026** (n=8) | +0.015 |
| MiniGrid hierarchy gain | "18/18, every cell every seed" | **27/32** (n=8) | luck |
| MapWM-Hier compositional | seed1 0.625 vs ~0.30 | high-variance at n=8 | outlier |
| loop vs 3 real layers | +0.273 | underpowered (n=8) | retracted |

Not one held. Two inverted a claim.

---

## 1. Match-Query: five of six arms are n=3, and the sixth already moved

`BASELINE_TABLE.md` section C, `MATCH_QUERY_RESULTS.md`.

Only MapWM-Flat was extended to n=5, and it **fell 0.158** when seeds 3 and 4 came
in at 0.398 and 0.589. The other arms have never had those seeds run:

| arm | TQ=256 | n |
|---|---|---|
| MapWM-Flat | 0.888 -> **0.730** | 3 -> **5** |
| MapPoPE-Hier | 0.847 +/- 0.132 | **3** (per-seed 0.775 / 1.000 / 0.768) |
| Level15 | 0.876 +/- 0.213 | **3** |
| MapEM-os (single p_0) | 0.808 +/- 0.168 | **3** |
| MapWM-Hier | 0.786 +/- 0.227 | **3** |
| MapEM sep-q0/k0 | 0.450 +/- 0.332 | **3** |
| PoPE-Flat | 0.117 +/- 0.011 | **3** |

**This one is live in the presentable document.** `positional_review.tex` writes
the MapWM-Flat figure correctly as `0.730 (n=5)` and then, in the next sentence,
cites **0.847 against 0.117 with no n at all** -- from the same table, at n=3,
as a parallel comparison. The arm beside it lost 0.158 on extension, and
Match-Query is separately documented as not reproducible across batches (the seed
sd there is partly landscape noise, not seed variance).

The *direction* is not in doubt -- 0.847 vs 0.117 is a 7x gap and no path-integrated
seed overlaps an index seed anywhere on this task. The *value* is.

## 2. Family tree: the headline is unmeasured; the claim it displaced is not

`FAMILY_TREE_WM_GAP.md`, n=3 throughout, never extended. Per-seed data is in the
file, so the contrasts can be computed rather than eyeballed:

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| MapWM-Flat − MapEM-NC-NL | +0.077 | 0.068 | 0.111 | 2/3 | unmeasured |
| MapWM-Flat − MapEM-os | +0.090 | 0.065 | 0.106 | 3/3 | unmeasured |
| Level15 − MapWM-Flat | +0.038 | 0.069 | 0.111 | 2/3 | unmeasured |
| MapEM-NC-NL − MapEM-os | +0.013 | 0.005 | 0.008 | 3/3 | **DETECTABLE** |
| MapWM-Flat − Plain-Flat | +0.205 | 0.073 | 0.118 | 3/3 | **DETECTABLE** |

`CLAUDE.md` states flatly: *"the missing plain-WM arm beats every published
variant (0.805 vs MapEM-NC-NL 0.729)"*. That gap is **+0.077 against an MDE of
0.111** -- unmeasured, on 2/3 seeds, and the whole margin is one MapEM seed.

The inversion is worth stating plainly. The claim used to **diminish** the paper's
non-commutativity result is unmeasured; the paper's own axis (+0.013) **is**
detectable, because the EM arms have a seed sd of 0.005. What survives at n=3 is
the contrast this project actually cares about: path integration over an index
code, **+0.205 (3/3, MDE 0.118)**.

Nothing here needs retracting -- but "beats every published variant" needs
n>=8 before it is written again.

## 3. Compositional: the n=3 arms and the n=8 arms are not randomly assigned

`COMPOSITIONAL_MULTISEED.md` line 5 records seeds per variant. Every **flat
MapFormer/PoPE** arm is n=3; every **hierarchy** arm is n=8:

- n=3: `MapWM-Flat` 0.270, `MapEM-Flat` 0.097, `PoPE-Flat` 0.319, `MapPoPE-Flat` 0.366
- n=8: all six hierarchy arms, plus `Plain-Flat` and `MapWM-FlatHG`

So the headline "hierarchy helps" gap -- MapWM-Hier 0.415 (n=8) vs MapWM-Flat
0.270 (n=3) -- compares across seed counts, and the direction of the confound is
the unhelpful one: the n=3 arm has the tight-looking sd (0.030) that three seeds
produce, the n=8 arm the honest one (0.096).

There is a control already in the table that does not have this problem:
**`MapWM-FlatHG`, the parameter-matched flat scaffold, is n=8 and scores 0.285** --
*higher* than the n=3 `MapWM-Flat` it would replace. Against it the gap is +0.130,
both arms at n=8. **Cite that pair instead.** `BASELINE_TABLE.md` section D already
flags this table as "the weakest provenance here"; this names the specific defect.

## 4. Knob sweep: four of seven conditions are n=3 at a budget known to be short

`KNOB_SWEEP.md`, `BASELINE_TABLE.md` section I. Only `baseline`, `rotate` and
`allcombined` were re-run at n=8:

| condition | position effect | n |
|---|---|---|
| baseline | +0.438 | 8 |
| rotate | +0.050 | 8 |
| allcombined | -0.076 | 8 |
| wall | +0.251 | **3** |
| ego | +0.265 | **3** |
| richobs | +0.299 | **3** |
| small | +0.324 | **3** |

Two compounding problems. The n=3 conditions ran at **16 epochs, 1 layer** -- the
exact budget rule 5 was bought with, and `rotate` itself moved **+0.004 -> +0.050**
when that budget was extended. And the surviving claim, *"rotation actions
dominate, at twice the next knob"*, is a **ranking** whose runner-up is an n=3
number at the short budget.

The dominance claim is probably safe -- rotate's -0.388 is larger than the sum of
the other four, so the ordering has room. The margin "twice the next knob" does not.

## 5. Map extent: the endpoint that makes it a threshold is n=3

Cited in `RESULTS_INDEX.md`'s headline, `positional_review.tex` (unlabelled) and
`CLAUDE.md`: *"at matched aliasing it is -0.010 / +0.015 / +0.305 for 32 / 128 /
512 occupied cells -- a threshold, not a gradient."*

| occupied cells | effect | n | sd recorded? |
|---|---|---|---|
| 32 | -0.010 | 3 | no |
| 128 | +0.015 | 3 | no |
| **512** | **+0.305** | **3** | **no** |

The threshold reading rests entirely on the jump from +0.015 to +0.305, and the
+0.305 endpoint is n=3 with **no sd or MDE recorded anywhere in the repo**. The
`t=2.52` endpoint test in `CLAUDE.md` compares an n=5 value (+0.178) against this
n=3 one.

Mitigating: the gap is 0.290, roughly 2x the measured 0.150 noise floor, and the
three sibling conditions in `ALIASING_CONTROLLED.md` do carry sds (0.094 at n=5,
0.031 at n=4). The conclusion is likely safe; the number is not documented to the
standard the rest of the file uses.

## 6. H=12 budget curve at nb=4000 -- already flagged

`H12_BUDGET_CURVE.md`, n=3, bimodal. `BASELINE_TABLE.md` line 330 already says
"needs more seeds before the number is paper-ready". No action; listed for
completeness.

---

## What to do

Ranked by cost-to-fix against exposure:

1. **Label the n in `positional_review.tex`** (items 1 and 5). Free, and it is the
   only presentable document. Two unlabelled n=3 numbers in a document that
   labels n everywhere else.
2. **Swap the compositional citation** to the `MapWM-FlatHG` pair (item 3). Free --
   the n=8 control is already in the table.
3. **Soften two sentences**: "beats every published variant" (item 2) and "twice
   the next knob" (item 4), to what n=3 supports.
4. **Run seeds 3-7 on Match-Query's five unextended arms** (item 1). The only item
   here that costs GPU, and the one with the worst base rate behind it.

Nothing in this list is retracted. Every item is a claim whose *direction* is
supported and whose *value or margin* is not yet measured.
