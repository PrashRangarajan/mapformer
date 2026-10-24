# Is weight decay necessary for shared rank 4 to learn the T = 1024 torus? -- results (registered, `RANKWD_PREREG.md` + Amendment 1)

Batch 2026-10-23 03:00 -> 11:35 on GPU 1 only, code `36f1752` (clean tree, md5 guard over 97 files), torch 2.10.0+cu128,
16 runs: `Vanilla_r4mi` (shared rank 4) at AdamW weight decay 0.05 vs 0, fresh seeds 2100-2107, torus 64 x 64, T = 1024,
900 epochs, lr 1e-3 cosine, the RANK_MI / RANKCLOCK recipe. Independent results-blind check before launch (Amendment 1:
2 MAJOR, 5 MINOR, 3 NOTE, all folded in). Existence checked before launch: a solved rank-4 model HOLDS its solution through
60 epochs at peak lr 1e-3 with weight decay 0 (prereg Sec. 2). Analysis `analyze_rankwd.py` -> `RANKWD_ANALYSIS.txt`,
`RANKWD_VERDICTS.json`; accuracy `RANKWD.md` / `.json`, `RANKWD_WD0.md` / `.json`.

**Reading (registered primary): DECAY NEEDED -- weight decay 0.05 solves 8/8, weight decay 0 solves 1/8 (Fisher p 0.0014).**
This replicates, on fresh seeds and as a registered contrast, G-1's post-hoc 8/8 vs 1/8. **Scope (Amendment 1, finding 1):**
AdamW here decays EVERY parameter, so the reading is credited only to "AdamW decay 0.05 on all parameters, at lr 1e-3 cosine,
within 900 epochs" -- not to the step map, and not to the clock.

> **CORRECTED 2026-10-24 (descriptive section only; the registered reading DECAY NEEDED is unchanged).** The omega norm in
> the table below is NOT gauge-invariant: omega_c and row c of W_out trade scale exactly (rescaling one by c and the other by
> 1/c moves the logits by <= 5e-7 on a scale of 1.4; likewise W_out's columns against W_in's rows -- verified 2026-10-24, found
> by RANKDECAY's independent check). So "omega shrinks to ~0.15 with decay" says where the scale sat, not what the map does.
> The gauge-invariant quantity is the effective angle a move adds per channel, omega_c (W_out W_in e_move)_c, from the same
> snapshots: end / init norm **0.451** (0.374-0.700) with decay vs **9.218** (6.498-14.859) without; the largest angle a single
> move adds ends at **1.85 rad** with decay vs **37.8 rad** without (4.98 at init). The contrast is that WITHOUT decay the
> effective angles grow ~9x -- tens of radians per move, many wraps per cell (rule 8) -- while with decay they stay bounded.
> That replaces the omega-shrink reading below as the lead for a mechanism; it is still descriptive and untested.

## Primary

| arm | SOLVED | classes (S / T / D) | acc T = 512 | acc T = 1024 | acc T = 2048 |
|---|---|---|---|---|---|
| wd 0.05 | **8/8** | 8 / 0 / 0 | 1.000 +/- 0.000 | 0.9995 (1.000 +/- 0.001) | 0.976 +/- 0.023 |
| wd 0 | **1/8** | 1 / 2 / 5 | 0.865 +/- 0.123 | 0.860 +/- 0.124 | 0.846 +/- 0.124 |

G0 passes (wd 8/8 >= 6). P: k_wd0 = 1 <= 2 and Fisher p 0.0014 < .05 -> DECAY NEEDED. Accuracy at T = 1024: 0.9995 vs 0.8603,
unpaired permutation p 0.0014, paired sign-flip p 0.0156 (exact; the smallest p the n = 8 design allows is 0.0078).

Per seed, accuracy at T = 1024 (class): wd 0.998S 1.000S 0.999S 1.000S 1.000S 1.000S 1.000S 1.000S; wd 0 0.853D 0.866D 0.837T
0.952D 0.870D 1.000S 0.920D 0.585T.

## Secondary (S1), as registered -- and, as Amendment 1 said it would be, uninformative here

- **X2 (park vs remove, solved wd-0 runs): X2 UNREADABLE** (1 solved < 4). Amendment 1 finding 2: in the predicted outcome
  this readout cannot be read by construction.
- **X2b ordering: NO ORDER** for both arms (wd 0: no usable runs; wd 0.05: read clock first in 3/4 usable runs, sign p 0.625).
- The wd 0.05 reference (median shrink of the clock 0.202, readgain 0.058 of initial) is **non-diagnostic of removal**: passive
  decay alone gives 0.089 over this schedule.

## Descriptive (not registered as contrasts): what decay does to the parameters

Norm at the end / at initialisation, median (range over 8 seeds); the LayerNorm entry is the RMS gain (init 1):

| | omega | token_emb | W_in | W_out | LayerNorm gains |
|---|---|---|---|---|---|
| wd 0.05 | **0.145** (0.135-0.169) | **0.099** (0.098-0.103) | 4.364 (3.95-4.60) | 0.520 (0.48-0.56) | 0.415 (0.38-0.45) |
| wd 0 | 0.957 (0.93-0.99) | 0.997 (0.99-1.00) | 2.373 (2.11-2.62) | 0.888 (0.85-0.93) | 1.132 (0.65-1.27) |

The ranges do not overlap in any column. Two readings, both descriptive:
- **The token embeddings are almost entirely decayed** (0.099 against the 0.089 passive-decay factor): the gradient barely
  holds them, and W_in grows ~4.4x, roughly compensating in the step map's input.
- **The frequency ladder omega shrinks to ~15% of its initial size with decay and stays at ~96% without it.** The angle a
  move adds is omega times its step, and the step map rescales too (W_in up ~4.4x, W_out and the embeddings down), so omega's
  shrink alone does not fix how many times a channel wraps per cell (rule 8); the effective per-move angles were not measured.
  Still, omega is the parameter decay changes most among those the map is built from, which makes Amendment 1's mechanism (c)
  -- decay pulling omega toward lower frequencies -- the first candidate to test. It is NOT tested here: (a) clock removal,
  (b) weight-norm control of Adam's relative step and (d) the LayerNorm / output gains (shrunk to ~0.4) remain open.

Geometry medians over all runs: u_ratio 0.032 (wd) vs 0.091 (wd 0); clock shrink 0.202 vs 1.357 (the wd-0 latent clock GREW);
readgain 0.060 vs 0.625 (the wd-0 runs still read their clock).

## Pilot (seed 1993, outside the batch, never pooled)

Rank 4 at weight decay 0, 900 epochs: **SOLVED** (final-5% loss 0.0009, acc 1.0000 at T = 1024). It was still training when the
prereg was frozen (prereg Sec. 5). One seed; consistent with the batch's 1/8 -- wd 0 does sometimes find the solution, and the
warm start shows it can hold one -- not evidence against the registered reading.

## Checks on the reading

- **Budget (rules 3, 4):** "within 900 epochs". 5 of the 7 unsolved wd-0 runs are DESCENDING, so wd 0 may be SLOW rather than
  unable; this batch does not separate the two, and the registered claim is the 900-epoch one.
- **Rule 2:** r(log10 final loss, acc) = -0.834 / -0.841 / -0.827 at T = 512 / 1024 / 2048 over 16 runs; the primary is on SOLVED.
- **Within batch:** one GPU, one code version, 16 runs; the two arms share initial weights, data stream and evaluation
  trajectories at each seed (differing only in AdamW's decay term).

## What this changes

- **Every rank-4 success in this repo was trained with weight decay 0.05, and without it rank 4 mostly fails within 900
  epochs.** "Rank 4 learns the T = 1024 torus" should be stated as "rank 4 WITH weight decay 0.05 (AdamW, all parameters)".
  The rank story (rank 2 0/8 vs rank 4 8/8) is unaffected as a comparison -- both arms always had the same decay -- but the
  rank-4 side of it depends on decay.
- G-1's post-hoc observation is now a registered, replicated result (two independent seed sets: 8/8 vs 1/8 both times).
- **Open, and the obvious next test:** which parameters decay must act on. A batch that decays everything EXCEPT omega (and
  one that decays ONLY omega) would test mechanism (c) directly; the descriptive table above is why it is the first candidate.
