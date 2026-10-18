# Appendix to "Representable but not found: what path-integrated rotary attention learns, and how it fails"

*Draft 1 (2026-10-18), for `docs/paper/MAIN_DRAFT1.md`. The full theory appendix, with every check script and the
results not used here (P1(c)-(e), P2, P5, P7, Remark R), is `docs/paper/APPENDIX_theory_v3.md`; App. E below states and
proves what the main text uses and gives the v3 label of each statement. Every number is copied from the file named beside
it; a CORRECTED or AUDIT block at the top of a results file supersedes its body. "Registered" = a pre-registered decision or
declared secondary; "(analysis)" = computed after the batch on stored streams or checkpoints, never changing a verdict.
Delete this paragraph before submission.*

Contents. A: models, recipes, batches. B: floors. C: per-seed tables, SOLVED-threshold sensitivity, evaluation scale.
D: constructions. E: theory. F: exchange-rate details. G: rank details. H: sign details. I: CARoPE and CoPE. J: text
world. K: what and where. L: PoPE's score.

---

## A. Models, recipes, batches

### A1. Architecture

| component | setting |
|---|---|
| layers | 1 (Sec. 5: 1-4 real layers and one block looped x4; ROBUST torus: 2 layers, d_model 256) |
| heads, width | 2 heads, d_model 128, d_head 64; feed-forward 4 x d_model with GELU; LayerNorm before attention and before the feed-forward; dropout 0.1 on attention weights, attention output and feed-forward output |
| rotary channels | 32 two-dimensional blocks per head, 64 channels; every query and key coordinate is rotated |
| score | channel c contributes q_c^T R(theta_{t',c} - theta_{t,c}) k_c (content Q and K rotated by the path angle; not additive) |
| step map | Delta = W_out W_in x, W_in: 128 -> r, W_out: r -> 64, no bias; r = 4 shared across heads unless stated (MapFormer's latent is per head); per-head variants: one W_in / W_out block per head; block-diagonal rank 4: the shared rank-4 latent with the off-diagonal blocks of W_out held at zero (each head reads rank 2) |
| frequencies | omega_c = 2 pi (1 / Delta_max)^(i / 31) for block i = 0..31, the same ladder in each head; Delta_max = grid size (torus 64, ring 32, line 256); omega is a free parameter, its sign unconstrained |
| vocabulary | torus: 4 moves + 17 observation tokens (16 types + blank); ring and line: 2 moves + 17 |
| parameters | about 204k for the one-layer torus MapWM; 798,741 for four index layers; 203,925 for the looped block (`FACTORIAL_RESULTS.md`); 1.59M for the ROBUST torus model |

Step parameterisations (`model.py`, `model_sign.py`, `model_wrap_exist.py`): signed (`Signed_r4` = `Vanilla_r4`),
|.| (`Abs_r4`; the absolute value is applied after W_out on each of the 64 channel steps, adding no parameters), softplus
(`Pos_r4`; softplus(0) = 0.693, so every token starts at about the same step), observation-step-free (`SignedAO_r4`,
`AbsAO_r4` on the ring, `SignedAO4_r4`, `AbsAO4_r4` on the torus: every non-move token's step is 0 by construction). Index
RoPE (`RoPE`): canonical ladder base^(-2c / d_head), base 10000.

### A2. Recipes

| | torus (Secs. 4-6, 7) | ring and line (Secs. 5, 6) | Dyck-2 (Sec. 5) |
|---|---|---|---|
| optimiser | AdamW, weight decay 0.05 | AdamW, weight decay 0.05 | AdamW, weight decay 0.01 |
| peak learning rate | 1e-3 | 1e-3 (ROBUST and LRSWEEP vary it) | 1e-4 |
| schedule | 5% linear warm-up, cosine to 10% of the peak | same | same |
| batch | 16 walks of T = 1024 moves (2,048 tokens); T = 128 batches of 128 | 128 walks of T = 128 | 128 sequences |
| budget | 98 batches per epoch, 900 epochs (1800 in Sec. 6.1; 300 at T = 128) | 98 batches per epoch, 300 epochs | 560,000 sequences (1x) or 1,680,000 (3x) |
| loss | cross-entropy on revisited observations | same | next-token cross-entropy on every position |
| training map | one per run | redrawn for every walk | -- |
| evaluation | 100 walks on one held-out map, eval mode | 400 walks, maps redrawn | L32 D12 test set |

Run classes (`stats_core.classify_run`, per-epoch training loss): SOLVED if the mean of the final 5% of epochs is below
0.05; otherwise STALLED if the final-10% mean is within 5% of the preceding 10% mean, else DESCENDING. Continuations of
constructed models (App. D): ring 300 epochs at lr 1e-3 (schedule re-warmed, fresh optimiser state, fresh walk stream);
torus 300 epochs at lr 1e-4, fixed before launch (about the rate at which the source solution was reached).

### A3. Batches

"Code check" = an independent, results-blind code review, whose findings were amended into the pre-registration before
any result was read (before launch, except TEXTWORLD and LEAK, checked while the batch trained): a separately launched read-only AI code-review session given the
pre-registration and the code but no results. "Reproduced" = the analysis was re-run from stored checkpoints by an
independent reproduction audit (`docs/paper/review_draft5/REPRO_AUDIT.md`, `REPRO_AUDIT_nonsign.md`, 2026-10-16) and
matched byte for byte (spot re-evaluations within 7e-4). "Attention": standard, or `--fast-attn` (SDPA; agrees to ~1e-6 in
logits, not bit-identical in training). Pilots ran on seeds outside the batch except where noted.

| batch | main text | task | arms | seeds | epochs | attention | code check | reproduced |
|---|---|---|---|---|---|---|---|---|
| PAPER2X2 | 4 | torus T = 128 | RoPE, PoPE, MapWM r2 / r4, MapPoPE r2 / r4 | 0-7 | 300 | standard | no | yes |
| SIGN_MATCHED | 4, 6.2 | torus T = 1024 | signed, \|.\|, softplus, RoPE | 0-7 | 900 | standard | no (authors' pre-launch gate) | yes |
| GENERALITY | 4, 6.2 | torus; ring p_plus 0.5 | obs.-step-free signed / \|.\| and constructions; signed, CARoPE, CoPE, RoPE | 700-707 | 900 (ring 300); +300 | fast-attn (torus) | yes | yes |
| FACTORIAL | 5 | torus T = 1024 | path 1 layer; index 1, 4 layers, looped x4 | 500-503 (n = 4) | 900 | fast-attn | yes | yes |
| DYCK_MDEPTH | 5 | Dyck-2 L32 D12 | MapWM, MapPoPE (r2), RoPE, PoPE at 1-4 layers; 3x; mixture | 0-7 | 1x / 3x | standard | no | yes |
| CANCEL | 5 | ring p_plus 0.5-1.0 | path 1 layer; index 1-3 layers | 0-7; main text 2-7 (0-1 were the pilot) | 300 | standard | no | yes |
| RANK_MI | 6.1 | torus | shared r2, per-head r2, shared r4 | 0-7 (per-head r2 s0-1 were a pilot) | 900 | standard | no | yes |
| RANK_SEP | 6.1 | torus | block-diagonal r4, per-head r4 | 0-7 | 900 | standard | no | yes |
| RANK3 | 6.1 | torus | per-head r3 | 0-7 | 900 | standard | no | yes |
| LOOP_RANK_E1800 (part 1) | 6.1 | torus | shared r2, shared r4 | 0-7 | 1800 | standard | no | yes |
| RANK_PROJ | 6.1 | torus | projected r2 (frozen, trained), r4 control, r2 continued | 0-7 | +900 | standard | no | yes |
| RANK_ND | 6.1 | 2D (32 per side), 3D (10 per side) tori | per-head rank D, D + 1 | 0-7 | 900 | standard | yes | yes |
| RANK_WRAP | 6.1 | 2D and 3D tori of two sizes | per-head rank D + 1 | 0-7 | 900 | standard | yes | yes |
| RANK_TESTS | 6.1 | ring; torus escape test | rank 1, 2 (+ variants); lifted / fresh rank 4 | 20-27; 0-7 | 300; 600 | standard | yes | yes |
| RING_DIAL | 6.2 | ring p_plus 0.5 / 0.75 / 0.9 | signed, \|.\| | 30-37 | 300 | standard | yes | yes |
| WRAP_EXISTENCE | 6.2 | ring p_plus 0.5 | obs.-step-free signed / \|.\|, constructions | 600-607 | 300 + 300 | standard | yes | yes |
| LRSWEEP | 6.2 | ring | obs.-step-free signed / \|.\| at 1e-3 / 3e-3 / 1e-2 / 3e-2 | 1500-1507 | 300 | standard | yes | no (windowed run classes cross-checked) |
| ROBUST | 6.2 | ring (four recipes); torus 2 layers d 256 | obs.-step-free signed / \|.\| | 1400-1407 | 300; 900 | fast-attn (torus) | yes | yes |
| ACCOUNTS | 6.2 | ring; line | T1 continuations; T2 frozen latents; T4 CoPE on a line | 600-607; 1000-1007; 1100-1115 | 300 | standard | yes | yes |
| MECHANISM | 6.2 | ring p_plus 0.5 | eight start / step arms | 800-807 | 300 | standard | yes | yes |
| TEXTWORLD | 7 | torus in words | path 1 layer; index 1, 2 layers | 0-7; main text 2-7 (0-1 were the pilot) | 900 | standard | yes | yes |
| RING_BUDGET (H) | -- | ring p_plus 0.5 | \|.\| at 300 and 900 epochs | 40-47 | 300 / 900 | standard | yes | yes |
| TORUS_START (H) | -- | torus | start arms | 1200-1207 | 900 | fast-attn | yes | yes |
| GAUGE (H) | -- | ring, moves +1/-1 or +1/+2 | obs.-step-free signed / \|.\|, RoPE | 1300-1307 | 300 | standard | yes | yes |
| WRAP_COPE (I) | -- | ring | CoPE table starts | 900-907 | 300 | standard | yes | yes |
| LEAK (K) | -- | new-object torus | MapWM, ActOnly, NormStep | 0-7 | 900 | standard | yes | yes |
| GAIN_PHASE (K) | -- | new-object torus | step x score 2x2 | 8-15 | 900 | standard | yes | no |
| MAPPOPE_PAIR (L) | -- | torus T = 128 | MapWM r2, MapPoPE r2 (32 / 64 angles) | 10-25 | 300 | standard | yes | no |

ACCOUNTS T1's weight-decay-0.05 arms are bit-identical retrains of WRAP_EXISTENCE's constructed arms (a second readout of the
same runs, not a replication). The shared rank-2 arm of every rank contrast is one stored set of eight runs.

### A4. Reproduction of MapFormer v4 Table 2 (2D)

With the paper's recipe (16 epochs x 98 batches, linear decay, lr 3e-4; `python3 -m mapformer.main --device cuda --epochs
16 --n-batches 98`), our MapWM scores 0.989 +/- 0.010 and our MapEM (single p_0) 0.987 +/- 0.012 on fresh maps at T = 128
(`PAPER_TASK_ACCURACY.md`), against 1.00 / 1.00 in v4 and 0.99 / 1.0 in v1 (MapWM-r2, MapEM-os, 2D IID).

---

## B. Floors

A floor is the accuracy of the best map-free predictor on the same kind of evaluation walk. Rows marked (analysis) were
computed after the batches.

| task | predictor | value | stream | source |
|---|---|---|---|---|
| torus T = 128 (analysis) | blank / n-gram (orders 0-5) / retrace | 0.507 / 0.598 / 0.934 (retrace per seed 0.928-0.946); wrap-only share 0.000 | PAPER2X2 eval walks, seeds 0-7 | `docs/audits/2026-10-16/paper2x2_T128_floors_out.txt` |
| torus T = 1024 (analysis) | blank / n-gram / retrace | 0.506 / 0.575 / 0.843 (retrace per seed 0.837-0.849); wrap-only share 0.079 | SIGN_MATCHED eval walks, seeds 0-7 (shared by RANK_MI, RANK_SEP, RANK3) | `docs/audits/2026-10-16/sign_floors_out.txt` |
| torus T = 1024 (analysis) | same | 0.508 / 0.578 / 0.843; wrap-only 0.081 | GENERALITY, seeds 700-707 | `generality_floors_out.txt` (same directory) |
| torus T = 1024, 2 layers d 256 (analysis) | same | 0.509 / 0.578 / 0.844; every run of both arms above retrace (lowest \|.\| 0.910) | ROBUST, seeds 1400-1407 | `robust_floors_out.txt` |
| torus T = 1024 (analysis) | blank / retrace | 0.509 / 0.844 | FACTORIAL, seeds 500-503 | `docs/paper/figs/fig_factorial_strata_out.txt` |
| ring p_plus 0.5 / 0.75 / 0.9 (analysis) | retrace | 0.860 / 0.758 / 0.612 (blank 0.497-0.499) | 2000 walks per bias | `docs/audits/2026-10-16/ring_floors_out.txt` |
| ring p_plus 0.5 / 0.75 / 0.9 / 1.0 | better of best constant and n-gram (orders 1-5) | 0.554 / 0.544 / 0.520 / 0.498 | 1000 held-out walks | `CANCEL_PREREG.md`, `CANCEL_RESULTS.md` |
| line (256 cells) | blank | 0.503 | ACCOUNTS T4 | `ACCOUNTS_RESULTS.md` |
| ring, moves +1/+2 | blank (retrace reduces to it) | ~0.50 | GAUGE | `GAUGE_RESULTS.md` |
| torus in words | best constant / reversal-copy | 0.505 / 0.597 | TEXTWORLD eval set | `TEXTWORLD_RESULTS.md` |
| Dyck-2 L32 D12 | best stack-free A2f floor (n-grams of orders 1-6) | 0.594 | gate G4 | `DYCK_MDEPTH_RESULTS.md` |

**Per-stratum floors (analysis).** Torus, seeds 0-7 (`docs/paper/figs/torus_strata_floors_s0-7.json`): best constant
0.506 / 0.500 / 0.507 (short gap / long gap / wrap-only), retrace 0.911 on short gaps and equal to the constant on the
others (walk runs are at most 10 moves, so a retrace never reaches a gap of 128 moves or a wrap-only revisit). FACTORIAL's
stream: 0.508 / 0.515 / 0.517, retrace 0.912. Ring (`ring_strata_floors_out.txt`): retrace 0.9998 at gaps 2-7, 0.943 at
8-31, 0.570 at 32-127, 0.493 wrap-only; constant 0.49-0.50 in every stratum.

**Against its own seed's retrace floor** (analysis, torus T = 1024): signed falls below on 0/8 runs, |.| on 3/8 (0.637,
0.690, 0.717), softplus on 4/8, index RoPE on 8/8.

---

## C. Per-seed tables

Accuracy at the training length on held-out maps, eval mode. SOLV / STAL / DESC = SOLVED / STALLED / DESCENDING; a number
in brackets is the final-5% training loss.

### C1. Torus sign batch (Secs. 4, 6.2; `SIGN_MATCHED_ANALYSIS.txt`)

| seed | signed | \|.\| | softplus | index RoPE |
|---|---|---|---|---|
| 0 | 1.000 SOLV (0.003) | 0.931 DESC (0.201) | 0.590 STAL (1.250) | 0.730 STAL (0.983) |
| 1 | 0.997 SOLV (0.003) | 0.917 DESC (0.314) | 0.527 STAL (1.510) | 0.729 STAL (0.939) |
| 2 | 1.000 SOLV (0.005) | 0.868 STAL (0.492) | 0.630 STAL (1.084) | 0.720 STAL (0.931) |
| 3 | 1.000 SOLV (0.008) | 0.637 STAL (1.043) | 0.972 DESC (0.158) | 0.739 STAL (0.910) |
| 4 | 1.000 SOLV (0.004) | 0.690 DESC (0.962) | 0.674 STAL (1.009) | 0.737 STAL (0.928) |
| 5 | 0.990 SOLV (0.003) | 0.888 STAL (0.446) | 0.935 DESC (0.295) | 0.744 STAL (0.885) |
| 6 | 1.000 SOLV (0.004) | 0.922 STAL (0.349) | 0.922 DESC (0.345) | 0.722 STAL (0.956) |
| 7 | 1.000 SOLV (0.004) | 0.717 STAL (0.834) | 0.995 SOLV (0.049) | 0.728 STAL (0.976) |
| mean | 0.998 | 0.821 | 0.781 | 0.731 |

Registered: |.| - signed -0.177 (permutation p 0.0002; 0/8 vs 8/8, Fisher p 0.0002); softplus - signed -0.218 (p 0.0003;
1/8 vs 8/8, p 0.0014); signed - RoPE +0.267 (p 0.0002; 8/8 vs 0/8, p 0.0002). r(final loss, accuracy) -0.992 over 32 runs.

### C2. Torus, index depth (Sec. 5; `FACTORIAL_{L1,L4,LOOP}.json`)

| seed | path, 1 layer | index, 1 layer | index, 4 layers | index, looped x4 |
|---|---|---|---|---|
| 500 | 1.000 | 0.719 | 0.814 | 0.868 |
| 501 | 1.000 | 0.731 | 0.870 | 0.861 |
| 502 | 1.000 | 0.688 | 0.874 | 0.860 |
| 503 | 1.000 | 0.739 | 0.881 | 0.868 |

Every path run SOLVED, every index run STALLED. Registered C0-C3: path beats index at one layer (+0.281), four layers
(+0.141) and loop x4 (+0.136); depth and loop each help index (+0.140, +0.145); every firing a complete separation
(p 0.029).

### C3. Ring exchange rate, fresh seeds 2-7 (Sec. 5; `docs/audits/2026-09-27/fresh_seeds.txt`)

| p_plus | path 1 layer | index 1 layer | gap to path at 1 / 2 / 3 index layers |
|---|---|---|---|
| 0.5 | 1.000 | 0.710 +/- 0.020 | +0.290 / +0.051 / +0.002 |
| 0.75 | 1.000 | 0.677 +/- 0.009 | +0.323 / +0.038 / +0.002 |
| 0.9 | 1.000 | 0.824 +/- 0.007 | +0.176 / +0.013 / +0.001 |
| 1.0 | 1.000 | 1.000 | 0 / 0 / 0 |

All eight seeds (`CANCEL_RESULTS.md`): index 1 / 2 / 3 layers 0.717 / 0.954 / 0.998 (p_plus 0.5), 0.676 / 0.969 / 0.998
(0.75), 0.825 / 0.988 / 0.999 (0.9), 1.000 (1.0); path 1.000 in every cell (32/32 SOLVED); registered exchange rate k = 3,
3, 3, 1 index layers.

### C4. Rank (Sec. 6.1; `RANK_MI_ANALYSIS.txt`, `RANK_SEP_ANALYSIS.txt`, `RANK3_ANALYSIS.txt`, `RANK_SEP.json`, `RANK3.json`)

| seed | shared r2 | per-head r2 | block-diag r4 | per-head r3 | shared r4 | per-head r4 |
|---|---|---|---|---|---|---|
| 0 | 0.849 STAL (0.587) | 0.656 STAL (1.049) | 0.879 STAL (0.482) | 0.942 DESC (0.243) | 1.000 SOLV (0.004) | 0.996 SOLV (0.004) |
| 1 | 0.998 DESC (0.051) | 1.000 SOLV (0.007) | 0.936 DESC (0.302) | 0.990 SOLV (0.003) | 1.000 SOLV (0.003) | 1.000 SOLV (0.010) |
| 2 | 0.917 DESC (0.308) | 1.000 SOLV (0.003) | 1.000 SOLV (0.010) | 1.000 SOLV (0.003) | 1.000 SOLV (0.003) | 1.000 SOLV (0.004) |
| 3 | 0.917 DESC (0.321) | 0.723 STAL (0.910) | 0.945 DESC (0.234) | 0.967 DESC (0.134) | 1.000 SOLV (0.003) | 1.000 SOLV (0.004) |
| 4 | 0.947 DESC (0.177) | 0.959 DESC (0.194) | 0.964 DESC (0.202) | 1.000 SOLV (0.003) | 1.000 SOLV (0.002) | 0.998 SOLV (0.002) |
| 5 | 0.907 STAL (0.404) | 0.949 DESC (0.211) | 0.917 DESC (0.310) | 1.000 SOLV (0.006) | 0.994 SOLV (0.004) | 1.000 SOLV (0.008) |
| 6 | 0.773 STAL (0.675) | 0.943 DESC (0.248) | 1.000 SOLV (0.010) | 1.000 SOLV (0.003) | 1.000 SOLV (0.004) | 1.000 SOLV (0.005) |
| 7 | 0.842 STAL (0.605) | 0.851 STAL (0.539) | 0.947 DESC (0.166) | 1.000 SOLV (0.005) | 0.987 SOLV (0.010) | 1.000 SOLV (0.003) |
| mean | 0.894, 0/8 | 0.885, 2/8 | 0.948, 2/8 | 0.987, 6/8 | 0.998, 8/8 | 0.999, 8/8 |

Per-head r2 seeds 0-1 were a pilot run before registration (1/6 SOLVED on seeds 2-7). Registered: per-head rank 4 vs
block-diagonal 8/8 vs 2/8 (Fisher and permutation p 0.0070); shared r4 vs block-diagonal the same (it also changes
per-head rank); sharing (per-head r4 vs shared r4, +0.001) and W_out scale (block-diagonal vs per-head r2, +0.063) not
detected; per-head r3 vs r2 +0.102 (permutation p 0.027; SOLVED 6/8 vs 2/8, Fisher p 0.13); r4 vs r3 +0.012, not
detected. Shared r4 vs shared r2: 8/8 vs 0/8 (Fisher p 0.0002), +0.104 (p 0.0005).

**Twice the budget** (`LOOP_RANK_E1800_P1_ANALYSIS.txt`, 1800 epochs from scratch): shared r2 0/8 (final-5% losses
0.489, 0.383, 0.390, 0.340, 0.141, 0.231, 0.305, 0.571; 3 STALLED, 5 DESCENDING), accuracy 0.908; shared r4 7/8 (s5
STALLED at 0.180), 0.994.

### C5. Sign, ring and torus (Sec. 6.2)

Ring dial (`RING_DIAL_VERDICTS.json`; full models, S = SOLVED):

| seed | 0.5 signed | 0.5 \|.\| | 0.75 signed | 0.75 \|.\| | 0.9 signed | 0.9 \|.\| |
|---|---|---|---|---|---|---|
| 30 | 1.000 S | 1.000 S | 1.000 S | 0.994 S | 1.000 S | 1.000 S |
| 31 | 1.000 S | 0.977 | 1.000 S | 0.958 | 1.000 S | 0.999 S |
| 32 | 1.000 S | 0.987 | 1.000 S | 0.957 | 1.000 S | 0.885 |
| 33 | 1.000 S | 0.964 | 1.000 S | 0.954 | 1.000 S | 1.000 S |
| 34 | 1.000 S | 1.000 S | 1.000 S | 0.964 | 1.000 S | 1.000 S |
| 35 | 1.000 S | 0.944 | 1.000 S | 0.967 | 1.000 S | 0.825 |
| 36 | 1.000 S | 0.977 | 1.000 S | 0.993 | 1.000 S | 0.960 |
| 37 | 1.000 S | 0.931 | 1.000 S | 0.958 | 1.000 S | 0.964 |

Registered: p_plus 0.5 -0.028 (permutation p 0.007), 2/8 vs 8/8 (Fisher p 0.0070); 0.75 -0.032 (p 0.0002), 1/8 vs 8/8
(p 0.0014); at 0.9 the contrast did not reach the registered threshold (-0.046, below the 0.05 materiality; 4/8 vs 8/8,
p 0.077).

Observation-step-free ring from scratch and constructed (`WRAP_EXISTENCE_ANALYSIS.txt`):

| seed | signed | \|.\| | constructed \|.\|: frozen -> trained (max continuation loss) | constructed signed: frozen -> trained |
|---|---|---|---|---|
| 600 | 1.000 S | 0.905 STAL | 1.000 -> 1.000 S (0.041) | 1.000 -> 1.000 S (0.019) |
| 601 | 1.000 S | 0.949 STAL | 1.000 -> 1.000 S (0.030) | 1.000 -> 1.000 S (0.042) |
| 602 | 1.000 S | 0.853 STAL | 1.000 -> 1.000 S (0.154) | 1.000 -> 1.000 S (0.072) |
| 603 | 1.000 S | 0.951 STAL | 1.000 -> 1.000 S (0.046) | 1.000 -> 1.000 S (0.017) |
| 604 | 1.000 S | 0.907 STAL | 1.000 -> 1.000 S (0.024) | 1.000 -> 1.000 S (0.039) |
| 605 | 1.000 S | 0.930 STAL | 1.000 -> 1.000 S (0.032) | 1.000 -> 1.000 S (0.028) |
| 606 | 1.000 S | 0.950 STAL | 1.000 -> 1.000 S (0.031) | 1.000 -> 1.000 S (0.112) |
| 607 | 1.000 S | 0.922 STAL | 1.000 -> 1.000 S (0.036) | 1.000 -> 1.000 S (0.015) |

Registered: from scratch 0.921 vs 1.000, -0.079 (permutation p 0.0002), 0/8 vs 8/8 (Fisher p 0.0002); constructed |.|
HELD (8/8 SOLVED, 0/8 LOST).

Observation-step-free torus (`GENERALITY_ANALYSIS.txt`):

| seed | signed | \|.\| | constructed signed (lr 1e-4) | constructed \|.\| (lr 1e-4): peak loss, epoch of peak / of re-entry |
|---|---|---|---|---|
| 700 | 1.000 S | 0.874 STAL | 1.000 S | 1.000 S; 0.168, 29 / 153 |
| 701 | 1.000 S | 0.707 STAL | 1.000 S | 1.000 S; 0.110, 40 / 121 |
| 702 | 1.000 S | 0.946 DESC | 1.000 S | 1.000 S; 0.173, 19 / 204 |
| 703 | 1.000 S | 0.879 STAL | 1.000 S | 1.000 S; 0.143, 52 / 149 |
| 704 | 1.000 S | 0.892 STAL | 1.000 S | 1.000 S; 0.131, 13 / 119 |
| 705 | 1.000 S | 0.907 STAL | 1.000 S | 1.000 S; 0.155, 13 / 148 |
| 706 | 1.000 S | 0.877 STAL | 1.000 S | 1.000 S; 0.208, 36 / 164 |
| 707 | 1.000 S | 0.960 DESC | 1.000 S | 1.000 S; 0.117, 23 / 139 |

Registered: from scratch 0.880 vs 1.000, -0.120 (p 0.0002), 0/8 vs 8/8; the constructed |.| model leaves the solved
region on 6/8 seeds (peak loss 0.11-0.21; signed control 0.004-0.017) and ends SOLVED on 8/8 at 1.000: by the reading fixed
before launch, perturbed and re-found.

Recipe and scale (`ROBUST_ANALYSIS.txt`, seeds 1400-1407; accuracy at the training length, class):

```
ring  seed | def:SO | def:AO | wd0:SO | wd0:AO | lrx3:SO | lrx3:AO | lrd3:SO | lrd3:AO
1400 | 1.0000 SOLV | 0.8230 STAL | 1.0000 SOLV | 0.9078 DESC | 1.0000 SOLV | 1.0000 DESC | 1.0000 SOLV | 0.8893 STAL
1401 | 1.0000 SOLV | 0.9155 STAL | 1.0000 SOLV | 0.6239 STAL | 1.0000 SOLV | 0.9596 STAL | 0.6194 STAL | 0.6221 STAL
1402 | 1.0000 SOLV | 0.9237 STAL | 1.0000 SOLV | 0.8878 STAL | 1.0000 SOLV | 0.9702 STAL | 1.0000 SOLV | 0.8319 STAL
1403 | 1.0000 SOLV | 0.9136 STAL | 1.0000 SOLV | 0.8800 STAL | 1.0000 SOLV | 0.9995 SOLV | 1.0000 SOLV | 0.9167 STAL
1404 | 1.0000 SOLV | 0.9118 STAL | 1.0000 SOLV | 0.9119 STAL | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.8313 STAL
1405 | 1.0000 SOLV | 0.8791 STAL | 1.0000 SOLV | 0.8858 STAL | 1.0000 SOLV | 0.9650 STAL | 1.0000 SOLV | 0.8526 STAL
1406 | 1.0000 SOLV | 0.9211 DESC | 1.0000 SOLV | 0.9110 STAL | 1.0000 SOLV | 1.0000 SOLV | 0.9794 STAL | 0.9062 STAL
1407 | 1.0000 SOLV | 0.9997 SOLV | 1.0000 SOLV | 0.9735 DESC | 1.0000 SOLV | 0.9998 SOLV | 1.0000 SOLV | 0.8801 STAL
torus 1400 SO 1.0000 SOLV | AO 0.9941 DESC     1404 SO 1.0000 SOLV | AO 0.9935 SOLV
      1401 SO 1.0000 SOLV | AO 0.9981 SOLV     1405 SO 1.0000 SOLV | AO 0.9706 DESC
      1402 SO 0.9941 SOLV | AO 0.9705 DESC     1406 SO 1.0000 SOLV | AO 0.9098 DESC
      1403 SO 1.0000 SOLV | AO 0.9454 DESC     1407 SO 1.0000 SOLV | AO 0.9544 DESC
```

Learning-rate sweep (`LRSWEEP_RESULTS.md`, seeds 1500-1507): per-seed table in H2.

### C6. Text world (Sec. 7; `TEXTWORLD_ANALYSIS.txt`)

| seed | path, 1 layer | index, 1 layer | index, 2 layers |
|---|---|---|---|
| 0 (pilot) | 0.881 | 0.504 | 0.695 |
| 1 (pilot) | 0.998 | 0.504 | 0.723 |
| 2 | 0.999 | 0.503 | 0.927 |
| 3 | 0.995 | 0.506 | 0.722 |
| 4 | 0.999 | 0.505 | 0.926 |
| 5 | 1.000 | 0.506 | 0.706 |
| 6 | 1.000 | 0.506 | 0.760 |
| 7 | 0.882 | 0.504 | 0.720 |

Fresh seeds 2-7 (`docs/audits/2026-09-27/fresh_seeds.txt`): path 0.979 +/- 0.048 (6/6 SOLVED), index 1 layer 0.505
(0/6), index 2 layers 0.793 (0/6); path - index one layer +0.474, permutation and Fisher p 0.0022. All eight seeds: path
0.969 +/- 0.054 (7/8; seed 0 still descending), +0.464 (p 0.0002; Fisher p 0.0014); path - index two layers +0.197
(p 0.0009).

### C7. Dyck-2

Per-seed values are in `DYCK_MDEPTH_RESULTS.json` (cells `T12|{RoPE,PoPE,MapWM,MapPoPE}|{1-4}L|L32D12`); every one of the
eight seeds has a positive position effect at every depth (sign-flip p 0.0078 per depth).

### C8. SOLVED-threshold sensitivity

Sign batches (`docs/audits/2026-10-15/reanalyses/solved_thresholds_out.md`; predictions and decision rules written before
the re-analysis; two-sided Fisher p in brackets):

| batch | contrast | loss < 0.02 | loss < 0.05 (registered) | loss < 0.1 | accuracy >= 0.99 |
|---|---|---|---|---|---|
| SIGN_MATCHED torus | \|.\| vs signed | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 7/8 (0.0014) |
| SIGN_MATCHED torus | softplus vs signed | 0/8 vs 8/8 (0.0002) | 1/8 vs 8/8 (0.0014) | 1/8 vs 8/8 (0.0014) | 1/8 vs 7/8 (0.0101) |
| RING_DIAL p 0.5 | \|.\| vs signed | 2/8 vs 8/8 (0.0070) | 2/8 vs 8/8 (0.0070) | 4/8 vs 8/8 (0.0769) | 2/8 vs 8/8 (0.0070) |
| RING_DIAL p 0.75 | \|.\| vs signed | 1/8 vs 8/8 (0.0014) | 1/8 vs 8/8 (0.0014) | 2/8 vs 8/8 (0.0070) | 2/8 vs 8/8 (0.0070) |
| WRAP_EXISTENCE ring | \|.\| vs signed | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) |
| MECHANISM ring | \|.\| vs signed, same-direction start | 0/8 vs 7/8 (0.0014) | 0/8 vs 7/8 (0.0014) | 0/8 vs 7/8 (0.0014) | 0/8 vs 7/8 (0.0014) |
| ACCOUNTS T2 ring | frozen vs free move latents | 1/8 vs 7/8 (0.0101) | 1/8 vs 7/8 (0.0101) | 1/8 vs 7/8 (0.0101) | 1/8 vs 7/8 (0.0101) |
| GENERALITY torus | \|.\| vs signed | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) |
| GENERALITY ring | CoPE vs signed | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 1/8 vs 8/8 (0.0014) | 0/8 vs 8/8 (0.0002) |

ROBUST (declared secondary, loss < 0.1): ring default 1/8 vs 8/8 (p 0.0014), weight decay 0 1/8 vs 8/8 (0.0014), lr / 3
0/8 vs 7/8 (0.0014), lr x3 4/8 vs 8/8 (0.077), torus 2 layers d 256 5/8 vs 8/8 (0.20); on that torus cell the accuracy
contrast (0.967 vs 0.999, p 0.0005) carries the result. LRSWEEP (loss < 0.1, selection redone): 2/8 vs 8/8, same label.

Rank batches (analysis; read from the per-seed final-5% losses in C4): at loss < 0.02 the counts are 0 / 2 / 2 / 6 / 8 /
8 of 8 (shared r2, per-head r2, block-diagonal r4, per-head r3, shared r4, per-head r4), identical to the registered
0.05; at loss < 0.1 they are 1 / 2 / 2 / 6 / 8 / 8 (shared r2 seed 1 at 0.051 enters).

### C9. Attention-dropout evaluation scale

Models trained with attention dropout expect the 1/(1 - p) attention scale; evaluated with it, below-ceiling one-layer
runs score higher. The registered text-world evaluation re-run with the scale (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`,
all 24 runs) gives path 0.969 -> 0.993, index one layer 0.505 -> 0.506, index two layers 0.772 -> 0.772; path - index one
layer +0.464 -> +0.488 (p 0.0002). The re-score was a declared secondary of RING_DIAL, RING_BUDGET, WRAP_EXISTENCE,
GENERALITY, MECHANISM and WRAP_COPE and changed no label there; it was applied to SIGN_MATCHED by the same audit with no
verdict changed. ACCOUNTS, TORUS_START, GAUGE and ROBUST were not re-scored (the correction does not apply at two layers).
All main-text accuracies are eval mode without the scale.

---

## D. Constructions

### D1. The rank-2 projection (Sec. 6.1; `RANK_PROJ_RESULTS.md`)

Each solved shared rank-4 model (seeds 0-7, a stored batch) is projected onto the top two directions of its action latent
over the vocabulary (top-2 spectral energy 0.99952-0.99994), every other weight kept, and installed in the shared rank-2
architecture. Frozen, the projections score 0.9955 at T = 1024 against their sources'
0.9966 (at least 0.981 on every seed; per seed 0.9993, 1.0000, 0.9988, 1.0000, 0.9850, 0.9806, 1.0000, 1.0000). Trained
900 epochs from a learning-rate restart at 1e-3, 7/8 are SOLVED (one DESCENDING), as for the rank-4 control trained from
its own solution (7/8); rank 2 continued from its own 900-epoch weights solves 1/8. Registered verdict: STABLE.

### D2. Per channel (Lemma P1.0 in v3)

For a signed step Delta and omega != 0, Delta' = Delta mod (2 pi / |omega|) lies in [0, 2 pi / |omega|) and
omega Delta' = omega Delta (mod 2 pi) for either sign of omega. The model reads each channel's angle only through cos and
sin, so the model with Delta' is the same function.

### D3. The balanced |.| construction (Proposition 1(a); the registered existence test)

Given a solved signed observation-step-free model with m stepping move tokens (m = 2 ring, 4 torus; r = 4):

1. Re-solve W_in so that the move latents are c e_1, ..., c e_m (c = the source's mean move-latent norm). W_in acts only on
   the move embeddings, so this is a free gauge.
2. Per channel set omega' = max(|omega|, 1) and the steps to remainder(omega Delta, 2 pi) / omega' (angles in [0, 2 pi)
   for either sign of omega).
3. Set W_out[:, j] = step_j / c for the m move columns and 0 elsewhere; copy every other weight.

The signed control is built by the same function with the unwrapped signed angle (steps omega Delta / omega'), so the two
differ only in the wrap. On the torus m = 4 = r, exactly at Proposition 1(a)'s limit.

*Exactness.* Ring: frozen accuracy = source on 8/8; maximum angle error 1.2e-6 rad (|.|), 6.1e-7 (signed); maximum logit
difference 5.1e-4 at T = 128. Torus: frozen = source on 8/8; maximum angle error 1.4e-6 rad (|.|), 5.8e-7 (signed).
*Weight scale.* Ring: maximum |W_out| 0.31-0.38 in the signed sources, 1.07-1.70 built; |W_in| 0.15-0.21 -> 0.19-0.33.
Torus: maximum |W_out| 0.48-0.58 -> 1.23-1.55, maximum |omega| 0.68-0.96 -> 1.00. Within about 5x of the trained scale.

*Why the scale matters.* A construction that keeps the source's W_in and solves W_out at minimum norm is also exact, but in a
solved signed model the two move latents are nearly antiparallel (cosine -0.987 to -0.998), so W_out must be huge (up to
16,320 against 0.31-0.38). Adam's per-entry steps then lose that solution in the warm-up (8/8 LOST) before re-finding it
(7/8 SOLVED). An existence test is informative only near a trained scale, which the balanced construction provides.

### D4. The full |.| arms

To show that the task, not only a given signed table, is representable by a full |.| model (observations stepping), W_in
is chosen to project onto two directions orthogonal to all 17 observation embeddings and W_out is solved per channel
(`docs/audits/2026-10-12/ring_dial_audit.md` M1). In the real modules: |.| steps >= 0, observation-token |Delta| <= 1.7e-6,
cos / sin equal to the signed code's to 6.3e-4 over a 512-step walk, circular mismatch <= 4.6e-4 on 2,000 same-cell
revisit pairs.

### D5. Softplus (Proposition 1(f))

Softplus cannot reach 0, but observation steps need only be 0 mod 2 pi. With W_in annihilating every observation
embedding, each observation steps softplus(0) = ln 2 in every channel, and omega'_c = 2 pi / ln 2 makes that exactly one
turn; the move columns are set to softplus^{-1}((remainder(Theta, 2 pi) + 2 pi) / omega'). Built in the real softplus model
from a solved ring model (`docs/paper/review_draft5/theory_check_extra.py` (2), CPU): minimum step 0.693 > 0, angle error
mod 2 pi 8.4e-15 rad over all 19 tokens, logits equal to the source's to 2.8e-11 (logit scale 21.6). On the torus m = 4 =
r, so the same construction applies.

### D6. Continuations and the LOST rule

A constructed model is trained further on a fresh walk stream and compared with its signed control at the same seed.
*LOST* (registered): the run's maximum per-epoch training loss during the continuation exceeds 0.3, or exceeds 10x its
control's maximum at that seed and 0.05. *HELD* = SOLVED on at least n - 1 seeds, mean accuracy at least control - 0.01,
and LOST on at most one seed. Ring (lr 1e-3): HELD, 8/8 SOLVED, 0/8 LOST (maximum per-epoch loss 0.024-0.154 against the
control's 0.015-0.112). Torus (lr 1e-4): LOST on 6/8 and SOLVED on 8/8 at the end, read by the rule fixed before launch as
perturbed and re-found; at lr 1e-3 (descriptive) 6/8 SOLVED (0.988).

### D7. What the held model keeps (ACCOUNTS T1, registered)

Weighted by usage (mean query norm at move tokens times mean key norm at observation tokens, per channel), the held ring
|.| model keeps its cancelling code from 0.99 at construction to 0.98 after training; unweighted (every channel equal) the
share falls from 0.87 to 0.18, because channels attention does not read leave the code. Weight decay 0 vs 0.05 changes
nothing detected (-0.003).

---

## E. Theory

Labels: main-text Proposition 1 = P1(a) and P1(f) of `APPENDIX_theory_v3.md`; Lemma 1 = T1; Proposition 2 = P6;
Propositions 3 and 4 below = P3 and P4. Every check script is in `docs/paper/appendix_checks/` (CPU, seconds each).

**Notation.** A rotary channel is c = (h, b), head h and block b, acting on coordinates (2b, 2b + 1) of head h; R(phi) is
the 2D rotation. The phase is theta_{t,c} = omega_c sum_{u <= t} S_{c, x_u}, with S = W_out W_in E the C x V step table
(rank at most r) and Theta = diag(omega) S the angle table. omega is free in sign. For an inverse pair of tokens (x, y),
channel c *cancels through zero* if Theta_{c,x} + Theta_{c,y} = 0 with Theta_{c,x} != 0, *cancels by wrapping* if the sum
is a non-zero multiple of 2 pi, and is *blind* if both angles are 0 mod 2 pi.

### E1. Score identity (A0)

With alpha piecewise constant, alpha = Delta_u on (u - 1, u], the homogeneous part of Howard et al.'s integrator,
dF/dt = -s alpha(t) F, has propagator F(t) = exp(-s (Phi_t - Phi_{t'})) F(t'). With s = i omega, channel c of the rotary
score is q_c^T R(theta_s - theta_t) k_c = Re(conj(q~_c) k~_c exp(-i omega_c (Phi_t - Phi_s))), with q~ = q_{2b} + i
q_{2b+1}. *Proof.* R(theta) acts on q~ as multiplication by e^{i theta}; flows over tokens compose by adding exponents. The
identity for complex SSMs is Mamba-3's Proposition 3; ours is only its mapping onto Howard's three rate cases. Checked
(`a0_propagator.py`): 3e-14 (propagator), 4e-15 (score).

### E2. Proposition 1 (representability by wrapping; P1(a), P1(f))

*(a) Let S be a signed table with at most m <= r non-zero columns (stepping tokens x_1, ..., x_m), whose embeddings are
linearly independent modulo the span of the other embeddings (generic when V <= d_model, here 19 or 21 <= 128; automatic
in observation-step-free models). Then a |.| model with the same embeddings and content weights has an angle table equal to
Theta mod 2 pi, entrywise.*

*Proof.* Pick W_in' with W_in' e_{x_j} = u_j (the j-th unit vector of R^r) and W_in' e_y = 0 for every other token. Set
omega' = max(|omega|, 1) per channel and W_out'[:, j] = remainder(Theta_{., x_j}, 2 pi) / omega' >= 0, other columns zero.
Then W_out' W_in' E is entrywise non-negative, so |.| leaves it unchanged, and it reproduces each stepping column mod 2 pi.
[] *Full arms:* choosing W_in' to annihilate every observation embedding leaves only the moves stepping, so every |.| model
in the paper can represent any code in which only the moves step, including the map codes of the trained signed
observation-step-free models. Checked (`p1_wrap.py`): angle error <= 7e-14, logits <= 8e-14.

*(f) Softplus. Under the same assumptions, send the stepping embeddings to u_1, ..., u_m and every other embedding to 0,
set omega'_c = 2 pi / ln 2 for every c, and W_out'[c, j] = softplus^{-1}((remainder(Theta_{c,x_j}, 2 pi) + 2 pi) /
omega'_c). Every non-stepping token then steps softplus(0) = ln 2, an angle of exactly 2 pi, and every stepping token's
angle is Theta mod 2 pi.*

*Proof.* softplus is a bijection from R onto (0, inf); the targets lie in [ln 2, 2 ln 2); omega' ln 2 = 2 pi. []
Checked (`theory_check_extra.py` (2)): D5.

### E3. Lemma 1 (relative-sign gauge; T1)

*Take a MapWM model with any number of layers and a set F of channels. Form a second model by (a) negating every token's
step in each channel of F (by negating omega_c; in the signed class equivalently row c of W_out, which is not available in
the |.| class since |.| is applied after W_out) and (b) negating output coordinate h d_head + 2b + 1 of the query and key
projections (weight row and bias entry, every layer) for each c = (h, b) in F. The two models have identical attention
scores at every layer, hence identical logits.*

*Proof.* With J = diag(1, -1), J R(phi) J = R(-phi) and J is orthogonal. (a) replaces theta_{t,c} by -theta_{t,c}; (b)
replaces q_c, k_c by J q_c, J k_c, exactly including biases, since nothing between the projection and the rotation mixes
coordinates. The channel term becomes (J q_c)^T R(-(theta_s - theta_t)) (J k_c) = q_c^T R(theta_s - theta_t) k_c. All
other components are untouched, and every layer uses the same cos / sin, so the scores agree layer by layer. [] Checked
(`t1_gauge.py`): 35 channels flipped and reflected give bitwise-identical logits (control without the reflection: 6.6).

*Corollaries.* (i) The |.| class realises a signed angle table Theta exactly as real numbers, with the same latent, iff
every row of Theta is sign-coherent (all stepping tokens in a channel step the same way); omega carries each channel's
direction. So the |.| constraint removes only the relative signs of the tokens that step in a channel. (ii) It therefore
excludes no map *function* for which a code with sign-coherent rows exists, and that code needs no wrapping: one stepping
token (parity, Z_m counting), or two moves that step the same way (+1 and +2: a(+2) = 2 a(+1)). Other codes of the same
function can still be excluded as parameter points: on +1/+2 the code a(+1) = 1, a(+2) = 2 - 2 pi has an opposite-sign
row. (iii) Up to the gauge, the excluded tables are exactly those with a row containing entries of strictly opposite sign.
For an inverse pair, every through-zero cancelling channel is such a row; cancellation by wrapping, (a, 2 pi - a), is
allowed and is representable whenever Proposition 1 applies. The constraint removes the through-zero branch of the
inverse-pair problem, not cancellation.

### E4. One channel's geometry (P3, P4)

Take one channel and an inverse pair (x, y) with angles (a_x, a_y), everything else fixed; the loss is a function on the
angle torus. Write sigma = a_x + a_y.

*Proposition 3 (cut circle).* (a) The closed quadrant [0, inf)^2 maps onto the torus, so the signed and |.| classes
attain the same loss values in this channel. (b) The non-blind cancelling set in the quadrant consists of the segments
sigma = 2 pi k, k >= 1; the through-zero line meets the quadrant only at the blind origin. (c) From a start in the open
quadrant with sigma < 2 pi, the infimum distance (in angle coordinates) to a non-blind cancelling point is min(sigma,
2 pi - sigma) / sqrt 2 in the signed class and (2 pi - sigma) / sqrt 2 in the |.| class; they coincide iff sigma >= pi.
(d) Any continuous |.| path from |sigma| < pi to a non-blind cancelling point passes through |sigma| = pi.

*Proof.* (a), (b) from the quotient map. (c) The distance to sigma = 2 pi k is |sigma - 2 pi k| / sqrt 2; the |.| class
has only k >= 1 (with omega < 0 also the negative quadrant, which is farther). (d) |sigma| is continuous along a |.| path
and a non-blind cancelling point has |sigma| >= 2 pi; intermediate value theorem. [] Checked (`p3_p5_geometry.py`): a
brute-force grid matches both formulas to 2e-4.

*Proposition 4 (sticky wall).* Let s be the pre-|.| step of one token in one channel and a = omega |s|, with
L(s) = l(|s|) and l continuously differentiable near 0. (a) For s != 0, dL/ds = omega sign(s) dL/da. (b) If l'(0) > 0, s
= 0 is a strict local minimum of L along s with every other parameter fixed: L(s) - L(0) = l'(0) |s| + o(|s|). (c)
Gradient descent on s with step eta, l' in [mu, M] on [0, rho], mu > 0, eta M <= rho and |s_0| <= rho, gives
|s_{k+1}| <= max(|s_k| - eta mu, eta M): the iterate chatters within eta M of the wall.

*Proof.* (a) chain rule for |.|; (b) first-order expansion; (c) case split on |s_k| >= eta l'. [] Proposition 4 is a
statement along one coordinate; in parameter space s = w_c . z_x is shared across tokens and channels.

### E5. Proposition 2 (CARoPE cannot cancel; P6)

CARoPE (faithful, `model_mono_baselines.MapFormerWM_CARoPEF`) advances channel i of head h by f_h(x)^i per token, with
f in (0, 1), so every per-token advance lies in (0, 1]; channel 0 advances by exactly 1, an index clock.

*(a) The phase of any interval of k >= 1 tokens lies in (0, k]. The shortest return (out and back) spans 3 tokens between
the move query and the observation key, and 4 between two observation tokens, so its phase lies in (0, 4] subset (0, 2 pi):
no channel, at any parameter value, reads every return at phase 0. (b) Fix eps in (0, pi - 2). If a channel reads each
return of an excursion of J moves out and J back within eps of 0 mod 2 pi, then the move-driven phase of any interval with
at most 2J moves lies in [0, 2 eps]: every channel is either a clock (it misses some short return) or blind over that
range.*

*Proof.* (a) Sum k numbers in (0, 1]. (b) The depth-j return phase P_j satisfies P_1 in (0, 3] and P_{j+1} - P_j in
(0, 4]; the arc (eps, 2 pi - eps) is longer than 4 when eps < pi - 2, so a sequence starting in (0, eps] with increments at
most 4 cannot jump it; induction gives P_j in (0, eps], hence J (c_+ + c_-) <= eps. [] Checked (`p6_carope.py`): eight
trained ring CARoPE models; 10,331 random draws satisfy the bound.

---

## F. Exchange-rate details

**Dyck-2** (`DYCK_MDEPTH_RESULTS.md`). Task and metric as MapFormer v4 App. B.4, trained and scored at length L = 32 and
nesting depth D = 12; primary metric A2f, Hewitt's distance-averaged closing accuracy over prefixes where a closer can be
emitted (CE-optimal predictor 1.000, chance 0.500, best stack-free floor 0.594). 2 heads, d_model 128, 1-4 layers;
path-integrated arms MapWM and MapPoPE at rank 2; index arms RoPE and PoPE at base 10000 (and base 32 in the 4-layer 3x
cell); n = 8 paired seeds. Position main effect = ((MapWM - RoPE) + (MapPoPE - PoPE)) / 2.

| condition | layers | RoPE | PoPE | MapWM | MapPoPE | position main effect |
|---|---|---|---|---|---|---|
| 1x budget | 1 | 0.648 | 0.640 | 0.996 | 0.998 | +0.353 |
| 1x budget | 2 | 0.889 | 0.850 | 1.000 | 1.000 | +0.130 |
| 1x budget | 3 | 0.964 | 0.946 | 1.000 | 1.000 | +0.045 |
| 1x budget | 4 | 0.982 | 0.971 | 1.000 | 1.000 | +0.024 |
| 3x budget | 4 | 0.998 | 0.997 | 1.000 | 1.000 | +0.002 |
| mixture of depths 4-12 | 4 | 0.898 | 0.881 | 0.999 | 0.999 | +0.110 |

Every effect is positive on 8/8 seeds (sign-flip p 0.0078). Index base 32 minus base 10000 at four layers, 3x budget:
+0.001 (RoPE), -0.001 (PoPE), within the MDE. Three times the budget raises the index arms by +0.021 at four layers (8/8).

**Ring** (C3). The index-one-layer accuracies are not monotone in p_plus (0.710, 0.677, 0.824, 1.000 on fresh seeds; every
pair differs, p 0.0022-0.0108), so the exchange rate, not the one-layer accuracy, is the quantity that tracks whether steps
are undone.

**Torus** (`FACTORIAL_RESULTS.md`, Fig. F1). Strata (short gap / long gap / wrap-only; mean and run range, n = 4): path one
layer 1.000 / 1.000 / 1.000; index one layer 0.769 (0.735-0.787) / 0.480 (0.456-0.508) / 0.482 (0.470-0.495); four
layers 0.943 (0.900-0.958) / 0.458 (0.367-0.506) / 0.457 (0.381-0.494); looped x4 0.938 (0.937-0.940) / 0.507
(0.496-0.531) / 0.509 (0.496-0.521). Floors on this stream (analysis): constant 0.508 / 0.515 / 0.517, retrace 0.912 on
short gaps. Memorisation check: four index layers 0.897 on their own training map against 0.860 held out; looped 0.862
against 0.864; path 1.000 on both.

![Figure F1](figs/fig_factorial_strata.png)

*Figure F1 (`fig_factorial_strata`). Revisit accuracy by stratum on the torus at T = 1024 (trained and tested at T = 1024;
bars: means of n = 4 runs, dots: runs). Dashed: always-blank within each stratum, 0.508 / 0.515 / 0.517; dotted: retrace
within the short-gap stratum, 0.912 (elsewhere it equals the constant); floors on this batch's own evaluation walks. Path
integration with one layer is at 1.000 in every stratum. Four index layers or a loop of four lift short-gap revisits to
0.94, just above retrace; at long gaps and wrap-only revisits every index model stays at or below the constant
(0.46-0.51).*

---

## G. Rank details

**Arms** (A1). Shared r2 (`Vanilla`), per-head r2 (`Vanilla_r2ph`), block-diagonal r4 (`Vanilla_r4mibd`: the shared rank-4
latent with W_out's off-diagonal blocks held at zero), per-head r3 (`Vanilla_r3ph`), shared r4 (`Vanilla_r4mi`), per-head
r4 (`Vanilla_r4ph`), all built from the shared rank-2 model's initial weights at each seed. Initial angle-increment scale
(std): 0.208 for block-diagonal r4 against 0.33-0.34 for the other arms, which fail (rank 2) or solve (rank 4) from the
same scale. Per-seed results: C4.

**Training curves** (Fig. G1; `fig_loss_curves_rank_out.txt`). Shared r2 0/8 SOLVED (4 STALLED, 4 DESCENDING; median final
loss 0.35) against shared r4 8/8 (2.8e-3); per-head r2 2/8 (median 0.22), r3 6/8 (3.5e-3), r4 8/8 (3.1e-3). The rank-3 and
rank-4 medians cross the threshold at epochs 478-532; the rank-2 medians never do.

![Figure G1](figs/fig_loss_curves_rank.png)

*Figure G1 (`fig_loss_curves_rank`). Training loss per epoch on the torus (T = 1024) by the per-head rank of the
content-to-angle map; thin lines are runs, thick lines medians of 8; dashed: the SOLVED threshold. (a) One latent shared by
the heads: rank 2 solves 0/8, rank 4 8/8. (b) Per-head latents: rank 2 2/8, rank 3 6/8, rank 4 8/8.*

**Restart curves** (`RANK_PROJ_ANALYSIS.txt`). Projected rank 2: peak training loss 0.37-0.46 after the restart, back
below 0.05 at epochs 553-654 on 7/8 seeds. Rank-4 control: peak 0.26-0.30, back at 497-669 on 7/8. Rank 2 continued: peak
0.55-0.82, back on 1/8.

**Rank against dimension** (`RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md`, `RANK_TESTS_RESULTS.md`). Per-head rank, every arm
from the same base, trained and tested at T = 1024, 900 epochs:

| torus | D | per-head rank | wrap-only share of revisits | SOLVED | held-out accuracy |
|---|---|---|---|---|---|
| 32 per side | 2 | 2 | 0.37 | 1/8 | 0.769 |
| 32 per side | 2 | 3 | 0.37 | 8/8 | 0.999 |
| 10 per side | 3 | 3 | 0.71 | 1/8 | 0.800 |
| 10 per side | 3 | 4 | 0.71 | 4/8 | 0.847 |
| 18 per side | 3 | 4 | 0.35 | 8/8 | 1.000 |

2D rank 3 vs rank 2: 8/8 vs 1/8 (Fisher p 0.0014), +0.230 (permutation p 0.0014). 3D, 18 vs 10 per side at rank 4: +0.153
(p 0.0085). On a 100-cell 2D torus (10 per side) rank 3 memorises its training map (0.986 on it, 0.273 on an unseen map),
so that cell does not measure map learning. 1D ring (32 cells, T = 128, 300 epochs): rank 1 7/8 (0.993), rank 2 8/8
(1.000); with the component common to the move steps removed by construction, rank 1 solves 8/8 in a median of 24 epochs
(58 without).

**Escape test** (`RANK_TESTS_RESULTS.md`). Lifting the eight stalled shared-rank-2 runs to rank 4 (function-preserving)
and training 600 epochs solves 3/8, against 1/8 for rank 2 continued and 2/8 for a fresh rank-4 step map on the same
networks; at this budget the contrasts are not resolved.

---

## H. Sign details

### H1. Learning rates (`LRSWEEP_RESULTS.md`; ring, observation-step-free, seeds 1500-1507)

| learning rate | signed SOLVED, accuracy | \|.\| SOLVED, accuracy |
|---|---|---|
| 1e-3 | 8/8, 1.000 | 1/8, 0.917 |
| 3e-3 | 8/8, 1.000 | 2/8, 0.963 |
| 1e-2 | 4/8, 0.972 | 1/8, 0.872 |
| 3e-2 | 1/8, 0.559 | 0/8, 0.584 |

Registered B1 (each arm at its best rate, selection redone in the null over all 12,870 relabellings): |.| 3e-3 2/8 vs
signed 1e-3 8/8, -0.037, decision p 0.0070 (SOLVED) / 0.0057 (accuracy). The |.| optimum is interior to the grid. Three
|.| runs at 3e-3 were still descending at 300 epochs. At 3e-2 both arms sit near the 0.50 blank floor. Per seed:

```
seed | 1e-3:SO | 1e-3:AO | 3e-3:SO | 3e-3:AO | 1e-2:SO | 1e-2:AO | 3e-2:SO | 3e-2:AO
1500 | 1.0000 SOLV | 0.9362 STAL | 1.0000 SOLV | 0.9111 STAL | 0.9460 DESC | 0.9936 DESC | 0.4960 STAL | 0.4958 STAL
1501 | 1.0000 SOLV | 0.8759 STAL | 1.0000 SOLV | 0.9662 DESC | 0.9460 STAL | 0.4960 STAL | 0.4960 STAL | 0.4960 STAL
1502 | 1.0000 SOLV | 0.8995 STAL | 1.0000 SOLV | 0.9428 STAL | 0.9407 STAL | 0.9703 DESC | 0.4960 STAL | 0.4960 STAL
1503 | 1.0000 SOLV | 0.9116 STAL | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.6000 STAL | 0.4960 STAL | 0.4960 STAL
1504 | 1.0000 SOLV | 0.9415 STAL | 1.0000 SOLV | 0.9686 STAL | 0.9460 STAL | 1.0000 DESC | 0.4960 STAL | 0.8159 DESC
1505 | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.9993 SOLV | 1.0000 SOLV | 0.9950 DESC | 0.9998 SOLV | 0.4961 STAL
1506 | 1.0000 SOLV | 0.8736 STAL | 1.0000 SOLV | 0.9595 DESC | 1.0000 SOLV | 0.9996 SOLV | 0.4960 STAL | 0.4960 STAL
1507 | 1.0000 SOLV | 0.8971 STAL | 1.0000 SOLV | 0.9587 DESC | 1.0000 SOLV | 0.9184 DESC | 0.4959 DESC | 0.8799 DESC
```

### H2. Recipe cells (`ROBUST_RESULTS.md`, seeds 1400-1407)

| cell | recipe | \|.\| accuracy, SOLVED | signed accuracy, SOLVED | d [95% CI] | perm p | Fisher p |
|---|---|---|---|---|---|---|
| ring | lr 1e-3, weight decay 0.05 | 0.911, 1/8 | 1.000, 8/8 | -0.089 [-0.125, -0.055] | 0.0002 | 0.0014 |
| ring | lr 1e-3, weight decay 0 | 0.873, 0/8 | 1.000, 8/8 | -0.127 [-0.200, -0.075] | 0.0002 | 0.0002 |
| ring | lr 3e-3, weight decay 0.05 | 0.987, 4/8 | 1.000, 8/8 | -0.013 [-0.025, -0.005] | 0.026 | 0.077 |
| ring | lr 3.3e-4, weight decay 0.05 | 0.841, 0/8 | 0.950, 6/8 | -0.109 [-0.215, +0.010] | 0.089 | 0.0070 |
| torus | 2 layers, d_model 256 | 0.967, 2/8 | 0.999, 8/8 | -0.032 [-0.055, -0.015] | 0.0005 | 0.0070 |

At lr 3e-3 the registered contrast was below its decision threshold in this batch (4/8 vs 8/8); on fresh seeds in the
sweep the same rate gives 2/8 vs 8/8 (H1). Declared secondaries: cancelling share (unweighted) |.| 0.04-0.13 in every
cell, signed 0.43-1.00; torus strata (short / long / wrap-only) signed 1.000 / 0.999 / 0.992, |.| 0.989 / 0.911 / 0.799;
on the torus 6/8 |.| runs were still descending at 900 epochs (final losses 0.018-0.144).

### H3. Starts (`MECHANISM_RESULTS.md`, ring p_plus 0.5, seeds 800-807; `TORUS_START_RESULTS.md`, torus, seeds 1200-1207)

Observation-step-free models differing only in the initial move angles (every other weight identical at a seed):

| initial move angles | signed (ring) | \|.\| (ring) | signed (torus) | \|.\| (torus) |
|---|---|---|---|---|
| default (small, signs independent) | 8/8 | 0/8 | 8/8 | 1/8 |
| the \|.\| start: both moves rotating the same way | 7/8 | 0/8 | -- | -- |
| large, opposite-move sums below pi, U(0.9, 1.5) | -- | 2/8 | -- | 0/8 |
| large, sums above pi, U(1.7, 4.5) | 8/8 | 8/8 | 2/8 | 0/8 |

Ring, registered: same start, sign matters (0/8 vs 7/8, Fisher p 0.0014; -0.092 [-0.135, -0.045]); a start past pi rescues
|.| (8/8 vs 0/8, +0.109 [+0.085, +0.135], p 0.0002); a large start with sums below pi does not (2/8; vs past pi Fisher p
0.0070); from past pi the sign makes no difference (8/8 vs 8/8). The large start also moves the angles 13-22x faster per
Adam step (the sub-pi control about 5x), so sum and size are only partly separated. Analysis: signed models from the small
starts end with 98-99% of channels opposite-signed and cancel through zero; from past pi both signs cancel by wrapping
(signed 88%, |.| 100% of cancelling channels); the rescued |.| models cancel in 23% of channels. Torus, registered: the past-pi
start does not rescue |.| (0/8 vs 1/8, +0.005 [-0.060, +0.070]), and it slows the signed model (2/8; 5/8 runs per large-start
arm still descending at 900 epochs); the default-start sign cost replicates (1/8 vs 8/8, Fisher p 0.0014). The start lever
is a ring result.

### H4. Moves +1/-1 vs +1/+2 (`GAUGE_RESULTS.md`, ring, seeds 1300-1307)

By Lemma 1 (ii) a |.| step represents, without wrapping or crossing zero, a map whose two moves step the same way. With
moves +1/-1 (an inverse pair) the cost replicates: |.| 0.931, 1/8 vs signed 1.000, 8/8 (-0.069 [-0.095, -0.045], Fisher p
0.0014). With moves +1/+2, which needs integration (index RoPE 0.649, 0/8), |.| scores 0.883, 2/8 against signed 0.947,
5/8 (-0.064 [-0.200, +0.075], Fisher p 0.31): the registered comparison is not resolved, because +1/+2 (every revisit a
lap of at least 16 moves) is harder for the signed step as well. Signed models on +1/+2 end with 0.92 of channels having
both move angles of one sign (0.01 on +1/-1), the sign-coherent code Lemma 1 allows. The |.| step does not learn +1/+2
more reliably than +1/-1 (2/8 vs 1/8).

### H5. Three times the ring budget (`RING_BUDGET_RESULTS.md`, seeds 40-47)

The |.| step at 900 epochs solves 3/8, as at 300 epochs on the same seeds (3/8); the solved sets differ.

### H6. Cancelling share and ring strata

![Figure H1](figs/fig_wrap_share.png)

*Figure H1 (`fig_wrap_share`; analysis). Share of rotary channels in which opposite moves cancel mod 2 pi, per run,
weighted by how much attention uses each channel (dots; filled = SOLVED) and unweighted (faint dashes, joined to the same
run). A channel cancels when |wrap(a(move) + a(opposite move))| < 0.1 rad; channels whose move angle has collapsed to
within 0.05 rad of 0 mod 2 pi are excluded from the denominator. Usage weight: mean query norm of the channel at move
tokens times mean key norm at observation tokens. Ring of 32 cells at three move biases (n = 8 each) and the torus at
T = 1024 (n = 8). Bars: arm means of the weighted share. Signed steps 0.70-0.93; |.| steps 0.02-0.10; softplus 0.01. Seven
of the eight non-negative runs that solve stay at or below 0.07.*

| cell | arm | SOLVED | weighted mean (range) | unweighted mean (range) |
|---|---|---|---|---|
| ring p_plus 0.5 | signed | 8/8 | 0.79 (0.41-0.99) | 0.79 (0.59-0.92) |
| ring p_plus 0.5 | \|.\| | 2/8 | 0.06 (0.00-0.29) | 0.06 (0.02-0.16) |
| ring p_plus 0.75 | signed | 8/8 | 0.87 (0.55-0.99) | 0.78 (0.64-0.86) |
| ring p_plus 0.75 | \|.\| | 1/8 | 0.02 (0.00-0.07) | 0.03 (0.02-0.04) |
| ring p_plus 0.9 | signed | 8/8 | 0.70 (0.39-0.99) | 0.64 (0.44-0.91) |
| ring p_plus 0.9 | \|.\| | 4/8 | 0.10 (0.00-0.34) | 0.06 (0.02-0.10) |
| torus T 1024 | signed | 8/8 | 0.93 (0.64-1.00) | 0.98 (0.90-1.00) |
| torus T 1024 | \|.\| | 0/8 | 0.09 (0.00-0.29) | 0.11 (0.00-0.27) |
| torus T 1024 | softplus | 1/8 | 0.01 (0.00-0.09) | 0.03 (0.00-0.12) |

![Figure H2](figs/fig_ring_strata.png)

*Figure H2 (`fig_ring_strata`; analysis). Revisit accuracy on the 1D ring by stratum (eval mode; 400 trajectories,
T = 128; maps redrawn). "Plain" revisits have an earlier visit at the same unwrapped position; "wrap-only" revisits were
seen only a lap away. Lines: arm means; dots: runs; hollow markers: plain revisits re-weighted to the wrap-only gap
distribution. Signed steps (24 runs over three batches) and |.| steps started past pi are at 1.000 everywhere. |.| steps
from the default start lose accuracy with the gap (0.96-0.99 at gaps 2-7, 0.87-0.96 at gaps 32-127). Dotted: retrace
within each stratum, 1.000 at gaps 2-7 and 0.943 at gaps 8-31 (0.57 and 0.49 in the long-gap and wrap-only strata, off
axis); constant floor 0.50.*

---

## I. CARoPE and CoPE

**CARoPE** (`model_mono_baselines.py`, `CARoPEF`). Per head h and token t a scalar rate f_h(x_t) = 1 / (softplus(x_t W)_h
+ 1) in (0, 1) replaces RoPE's base frequency, so channel i advances by f_h(x_t)^i per token; initialised as index RoPE
(W = 0, a per-head bias with 1 / (softplus(b) + 1) = 10000^(-2 / d_head)). Every token, observations included, advances
the phase; the layer is MapWM's, so only the phase differs. Proposition 2 (E5): no channel can read every return.
Registered (GENERALITY): torus 0.667 (0/8), below index RoPE (0.722, -0.055, p 0.040); ring 0.741 against index 0.730
(+0.011, not resolved).

**CoPE** (`CoPEF`). Golovneva et al.'s Listing 1 in torch: per layer one module shared by the heads; gates g_ij =
sigmoid(q_i . k_j / sqrt(d_h)); count p_ij = sum_{k=j}^{i} g_ik, clamped to 2047; position logit by linear interpolation
of q_i . e[floor p] and q_i . e[ceil p], table e at zeros initially; added to the attention logits; no rotary position.
*Representability (constructed):* gates 17/16 and 15/16 per (move, observation) pair make p_ij mod 2 a function of net
displacement, read by a period-2 table (1.000 revisit accuracy). The exact solutions form a lattice (P7 of v3: period 2 or
3).

| batch | task | CoPE | SOLVED | comparison |
|---|---|---|---|---|
| GENERALITY (700-707) | ring | 0.887 | 0/8 | signed 1.000, 8/8 (-0.113, p 0.0002); index 0.730 (+0.157, p 0.0002) |
| WRAP_COPE (900-907) | ring, table at zeros / sinusoid past pi / below pi / slope-matched | 0.840 / 0.913 / 0.867 / 0.943 | 1/8, 2/8, 1/8, 0/8 | contrasts not resolved (bimodal seeds) |
| ACCOUNTS T4 (1100-1115) | 256-cell line (never wraps) / ring | 0.914 / 0.873 | 1/16 / 0/16 | line: signed 1.000, 8/8 (Fisher p < 1e-4); index 0.772, 0/8 |

On both line and ring several CoPE seeds reach 0.99-1.00 held-out accuracy without meeting the loss criterion (at a held-out
accuracy cut of 0.999 the line reads 5/16 and the ring 2/16; analysis). MapFormer v4 Table 2 reports CoPE failing 1D and 2D
navigation (one layer 0.63 +/- 0.21 in 1D and 0.65 in 2D; four layers 0.88 and 0.76).

---

## J. Text world

**Task** (`TEXTWORLD_RESULTS.md`, `train_textworld.py`). The 64 x 64 torus walk rendered as English: 58 words, three
synonyms per direction, movement-free verbs, fillers and asides; T = 1024 words (about 132 steps); held-out map; predict
the observation word at revisits. A direction word appears only inside a movement clause (context-free steps). Floors:
best constant 0.505, reversal-copy 0.597. Recipe: the torus recipe (A2), path MapWM r4 at one layer against index RoPE at
one and two layers; 200 evaluation walks.

**Step tables per seed** (declared secondaries, `fig_textworld_out.txt`; "minus common" = opposition after removing the
component shared by the four direction steps; ablation columns: held-out accuracy, eval mode without the dropout scale):

| seed | type | raw opposition | minus common | common / north | accuracy | non-direction steps zeroed | direction steps zeroed |
|---|---|---|---|---|---|---|---|
| 0 | clock | 1.730 | 0.0071 | 0.711 | 0.881 | 0.191 | 0.388 |
| 1 | map | 0.083 | 0.0029 | 0.041 | 0.998 | 0.893 | 0.505 |
| 2 | clock | 1.434 | 0.0064 | 1.144 | 0.999 | 0.665 | 0.482 |
| 3 | clock | 1.707 | 0.0022 | 0.480 | 0.995 | 0.673 | 0.463 |
| 4 | clock | 1.768 | 0.0052 | 0.501 | 0.999 | 0.539 | 0.445 |
| 5 | map | 0.120 | 0.0050 | 0.060 | 1.000 | 0.898 | 0.505 |
| 6 | map | 0.080 | 0.0117 | 0.040 | 1.000 | 0.950 | 0.505 |
| 7 | map | 0.119 | 0.0057 | 0.059 | 0.882 | 0.586 | 0.498 |

On the clock seeds the common component is parallel to the verbs' step (cosine +0.999 to +1.000), and 31-38 of 64 phase
channels drift between visits to the same cell (2-4 on the map seeds). Synonym cosine 0.9996-1.0000 on every seed.

---

## K. What and where (analysis)

**Do trained path models need the content x position interaction?** (`docs/WHAT_WHERE_CHECKS.md`; eval-only on the 48
short-walk torus checkpoints of PAPER2X2.) Held-out accuracy when the score is rebuilt (floors: n-gram 0.598, blank 0.507):

| score used | RoPE | PoPE | MapWM r2 | MapWM r4 | MapPoPE r2 | MapPoPE r4 |
|---|---|---|---|---|---|---|
| intact | 0.805 | 0.679 | 0.971 | 1.000 | 0.999 | 1.000 |
| position only (object identity removed, token type kept) | 0.463 | 0.476 | 0.837 | 0.989 | 0.974 | 0.986 |
| shared position kernel x content gain | 0.430 | 0.435 | 0.791 | 1.000 | 0.988 | 1.000 |

Converged path models keep 0.974-0.989 with object identity removed and 0.988-1.000 with a shared kernel times a content
gain; index models fall below the floor. Separation as such is MapFormer's (their Fig. 9).

**Converging on new objects** (`LEAK_RESULTS.md`, `GAIN_PHASE_RESULTS.md`, registered). On a task with a fresh map and 16
fresh objects per sequence (32 x 32 torus, T = 1024, rank 4), object tokens leak a small step into the phase. Removing the
step from object tokens (ActOnly) or normalising the step input (NormStep, the step reads LayerNorm(embedding)) lets the
model converge: 16/16 SOLVED (unseen-object accuracy 0.9997 / 0.9998) against 0/8 for MapWM (0.9890, still descending).
On fresh seeds NormStep again converges 8/8 against MapWM's 0/8 (unseen-object accuracy 0.9940 against 0.9895).

---

## L. PoPE's score

On the short-walk torus (T = 128, rank 2, n = 16, fresh seeds), MapPoPE's gain over MapWM comes with PoPE's score rule
(+0.0243, permutation p 0.012; SOLVED 16/16 vs 10/16), and doubling the angle count adds +0.0002 at a ceiling where both
arms sit at 0.9995-0.9997 (`MAPPOPE_PAIR_RESULTS.md`).
