# The clock side of the clock / map dissociation -- pre-registration (2026-10-12, before any run)

## Question
The paper's frame is Howard et al. 2014's integrator (`docs/lit/LIT_CLOCK_MAP.md`, `docs/lit/LIT_HOWARD_TRANSFORMER.md`):
one accumulator driven by a constant rate (Case I: time), a non-negative content-dependent rate (Case II: a content clock,
path length) or a signed rate (Case III: position, a map). The three families of rotary position are these cases:
index RoPE (I), monotone content-dependent phase (II: CARoPE, GRAPE-AP / softplus, |.|), signed content-dependent phase
(III: MapFormer). On the MAP task the prediction is established at matched length: forcing Case II on the torus fails
(`SIGN_MATCHED_RESULTS.md`, Abs - Signed -0.177, 0/8 vs 8/8). On the CLOCK side it is not: on k-back recency (a
content-counting task, where index RoPE fails at 0.234) every phase arm was at or near ceiling at the training length
(`RECENCY_RESULTS.md`: Signed 1.000, CARoPE 1.000, Pos 0.979, Abs 0.961), and the registered -0.004 was read out of
distribution (T = 2048 after training at 1024). **On a content-clock task made hard enough to stay below ceiling, at the
training length, does a monotone (Case II) phase beat the signed (Case III) one -- completing a double dissociation --
or does the signed phase match it (it can emulate a clock)?**

## Task and recipe (`train_recency.py`, unchanged; = `run_recency.sh` except k_max / T, set by the calibration below)
k-back recency: retrieve the k-th most recent symbol among fillers, 16 symbols, p_query 0.25, min_gap = k_max (the
default, which keeps the answer-stream shortcut closed), 300 epochs x 48 batches, batch 16, cosine, lr 1e-3, 1 layer,
d 128, 2 heads, `--fast-attn` (as RECENCY; every arm shares it). Evaluation at the training length T and at 2T
(secondary, out of distribution).

## Calibration (pilot, seed 100, outside the batch; rule fixed now)
Settings tried in order: (k_max 128, T 1024), (k_max 128, T 2048), (k_max 256, T 2048); arms Signed_r4 and CARoPE_r4 at each.
The batch uses the FIRST setting at which the pilot's Signed_r4 accuracy at the training length is in [0.30, 0.95]
(below ceiling, well above the floor). If none qualifies: no batch; report. The pilot's CARoPE result is disclosed but
does not enter the choice. The gate (`validate_recency.py`: chance, most-recent shortcut, n-gram floors, scored targets
per sequence) is run for the chosen setting before launch.

## Arms (`train_variant.VARIANT_MAP`; n = 8, seeds 200-207, fresh)
| arm | Howard case | step |
|---|---|---|
| Signed_r4 | III (signed) | `W_out W_in x` (rank 4) |
| CARoPE_r4 | II (monotone) | `1 / (softplus(.) + 1)` (CARoPE) |
| Pos_r4 | II (monotone) | `softplus(.)` (GRAPE-AP) |
| RoPE | I (constant) | token index |

## Registered decisions (`analyze_clock_side.py`, to be written with the amendment; two-sided permutation tests, n = 8 v 8)
Primary readout: held-out accuracy at the training length (eval mode). Monotone = mean of CARoPE_r4 and Pos_r4 per seed index
(RECENCY's registered composite).
- **D1 CLOCK SIDE.** contrast mean(monotone) - Signed. **CASE II ADVANTAGE (double dissociation)**: d > 0, perm p < .05
  and d >= 0.02. **SIGNED BETTER**: d < 0, p < .05, |d| >= 0.02. **NO DIFFERENCE (signed matches the clock)**: neither
  fires and the 95% CI lies within [-0.03, +0.03]. **UNMEASURED**: otherwise.
- **D2 INDEX CHECK.** RoPE - Signed: **INDEX FAILS** if d <= -0.10 with p < .05 (a token clock cannot count content);
  otherwise reported as it falls.
Secondary: SOLVED counts (final-5% loss < 0.05) with Fisher tests; per-arm accuracy at 2T (OOD, rule 10); dropout-scale
re-score of the primary (one layer); r(final loss, acc); the growth exponent alpha per arm.

## Process
Pilot (calibration) -> Amendment 1 (setting chosen, gate) -> independent code audit (rule 29) -> Amendment 2 -> batch on
GPU 0 only while another user holds GPU 1 (user's permission 2026-10-12). Cost estimate: 32 runs x 300 epochs at
2-4 s/epoch, 4 concurrent on one GPU: ~1.5-2.5 h.

## Amendment 1 (2026-10-12, calibration pilot and gate; before the audit and the batch)
Calibration (`docs/audits/2026-10-12/clock_side_calib.sh`, d0ef56c clean, seed 100, GPU 0), accuracy at the training length:
| setting | Signed_r4 | CARoPE_r4 (disclosed, not used for the choice) |
|---|---|---|
| k_max 128, T 1024 | **0.493** (final loss 1.77) | 0.771 (0.88) |
| k_max 128, T 2048 | 0.924 (0.24) | 0.950 (0.41) |
| k_max 256, T 2048 | 0.061 (2.77, nothing learned) | 0.061 (2.77) |
By the registered rule the batch uses **k_max 128, T 1024** (the first setting with Signed in [0.30, 0.95]). Eval targets per
run at T = 1024: ~205 (per-seed accuracy SE ~0.03). Gate (`CLOCK_SIDE_GATES.md`, `validate_recency.py`, 800 episodes,
min_gap 128): chance 0.0625, marginal 0.0716, most-recent 0.0709 (floor 0.0698), word n-gram orders 1 / 2 / 3 / 5: 0.065 /
0.065 / 0.079 / 0.082, oracle 1.000. The strict gate rule (every order within 0.01 of chance) flags orders 3 and 5 (n = 2638
scored targets; likely n-gram overfitting); the **best trivial predictor is 0.082**, reported as the floor. Constants set:
`analyze_clock_side.py` T = 1024, K_MAX = 128; `run_clock_side.sh` K_MAX=128, T=1024 (GPU 0 only, 4 per GPU).
Note: one pilot seed shows the monotone arm ahead (0.77 vs 0.49); that is the hypothesis's direction, n = 1, no verdict.

## Amendment 2 (2026-10-12, after the independent code audit `docs/audits/2026-10-12/clock_side_audit.md`; before the batch)
The audit (blind to the calibration results) verified the arms, matched initialisation (Signed / Abs / CARoPE / Pos share
initial weights at a seed; RoPE differs), identical training and eval streams across arms at a seed, min_gap 128 at T and
2T, --fast-attn validity for every arm (logits <= 9.5e-7, gradient cosine 1.000), the driver (dry run: 32 unique launches on
cuda:0) and the test directions. Findings -> changes; this amendment governs where it conflicts with the text above.
- **M1 (power).** ~213 eval targets per run made NO DIFFERENCE practically unreachable (a true null read UNMEASURED 85-93% in
  the audit's simulation). -> **Primary readout = a re-evaluation of every saved checkpoint on 512 sequences (~1,690 targets
  per run)** at T = 1024 from a fresh stream (`eval_clock_side.py`: `train_recency.evaluate` unchanged, np RandomState(7000 +
  seed), the same items for every arm at a seed; tested on the two calibration checkpoints: n 1691). The in-training eval
  (~213 targets) is printed beside it.
- **M2 (arms).** The composite of CARoPE and Pos changes more than the sign (CARoPE's step is bounded in (0, 1) and starts
  as a near-uniform token clock; both move slower per update), and RECENCY's composite also included Abs. -> **Abs_r4 is
  added** (`|W_out W_in x|`, the same parameterisation as Signed with the sign removed; 8 runs) and is the **primary
  contrast D1 = Abs_r4 - Signed_r4**. CARoPE_r4 - Signed_r4 (D1c) and Pos_r4 - Signed_r4 (D1p) are each registered
  separately with the same branches; no composite. 5 arms x 8 seeds = 40 runs.
- **M3 (what a Case II advantage can mean).** A signed step can represent every monotone solution (RECENCY's amendment), so
  CASE II ADVANTAGE can only reflect search or training speed. -> The label always carries "[search / training speed: the
  signed arm can represent every monotone solution]" and the count of monotone runs still DESCENDING at 300 epochs. A
  2x-budget rerun is a candidate follow-up, not part of this batch.
- **M4 (dropout scale).** One layer below ceiling is where eval mode under-reports. -> The re-evaluation also scores every
  run under the 1/(1-p) re-score (`rescore_hook`, same process); if a D1 label differs between eval mode and the re-score,
  the line carries a FLAG with the re-scored label (calibration checkpoints: 0.4926 / 0.4950, 0.8114 / 0.8072).
- **B1 / B2.** Secondaries added to `analyze_clock_side.py`: SOLVED / STALLED / DESCENDING counts, eval n per run, the floor,
  paired-by-seed differences with the exact sign-flip test (the audit's pairing point), Fisher on SOLVED, r(final loss,
  acc) overall and per arm. Provenance: T, k_max and min_gap are asserted from each run's log header and checkpoint,
  epochs from the loss curve. The growth-exponent secondary is dropped (not needed for any decision).
- **Gate.** At 8,000 episodes (`CLOCK_SIDE_GATES_8000.md`) the gate PASSES: n-gram orders 0.058-0.069, marginal 0.0666,
  most-recent 0.0724, oracle 1.000; the 800-episode flags were noise. Floor (best trivial predictor) **0.072**.
- **Minor.** "RECENCY's registered composite" in the text above was inexact (it averaged three arms including Abs); TF32 is
  on under --fast-attn in every arm (as RECENCY) and runs are not bitwise deterministic; per-k readouts (~13 targets per
  offset on the re-evaluation) are descriptive only.
Cost: 40 runs x 300 epochs at ~3-4 s/epoch, 4 at a time on GPU 0: ~2.5 h, plus the re-evaluation (~5 min).

## Amendment 3 (2026-10-12, interpretation only; written while the batch runs, before any batch result was read; no code change)
The independent RING_DIAL audit (`docs/audits/2026-10-12/ring_dial_audit.md` M1) showed that a non-negative ROTARY step is not
representationally monotone: angles wrap mod 2 pi, so setting the second token's step to 2 pi / omega minus the first's
reproduces any signed code exactly (constructed in the real modules). Abs_r4, Pos_r4 and CARoPE_r4 can therefore represent
every signed solution, and Signed_r4 every monotone one. Every D1 / D1c / D1p label in this batch is a statement about
LEARNABILITY under each parameterisation within 300 epochs, not about what the arms can represent; "Case II vs Case III"
in Howard's sense (real, non-wrapping rates) is not separated by rotary arms. The decision rules are unchanged.
