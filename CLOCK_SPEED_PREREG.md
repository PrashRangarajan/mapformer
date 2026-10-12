# Does a non-negative rotary step learn a content clock faster? -- pre-registration (2026-10-12, before any run)

## Question
In the rotary slot a signed and a non-negative step represent the same codes (angles wrap; `RING_DIAL_PREREG.md` Amendment
1). What separates them is learnability. On the ring map task the signed step learns the map and the non-negative step
does not (`RING_DIAL_RESULTS.md`, `RING_BUDGET_RESULTS.md`: 8/8 vs ~3/8, unchanged by 3x budget). On the content-clock task
(k-back recency) CLOCK_SIDE could not decide (nothing converged), but its training curves (post hoc,
`docs/audits/2026-10-12/clock_side_curves_out.txt`) show the non-negative arms leaving the chance plateau earlier (median
epoch 176-180 vs 230) and lower at epochs 200 / 300. **Is the non-negative step's faster learning of the content clock
real (a learnability double dissociation with the ring), measured as learning speed on fresh seeds?**

## Design (= CLOCK_SIDE's task and recipe; `train_recency.py` unchanged)
k-back recency, 16 symbols, k_max 128, min_gap 128, p_query 0.25, T = 1024, 300 epochs x 48 batches, batch 16, cosine, lr
1e-3, 1 layer, d 128, 2 heads, --fast-attn (TF32). Arms: **Signed_r4, Abs_r4 (the sign alone), CARoPE_r4** (Pos and RoPE
dropped: Pos tracked Abs in CLOCK_SIDE, RoPE fails the task). Seeds **210-217** (fresh), n = 8; matched initial weights
across the three arms at a seed (audited for CLOCK_SIDE). 24 runs, GPU 0 only (another user holds GPU 1).

## Registered decisions (`analyze_clock_speed.py`)
Primary readout per run: **AUC = mean training loss over the 300 epochs** (lower = learned faster; chance loss 2.77).
- **PAIRED BY SEED (primary test).** The three arms share initial weights, training data and eval items at a seed, so the
  contrast is paired: d_i = AUC(arm, seed i) - AUC(Signed_r4, seed i), exact sign-flip test on mean(d) (n = 8, minimum two-sided
  p 0.0078). This choice was made after running this analysis on CLOCK_SIDE's existing curves (seeds 200-207: paired CARoPE
  8/8 faster, p 0.0078; Abs 6/8, p 0.086; unpaired both p > .09); the batch uses fresh seeds 210-217.
- **S1 (Abs - Signed, the sign alone)** and **S2 (CARoPE - Signed)**, each: **NON-NEGATIVE LEARNS THE CLOCK FASTER**: mean d < 0,
  sign-flip p < .05 and |mean d| >= 0.10. **SIGNED LEARNS THE CLOCK FASTER**: the mirror. **NO DIFFERENCE**: |mean d| < 0.05 and
  p >= .05. **UNMEASURED**: otherwise.
Secondary: the unpaired permutation test and CI; first epoch the 10-epoch running loss falls below 2.0 and 1.0 (censored at 301);
loss at epochs 100 / 200 / 300; in-training eval accuracy at T = 1024 (~213 targets) and 2T; SOLVED counts; r(AUC, accuracy).
Materiality 0.10 in loss units: CLOCK_SIDE's arm means differed by 0.4-0.8 at epoch 200 and the per-seed sd of the epoch-300
loss there was ~0.5; a 0.10 AUC difference is about a tenth of the distance from chance to the best arm.

## Process
Independent code check (rule 29) before launch; no pilot (the task, arms and recipe ran in CLOCK_SIDE; the readout is
computed from saved loss curves, smoke-tested on CLOCK_SIDE's runs as fabricated input). Cost ~2 h on GPU 0.
