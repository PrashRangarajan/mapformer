# Fresh-seed replication of CLOCK_SPEED's primary contrast -- pre-registration (2026-10-12, before any run)

## Question
`CLOCK_SPEED_RESULTS.md` (registered): on k-back recency the non-negative rotary step (Abs_r4) learned faster than the signed
one (Signed_r4) on 12/16 paired seeds, mean paired AUC difference -0.197, exact sign-flip p 0.053 -> UNMEASURED. **On 16 fresh
seeds, with the same task, recipe, readout and rule, does S1 fire?** This is a new batch with its own registered test, not an
extension of the first one; the result is reported as it falls. A pooled reading of both batches (32 seeds) is printed as a
labelled post hoc secondary only.

## Design (= CLOCK_SPEED's, `train_recency.py` unchanged)
k-back recency, 16 symbols, k_max 128, min_gap 128, p_query 0.25, T = 1024, 300 epochs x 48 batches, batch 16, cosine, lr
1e-3, 1 layer, d 128, 2 heads, --fast-attn. Arms **Signed_r4, Abs_r4** (matched initial weights at a seed), seeds **230-245**
(fresh), n = 16 paired. 32 runs on **GPU 1** (free; GPU 0 runs FACTORIAL; user's permission 2026-10-12).

## Registered decision (`analyze_clock_speed_rep.py`, reusing `analyze_clock_speed.branch` unchanged)
S1 = Abs - Signed on AUC (mean training loss over the 300 epochs), paired by seed, exact sign-flip test:
NON-NEGATIVE LEARNS THE CLOCK FASTER (mean d < 0, p < .05, |mean d| >= 0.10) / SIGNED LEARNS THE CLOCK FASTER (mirror) /
NO DIFFERENCE (95% sign-flip CI within [-0.10, +0.10]) / UNMEASURED.
Secondary: per-arm curves (epoch to running loss < 2.0 / < 1.0, loss at 100 / 200 / 300), accuracy at T and 2T, the pooled
32-seed reading (post hoc, labelled).

## Process and cost
Independent code check (rule 29) before launch. ~9,600 run-epochs on one GPU at 4 concurrent: ~2.5-3 h.
