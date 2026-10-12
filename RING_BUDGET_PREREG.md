# Is the non-negative rotary step slower, or stuck? Ring budget extension -- pre-registration (2026-10-12, before any run)

## Question
RING_DIAL (`RING_DIAL_RESULTS.md`, registered) found that a non-negative rotary step (Abs_r4), which can represent the map by
wrapping, converges on 2/8 seeds at p_plus 0.5 within 300 epochs against 8/8 for the signed step, and never finds the
wrap solution (4% of channels cancel mod 2 pi vs 82%). Rule 4: before reading a budget-limited miss as a handicap, extend
the budget. **With three times the training (900 epochs, the same cosine schedule stretched), does Abs_r4 converge on the
ring at p_plus 0.5?**

## Design
Task, model and recipe = RING_DIAL's (`train_ring_dial.py`, unchanged: 1D ring, 32 cells, T = 128, batch 128, 98 batches,
lr 1e-3, cosine, 1 layer, 2 heads, d 128; fresh-stream evaluation at T = 128 and 512) except **--epochs 900**. One arm,
Abs_r4, p_plus 0.5, **seeds 40-47** (fresh; RING_DIAL used 30-37), n = 8. No signed arm: Signed_r4 solved 8/8 at 300 epochs
(the comparison is descriptive, across batches; no contrast is tested between them).

## Registered decision (`analyze_ring_budget.py`)
SOLVED = final-5% training loss < 0.05 (`stats_core.classify_run`). k = SOLVED count of 8.
- **SLOWER, NOT STUCK**: k >= 6 (the handicap is training speed at this budget).
- **PERSISTENT HANDICAP**: k <= 2 AND at least 4 of the unsolved runs are STALLED (not still descending).
- **STILL DESCENDING (budget-limited again)**: k <= 2 and fewer than 4 unsolved runs STALLED.
- **PARTIAL (k/8)**: otherwise.
Secondary: accuracy at T = 128 (eval mode and dropout-scale re-scored) and 512; epochs to SOLVED; wrap-aware opposition
(median |wrap(step(+1) + step(-1))| and the share of channels cancelling within 0.1 rad, as RING_DIAL) -- whether solving
runs find the wrap solution or another code.

## Process and cost
Independent code check (rule 29) of the analysis and driver before launch; GPU 0 only (another user holds GPU 1).
8 runs x 900 epochs; RING_DIAL ran ~5,000 run-epochs per hour on GPU 0 -> ~1.5 h.

## Amendment 1 (2026-10-12, after the independent check `docs/audits/2026-10-12/ring_budget_audit.md`; before launch)
- **D1 / B1 (the stall split).** A flat final window after LR decay is not convergence (rule 3), and calibrated on RING_DIAL's
  300-epoch Abs runs the unsolved ones split ~half STALLED, half DESCENDING at similar losses (0.05-0.22): the split is not a
  usable decision. -> Replaced by a within-batch comparison:
- **D2 (the 300-epoch twin).** A 300-epoch Abs_r4 run on the SAME seeds 40-47 is added to the batch (16 runs), so 900 vs
  300 is one batch, same seeds. Stretching the cosine changes the LR at every epoch (integrated LR 493 vs 164): "more budget"
  means the whole stretched schedule; epochs to SOLVED below 300 are not read as "slower".
- **Decision (replaces the four branches above; k900, k300 = SOLVED counts of 8):** SLOWER, NOT STUCK AT THIS BUDGET:
  k900 >= 6, qualified "(by the wrap solution)" if the median share of channels cancelling within 0.1 rad among the
  solved 900-epoch runs is >= 0.5, else "(by another code)" (D3). NOT CONVERGED BY 900 EPOCHS: k900 <= 2. PARTIAL GAIN
  FROM 3x BUDGET: Fisher(k900 vs k300) p < .05 with k900 > k300. PARTIAL otherwise. Secondaries per run: class (RISING
  separated), final-5% and minimum loss, loss at epoch 300, accuracy (eval mode and re-scored) at T = 128 and 512, epochs to
  SOLVED, wrap readout.
- **Minor.** Provenance asserts restored (n_steps, size, epochs per budget); an assert that the re-score hook reached every
  model; lie_groups.py added to the md5 guard; 6 jobs at a time (CPU cores). End-to-end test of the amended analysis on a
  scratch layout of RING_DIAL's 300-epoch Abs runs: passes. (Seen there, descriptive: even RING_DIAL's SOLVED Abs runs cancel
  in only ~4% of channels: they solve by another code, not the wrap solution.)
- Cost: 8 x 900 + 8 x 300 = 9,600 run-epochs -> ~2 h on GPU 0.
