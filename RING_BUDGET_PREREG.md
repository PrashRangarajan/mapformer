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
