# Is the non-negative rotary step slower, or stuck? -- results (2026-10-12)

Pre-registration `RING_BUDGET_PREREG.md` + Amendment 1 (independent check: a 300-epoch twin on the same seeds replaced an
unreliable stall split). Batch 07:28-09:24 at f9ee377 on GPU 0; Abs_r4 on the 1D ring at p_plus 0.5, seeds 40-47, at 900 and
300 epochs (16 runs). Registered output `RING_BUDGET_ANALYSIS.txt`, `RING_BUDGET_VERDICTS.json`.

## Registered: PARTIAL -- three times the training does not help
| budget | SOLVED | acc T = 128 (re-scored) | acc T = 512 (OOD) | share of channels cancelling mod 2 pi (median) |
|---|---|---|---|---|
| 900 epochs | 3/8 | 0.980 (0.980) | 0.778 | 0.07 |
| 300 epochs (same seeds) | 3/8 | 0.967 (0.967) | 0.817 | 0.07 |
Fisher 3/8 vs 3/8, p 1.00. For comparison (other batch, other seeds): the signed step solved 8/8 at 300 epochs with 82%
of channels cancelling (`RING_DIAL_RESULTS.md`).

## Secondaries (declared; no verdict)
- More training lowers the loss of most runs (loss at epoch 300 vs final: e.g. 0.205 -> 0.002, 0.286 -> 0.129) but converts
  no more of them; the solved sets differ (900: s40, s43, s46; 300: s43, s45, s46).
- **No run finds the wrap solution**, solved or not: 2-11% of channels cancel within 0.1 rad (median 0.07), against 82% for
  the signed step. Solved Abs runs solve by another, non-cancelling code.
- That code generalises worse with more training: T = 512 accuracy 0.778 at 900 epochs vs 0.817 at 300 (one run falls to
  0.439).

## What it means
- Across three batches on the ring (RING_DIAL 300 epochs 2/8; here 3/8 at 300 and 3/8 at 900), a non-negative rotary step
  converges on about a third of seeds where the signed step converges on all, and three times the budget does not change
  that. Together with the wrap readout: **the handicap is not speed. Training reliably fails to find the cancelling
  (wrapped) map that the non-negative step can represent, and settles on a non-cancelling code that sometimes fits the
  training length but extrapolates worse.**
- For the framing: in the rotary slot, signed and non-negative steps represent the same maps, but only the signed one is
  learned as a map. Learnability, not representation, separates Howard's Cases II and III here.

## Caveats
- One task family (ring, 32 cells, T = 128), one arm, n = 8 per budget; the stretched cosine changes the LR at every epoch;
  the signed comparison is across batches and seeds (descriptive). The non-cancelling code's mechanism is not identified.
