# Does a non-negative rotary step learn a content clock faster? -- results (2026-10-12)

Pre-registration `CLOCK_SPEED_PREREG.md` + Amendment 1 (independent check: n = 16 for the primary contrast, CI-based NO
DIFFERENCE). Batch 09:32-11:08 at d081254 on GPU 0; k-back recency (k_max 128, T = 1024, 300 epochs), Signed_r4 and Abs_r4
on seeds 210-225, CARoPE_r4 on 210-217 (40 runs). Primary readout: AUC = mean training loss over the 300 epochs (lower =
learned faster), paired by seed. Registered output `CLOCK_SPEED_ANALYSIS.txt`, `CLOCK_SPEED_VERDICTS.json`.

| arm | n | AUC (mean training loss) | epoch to running loss < 2.0 / < 1.0 (median) | loss at 100 / 200 / 300 | accuracy T (2T) | SOLVED |
|---|---|---|---|---|---|---|
| Signed_r4 | 16 | 2.235 +/- 0.289 | 210 / 301 | 2.76 / 2.06 / 1.06 | 0.699 (0.687) | 0/16 |
| Abs_r4 | 16 | 2.039 +/- 0.199 | 172 / 270 | 2.72 / 1.49 / 0.90 | 0.723 (0.690) | 0/16 |
| CARoPE_r4 | 8 | 2.121 +/- 0.266 | 170 / 257 | 2.69 / 1.43 / 0.80 | 0.720 (0.619) | 0/8 |

## Registered
- **S1, primary (Abs - Signed, the sign alone, n = 16): UNMEASURED** -- paired mean -0.197 (exact sign-flip p 0.053; 95% CI
  [-0.390, -0.000]); Abs faster on 12/16 seeds. Narrowly short of the registered threshold; the direction is the
  non-negative step's.
- **S2 (CARoPE - Signed, n = 8): UNMEASURED** -- paired mean -0.018 (p 0.90).

## Secondaries (declared; no verdict)
- Unpaired: Abs - Signed -0.197 (perm p 0.038, CI [-0.380, -0.015]); CARoPE -0.018 (p 0.91).
- Both non-negative arms leave the chance plateau earlier (median epoch 170-172 vs 210) and are lower at epoch 200 (1.43-1.49
  vs 2.06); by epoch 300 the gap narrows (0.80-0.90 vs 1.06). No run converged; r(AUC, accuracy) -0.85.
- On CLOCK_SIDE's seeds (post hoc) CARoPE was faster on 8/8; here 4/8: the CARoPE contrast did not hold up on fresh seeds.

## What it means
- **The clock side of a learnability double dissociation is suggestive, not established.** On the map task (the ring) the
  signed step learns faster than the non-negative one on 8/8 paired seeds at every dial setting (loss-AUC companion,
  `docs/audits/2026-10-12/ring_auc_companion_out.txt`: +0.23 to +0.35, p 0.0078 each; SOLVED 8/8 vs ~3/8, RING_DIAL /
  RING_BUDGET). On the content-clock task the non-negative step is faster on 12/16 seeds by about the same amount (-0.20),
  but the registered paired test gives p 0.053. Read: an asymmetric learnability effect -- clear where a map is needed,
  a trend in the opposite direction where a content clock is needed.
- Both sides are learnability: the arms can represent the same codes (rotary steps wrap mod 2 pi).

## Caveats
- One clock task, 300 epochs, nothing converged, 1 layer; AUC is progress within this budget (dropout inflates training
  loss by ~0.15-0.23 by arm). The ring and recency sides differ in task, length and model details; the matched measure (AUC)
  is computed post hoc on the ring.
