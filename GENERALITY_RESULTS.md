# Generality: the torus existence test and published monotone methods -- results (2026-10-14)

Pre-registration `GENERALITY_PREREG.md` + Amendment 1 (independent check `docs/audits/2026-10-14/generality_audit.md`). Batch
2026-10-14 01:05-12:00 at 04adccb on both GPUs (alongside MECHANISM). Seeds 700-707, n = 8. Torus 64x64, T = 1024, 900 epochs
(continuations 300), --fast-attn; ring 32, p_plus 0.5, T = 128, 300 epochs. Registered output `GENERALITY_ANALYSIS.txt`,
`GENERALITY_VERDICTS.json`; evals `GENERALITY_TORUS_*.json`, strata `*_STRATA.json`, re-scores `*_RS.json`.

## A: the existence test on the 2D torus (observation-step-free models, only the four moves have steps)
| arm | what | acc T = 1024 (held-out map) | SOLVED | peak training loss in the continuation |
|---|---|---|---|---|
| SO4 | signed, from scratch | 1.000 | 8/8 | -- |
| AO4 | non-negative, from scratch | 0.880 (0.71-0.96) | **0/8** | -- |
| SO_bal | SO4 rebuilt at the balanced gauge, +300 epochs at lr 1e-4 | 1.000 | 8/8 | 0.004-0.017 |
| AO_bal | SO4 converted exactly to non-negative steps, +300 epochs at lr 1e-4 | 1.000 | 8/8 | **0.110-0.208** |
| SO_bal_hi / AO_bal_hi | the same at lr 1e-3 (descriptive) | 1.000 / 0.988 | 8/8 / 6/8 | 0.15-0.21 / 0.46-0.63 |
Floors (RANK_MI stream): blank 0.507, best n-gram 0.576.

**Registered.**
- **P0:** SO4 solves 8/8. **E1 EXISTS** for both constructions (frozen = source on 8/8; angle error <= 1.4e-6 rad).
- **E2 (primary, lr 1e-4): NOT HELD** by the LOST rule -- AO_bal leaves the solved region on 6/8 seeds (peak loss 0.11-0.21
  at epochs 13-52) while SO_bal never does (Fisher on not-LOST p 0.0070), and then re-finds it (loss below 0.05 for good from
  epoch 119-204; 8/8 SOLVED at the end, accuracy 1.000 = SO_bal). By the reading fixed before launch (Amendment 1 A3):
  **perturbed and re-found, not held in place.** The lr 1e-3 pair (descriptive): NOT HELD, 6/8 re-solve.
- **E3: SIGNED LEARNS BETTER** -- AO4 0/8 vs SO4 8/8, -0.120 (perm p 0.0002); AO4 slower on 8/8 paired seeds (loss AUC +0.54,
  p 0.0078).
**Secondaries.** AO4 fails long-gap and wrap-only revisits (0.928 / 0.709 / 0.585 for lag < 128 / >= 128 / wrap-only); AO_bal
is 1.000 on every stratum. Wrap share at the end: SO4 1.00, AO4 0.18; AO_bal 1.00 at construction -> 0.69 after training
(ring: 0.87 -> 0.18). W_out and omega move by similar relative amounts in both continuations (0.07-0.09).

## B / C: published monotone methods (faithful CARoPE and CoPE; in-batch index RoPE)
| task | arm | acc | SOLVED | vs Signed_r4 (B) | vs index RoPE (C) |
|---|---|---|---|---|---|
| torus T 1024 | Signed_r4 (MapWM) | 1.000 | 8/8 | -- | -- |
| | CARoPE | 0.667 | 0/8 | MONOTONE FAILS (sanity: cannot represent) | **WORSE THAN INDEX** -0.055 (p 0.040) |
| | RoPE (index) | 0.722 | 0/8 | -- | -- |
| ring p_plus 0.5 | Signed_r4 | 1.000 | 8/8 | -- | -- |
| | CARoPE | 0.741 | 0/8 | MONOTONE FAILS (sanity) | UNMEASURED +0.011 (p 0.39) |
| | **CoPE** | **0.887** | **0/8** | **MONOTONE FAILS** -0.113 (p 0.0002; a learnability test: CoPE can represent ring cancellation) | **BETTER THAN INDEX** +0.157 (p 0.0002) |
| | RoPE (index) | 0.730 | 0/8 | -- | -- |
CoPE was not run on the torus (its gated count at 2048 tokens costs ~8 h per run). Dropout re-score changes no label (all
within 0.01).

## What it means
- **The existence picture carries to the torus, with one difference.** An exact non-negative solution exists at a trained
  scale and training does not lose the task (8/8 SOLVED, 1.000 at the end), while the non-negative class never finds it from
  scratch (0/8 vs 8/8). Unlike the ring, the torus continuation is knocked out of the solved region early (peak loss ~0.1-0.2)
  before re-finding it: the non-negative solution is re-found from nearby, not held in place. Ring and torus E2 labels are not
  directly comparable (continuation lr 1e-3 vs 1e-4).
- **Published monotone content-dependent methods do not learn these maps either.** CARoPE cannot represent cancellation and
  is no better than (torus: slightly worse than) index RoPE; CoPE can represent ring cancellation but does not learn it in
  300 epochs (0/8), though its content-gated count beats the index by +0.16 on the ring.

## Caveats
- One recipe per task; n = 8; 1 layer. CoPE ring only. CARoPE's bias makes it start at index RoPE (the paper's stated init).
- E2's NOT HELD is the pre-registered rule reading a transient excursion (pre-stated as "perturbed and re-found"); the end
  state equals the control's.
