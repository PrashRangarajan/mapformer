# Does the sign cost follow the relative sign of the moves? (T1's corollary) -- pre-registration (2026-10-15, before the batch)

## Question
Theorem 1 of the theory appendix (relative-sign gauge, verified bitwise): flipping every step of one rotary channel and
reflecting that channel's query/key plane leaves the model unchanged. Corollary: the |.| parameterisation removes only the
RELATIVE sign between tokens that step in the same channel. Prediction: a non-negative step pays no sign cost on a ring map
whose two moves step in the SAME direction (+1, +2: the signed solution a(+2) = 2 a(+1) mod 2 pi has both angles of one sign
in every channel, so it lies in the |.| class up to the gauge), but pays it when they step in opposite directions (+1, -1).
(Mock review of draft 3, `docs/paper/MOCK_REVIEW_draft3.md`, suggested this as the cheapest decisive test.)

## Design (`environment_steps.py`, `train_gauge.py`; ring 32 cells, RING_DIAL recipe: T = 128, batch 128, 98 batches, lr 1e-3,
cosine, 300 epochs, 1 layer, 2 heads, d 128; seeds 1300-1307, n = 8)
| cell | moves | model |
|---|---|---|
| SO_m / AO_m | +1, -1 (environment_cancel's walk, reproduced exactly) | SignedAO_r4 / AbsAO_r4 |
| SO_p / AO_p | +1, +2 (both forward; revisits only from laps; revisit rate 0.75 as for +1/-1) | SignedAO_r4 / AbsAO_r4 |
| RoPE_p | +1, +2 | index RoPE (does the task need integration?) |
Observation-step-free models, so only the two moves step and the corollary applies exactly. Evaluation: train_cancel.evaluate
(fresh stream, 400 trajectories, T = 128 primary); SOLVED = classify_run (final 5% of training loss < 0.05).

## Registered decisions (`analyze_gauge.py`; Fisher on SOLVED, unpaired permutation on accuracy; labels as M4 of MECHANISM)
- **G1 (primary): AO_p vs SO_p** -- NO SIGN COST (AO SOLVED >= SO - 1 and mean >= SO - 0.01) / SIGNED LEARNS BETTER (Fisher
  p < .05 with SO ahead, or perm p < .05 with d <= -0.05) / UNMEASURED. Gate: UNREADABLE if SO_p SOLVED < 4.
- **G2: AO_m vs SO_m**, same labels (gate SO_m SOLVED >= 4); replication of the sign cost.
- **Reading, fixed now:** GAUGE PREDICTION CONFIRMED if G1 = NO SIGN COST and G2 = SIGNED LEARNS BETTER; REFUTED if G1 =
  SIGNED LEARNS BETTER; otherwise as falls.
Prediction: CONFIRMED.
Secondaries (declared): RoPE_p (index fails?), accuracy at 4T, loss AUC, and for SO_p the share of channels whose two move
angles have the SAME sign at the end (the corollary says the signed model need not use opposite signs here).

## Process and cost
Pilot (seed 1390, outside the batch): SO / AO / RoPE on +1/+2 (reported, never pooled). Independent code check before launch.
40 ring runs, ~10-15 min each: ~1 h on both GPUs.
