# Rank 3 per head -- pre-registration (2026-09-27, before any run)

## Why
`RANK_SEP_RESULTS.md`: on the torus at T=1024 within 900 epochs, per-head rank 2 solves 0/8, 2/8,
2/8 (three arms) and per-head rank 4 solves 8/8, 8/8. Nothing lies between. The torus has two
degrees of freedom. If a per-head rank of 3 solves like 4, the story is "rank must EXCEED the
task's degrees of freedom" (rank 2 = exactly the DOF is the hard case); if it solves like 2, it is
"rank must reach 4" (e.g. two independent axes per head need more than one spare direction).

## New arm (`model_rank_perhead.py`, `MapFormerWM_PerHead3`), built from our r=2's base at the same seed
| arm | variant | per-head rank | latent dims | bottleneck params |
|---|---|---|---|---|
| B (stored) | `Vanilla_r2ph` | 2 | 4 | 640 |
| **E (new)** | `Vanilla_r3ph` | 3 | 6 | 960 |
| D (stored) | `Vanilla_r4ph` | 4 | 8 | 1,280 |

Gated (`docs/audits/2026-09-27/gate_rank3.py`): all 22 non-bottleneck tensors identical to A
(`Vanilla`) at seeds 0 and 5 (max diff 0.0); causal leak 0. Initial angle-increment std on random
tokens, seed 0 / 5: A 0.337 / 0.330, B 0.374 / 0.328, **E 0.345 / 0.341**, D 0.359 / 0.353 -- E sits
inside the range of the other arms, so initial angle scale does not distinguish it.

Recipe exactly as `runs/rank_sep`: T=1024, batch 16, 900 epochs, lr 1e-3, warmup + cosine,
`--data-workers 3`, `--save-full-state`, 8 seeds. B and D are the stored arms (their code is unchanged;
the model file only gained a class); **one full reproduction run, D seed 0**, must match the stored
per-epoch losses exactly, or the batch is void as a comparison with stored arms.

## Readouts and branches (budget-scoped: SOLVED = final-5% loss < 0.05 within 900 epochs)
Two contrasts, each by Fisher on SOLVED counts and exact permutation test on T=1024 accuracy
(`stats_core`); a contrast FIRES if either p < 0.05, else UNMEASURED. Holm over the two reported.
- E vs B (rank 3 vs 2), E vs D (rank 3 vs 4).
Reading, fixed now:
- **RANK 3 SUFFICES** -- E vs B fires, E vs D does not, and E >= 6/8 SOLVED.
- **RANK 3 IS NOT ENOUGH** -- E vs D fires, E vs B does not, and E <= 2/8 SOLVED.
- **GRADED** -- both fire (3 is strictly between).
- Anything else: reported as it falls, no mechanism sentence.
Note on power: with n=8, Fisher needs 8/8 vs 2/8 (p 0.007) or 7/8 vs 1/8 (0.010); 6/8 vs 2/8 is
p 0.13 -- a 6/8 E would be UNMEASURED against B on count and rest on accuracy.
Secondary: T=512 / 2048; strata at T=1024; action geometry in angle space.

Void: the reproduction differs; any of the 9 runs missing; the md5 guard trips.
Scope: torus, T=1024, n_heads 2, d 128, one recipe, 900-epoch budget.
Cost: ~2 s/epoch solo, ~30-45 min per run; 9 runs over 4 slots ~ 2 h.
