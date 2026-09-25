# Rank separation -- pre-registration (2026-09-25, before any run)

## Why
At matched initialisation (`RANK_MI_RESULTS.md`), within 900 epochs at T=1024: A our shared r=2
0/8 solved, B a per-head r=2 2/8, C a shared r=4 8/8. B and C have the same 4 latent dims and
identical initial W_in but differ in three things at once: the rank of each head's
content-to-angle map, cross-head reading of a shared latent, and W_out's per-entry scale. This
batch separates them.

## New arms (`model_rank_perhead.py`), built from our r=2's base at the same seed like A-C
| arm | variant | per-head rank | heads read | W_out init | bottleneck params |
|---|---|---|---|---|---|
| **C_bd** | `Vanilla_r4mibd` | 2 | own block only | C's diagonal blocks (bound 0.5) | 768 (256 held at 0) |
| **D** | `Vanilla_r4ph` | 4 | own 4-dim latent | per head, bound 0.5 | 1,280 |

Gated before launch: non-bottleneck weights identical to A at the same seed (0.0); C_bd's W_in
and diagonal blocks identical to C's (0.0); C_bd's output identical to a per-head r=2 carrying
its weights (0.0), i.e. it IS a per-head r=2 at C's W_out scale; off-diagonal blocks stay exactly
0 through an AdamW step; causal leak 0. Initial angle-increment std (seed 5): A 0.335, B 0.328,
C 0.313, **C_bd 0.208**, D 0.351 -- zeroing half of C's W_out halves the variance, so C_bd vs B
changes the initial angle scale together with W_out's per-entry scale.

Same recipe as `runs/rank_mi`: T=1024, batch 16, 900 epochs, lr 1e-3, warmup + cosine,
`--data-workers 3`, 8 seeds. A, B, C are the stored `runs/rank_mi` arms (code for them is
unchanged; the model file only gained new classes); **one full reproduction run, C seed 0**,
must match the stored per-epoch losses exactly.

## Readouts and branches (budget-scoped: SOLVED = final-5% loss < 0.05 within 900 epochs)
Each contrast: Fisher on SOLVED counts and exact permutation test on T=1024 accuracy; a contrast
FIRES if either is p < 0.05 (Holm-corrected p over the four also reported); otherwise UNMEASURED.
- **Cross-head reading -- C vs C_bd** (same W_in, same diagonal blocks, off-blocks removed).
- **W_out scale (and initial angle scale) -- C_bd vs B** (both per-head r=2, same W_in).
- **Sharing -- D vs C** (both rank 4 per head; D's heads have separate latents; D has 8 dims).
- **Per-head rank -- D vs C_bd** (both unshared; 4 vs 2 per head).
Reading, fixed now: if cross-head reading does not fire but W_out scale does, the r=4 advantage
is an optimiser-scale effect, not rank; if per-head rank fires and sharing does not, it is the
rank of each head's map; if cross-head reading fires, heads reading each other's latent matters.
Any other pattern is reported as it falls, with no mechanism sentence.
Scope: torus, T=1024, n_heads=2, one recipe, 900-epoch budget.
