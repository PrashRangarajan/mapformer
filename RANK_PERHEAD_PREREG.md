# Paper-faithful per-head r=2 -- 2-seed pilot (2026-09-24, registered before launch)

## Why

Our `ActionToLieAlgebra` shares one r-dim latent across heads; the paper's is per head. At
2 heads: our r=2 = 2 latent dims (384 params), the paper's r=2 = 4 dims with a block-diagonal
W_out (640), our r=4 = 4 dims (768). At T=1024 (900 epochs) our r=4 solved 8/8 and our r=2
0/8, and r=2's deficit is search (`RANK_PROJ_RESULTS.md`). So far nothing has tested the
PAPER'S r=2. This pilot asks which side it falls on.

## Arm (`model_rank_perhead.py`, variant `Vanilla_r2ph`)

Per head h: Delta_h = W_out^h W_in^h x, r=2 each. Gated before launch: 204,629 params (640 in
the bottleneck); every non-bottleneck weight IDENTICAL to our r=2 at the same seed (max diff
0.0 over 22 tensors); output identical to our r=4 with a block-diagonal W_out (max logit
diff 0.0); zero causal leak. W_out^h is initialised like our r=2's W_out (bound 1/sqrt(2)),
so if it behaves like r=4 the init-scale confound is not the explanation.

## Design

Seeds 0 and 1, T=1024, batch 16, 900 epochs, lr 1e-3, warmup + cosine, `--data-workers 3`,
fresh start: the recipe of `runs/rank_matched_e900`, whose r=2 and r=4 seeds 0-1 are the
comparison (same seeds, same data stream). One REPRODUCTION run of our r=2 seed 0 in the
same batch; its per-epoch losses must match the stored e900 run (rule 12's check, since the
comparison arms are stored). Runs `runs/rank_perhead_pilot`.

## Readout (exploratory, n=2: reported, not a verdict)

Run class (Amendment 2: SOLVED / STALLED / DESCENDING), T=1024 accuracy and strata, action
code (opposition), against stored r=2 (s0 STALLED 0.587, s1 DESCENDING 0.051 at 900 ep) and
r=4 (both SOLVED).

- per-head r=2 SOLVED on 2/2 -> points to the SHARED bottleneck (or its dimension count) as
  the cause; "use r=4" would become "use the paper's per-head bottleneck".
- 0/2 -> points to the paper's own r=2 having the search problem.
- 1/2 -> uninformative at n=2.

Any outcome is only a pointer; a full batch (8 seeds, plus our r=2 at r=4's init scale) would
be registered separately and would include these two seeds, whose outcome is then known.
