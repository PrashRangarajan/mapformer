# Position effect at grid 32, BOTH ARMS CONVERGED

400 epochs, warmup + cosine, oracle recode, n=3. This is the comparison every earlier MiniWorld table was missing: at 100 epochs with linear decay neither arm reliably converged, so the reported effects measured time-to-solve.

| arm | seed | final loss | slope | flat? | nb_acc T=512 | T=1024 |
|---|---|---|---|---|---|---|
| Vanilla | 0 | 0.0038 | -0.00014/ep | YES | 1.000 | 0.911 |
| RoPE | 0 | 0.4990 | -0.00029/ep | YES | 0.725 | 0.336 |
| Vanilla | 1 | 0.0268 | -0.00012/ep | YES | 1.000 | 0.836 |
| RoPE | 1 | 0.4055 | -0.00032/ep | YES | 0.789 | 0.388 |
| Vanilla | 2 | 0.4302 | -0.00023/ep | YES | 0.781 | 0.489 |
| RoPE | 2 | 0.4334 | -0.00037/ep | YES | 0.748 | 0.391 |

**Repro control:** RoPE s0 retrained = 0.725 vs stored 0.725 (drift +0.000). Cross-batch comparison licensed.

| | Vanilla (path-int) | RoPE (index) | effect |
|---|---|---|---|
| mean nb_acc T=512 | **0.927** | 0.754 | **+0.173** |

**REAL, CONVERGED POSITION EFFECT.** Both arms trained to a flat loss and path integration still wins by more than the measured noise floor (0.150). This is the comparison the headline always needed.

## Precise reading (do not overclaim -- this line has been retracted twice)

**ESTABLISHED.** At grid 32 with BOTH arms converged (6/6 flat, slopes -0.0001 to
-0.0004/ep) and the repro control exact (RoPE s0 retrained 0.725 vs stored 0.725,
drift +0.000, so cross-batch is licensed):
- path-int 0.927 vs index 0.754, effect **+0.173**, 3/3 sign-consistent, above the
  MEASURED noise floor of 0.150 (mean |delta| between two provably function-identical
  models, GateDeltaCtl vs Vanilla, n=9).
- Cleanest framing is the CEILING, not the mean: **path integration SOLVES the task
  (1.000 in 2/3 seeds); index never exceeds 0.789 in any seed, though all arms are
  converged.** A capability difference, not a training-speed one.

**NOT ESTABLISHED.**
1. Uniformity. Per-seed deltas are +0.275 / +0.211 / **+0.033**; only 2/3 clear the
   noise floor and the ranges OVERLAP (RoPE max 0.789 > Vanilla min 0.781). Vanilla
   stays bimodal at convergence: two seeds at loss ~0.01, one flat at 0.43.
2. **THE CROSSOVER.** This is grid 32 ONLY. The crossover claim is about grid SIZE,
   and grid 8 has never been run at this budget -- at 100 epochs there RoPE got 0.977
   and Vanilla 0.448, BOTH unconverged. Whether that reverses when both converge is
   unknown. Until grid 8 is run at 400ep/cosine there is no crossover claim, only a
   grid-32 result.
3. T=1024 is not used here despite showing a larger effect (+0.40): training is 1024
   tokens and eval at T=1024 is 2048, so index RoPE is out-of-distribution BY
   CONSTRUCTION while the path-integrated angle stays grid-bounded. Only T=512 is an
   in-distribution measurement.

**WHAT WOULD COMPLETE IT:** grid 8, both arms, 400ep + cosine, n=3 (6 arms, ~3h --
grid 8 is cheaper). If index still wins there once converged, the crossover is real
and now rests on converged data at both ends. If path-int wins at grid 8 too, there
is no crossover -- just a position effect that the 100-epoch budget had inverted.
