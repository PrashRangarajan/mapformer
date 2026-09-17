# READING -- the budget was the explanation (amendment 2, 200,000 iterations)

- **R5 CONFIRMED: PoPE solves 7/8 at 200k**, against 1/8 at 100k. Among its solvers the mean is
  0.965, against the paper's 0.948 +/- 0.029. **The paper's Indirect Indexing result replicates**;
  our 100k failure was the budget, not the reimplementation. The delta clamp and LayerNorm-vs-RMSNorm
  suspects are exonerated for this task.
- **R6 CONFIRMED: MapPoPE holds and rises, 8/8** (5/8 at 100k), mean 0.921.
- **R7: the gap closes.** 8/8 vs 7/8, Fisher p = 1.0. Path integration's 100k advantage (5/8 vs 1/8)
  was a SPEED effect, not a capability difference: given enough budget PoPE finds the same solution.
- Lift-off (first evaluation above 0.5): MapPoPE median 70k, PoPE median 100k; paired on the 7 seeds
  both solve, MapPoPE is 15k steps earlier on 6/7 seeds -- but sd 52.6k against an MDE of 55.7k, so
  **unmeasured at n=7**. The direction is consistent, the size is not established.
- Two seeds lifted off after 135k, i.e. beyond the previous budget entirely; one PoPE seed (0.098)
  never lifted off at all in 200,000 steps.

**What this changes in the 100k reading.** "Path integration raises the solve rate 5/8 vs 1/8"
stands as a statement about that budget and is now explained: both encodings reach the same solution,
path integration reaches it sooner. It is not evidence that path integration is necessary here.
The corrected claim is: on this task path integration buys optimisation speed, and the end state is
PoPE's.

