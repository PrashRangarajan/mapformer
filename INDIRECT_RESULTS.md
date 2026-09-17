# READING

> **SUPERSEDED IN PART (2026-09-17, `INDIRECT_RESULTS_200k.md`)**: at 200,000 iterations PoPE
> solves 7/8 and MapPoPE 8/8, so the paper's result DOES replicate and the solve-rate gap below is a
> budget artefact -- path integration reaches the same solution sooner, it is not needed to reach it.

**The paper's own contrast does not replicate; the 2x2 it does not run gives the clearer result.**

- The task is BIMODAL: a run undergoes a late transition (lift-off between 40k and 80k steps, then
  saturation) or sits flat at ~0.09 for all 100,000 steps. Seed means are therefore not the statistic;
  SOLVE RATE is. Registered in amendment 1 before these seeds ran.
- **Solve rate (accuracy > 0.5): MapPoPE 5/8, PoPE 1/8, MapWM 0/8, RoPE 0/8.** The paper's PoPE is
  implicitly 3/3 and its RoPE 0/3; our RoPE matches, our PoPE does not (1/8, and the one solver ended
  at 0.803 still climbing).
- **Path integration is what makes the solution findable here.** MapPoPE - MapWM is the only
  detectable mean contrast (+0.507, 8/8 seeds) and 5/8 vs 0/8 on solve rate (Fisher p = 0.026).
  MapPoPE vs PoPE is 5/8 vs 1/8, Fisher p = 0.119 -- suggestive, not established at n=8.
- Best single run in the batch: **MapPoPE 0.996**, above the paper's PoPE mean of 0.948. Four MapPoPE
  seeds exceed 0.96. So the ceiling is reachable; what varies is whether training finds it.
- MapWM - RoPE is +0.024 (8/8, MDE 0.020) DETECTABLE but it is a plateau-height difference
  (0.092 vs 0.068), not a solution: no MapWM seed ever transitions.
- **Why our PoPE under-solves is unresolved**, and two suspects are live: this is a reimplementation
  (delta clamped to [-2pi, 0], LayerNorm rather than the paper's RMSNorm) and the budget binds --
  the solving PoPE seed was still rising at 100k. Neither is tested here. Until one is, read this
  batch as "path integration raises the solve rate of OUR PoPE", not as a claim about theirs.

