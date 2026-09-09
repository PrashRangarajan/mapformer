# Does "hierarchy buys compositional transfer" survive the recipe?

The published claim is `MapWM-Hier - MapWM-FlatHG = +0.130` (0.415 vs 0.285),
measured entirely under the old recipe -- whose own effect is **+0.160**
(`COMP_HEADROOM.md`). Both arms retrained here in ONE batch at cosine/1e-3/150ep,
8 seeds, same evaluator and `n_traj=200` as the published table.

| arm | cross_nb T=256 | cross_nb T=512 |
|---|---|---|
| `MapWM-Hier` | **0.514 ± 0.126** | **0.390 ± 0.126** |
| `MapWM-FlatHG` | 0.378 ± 0.103 | 0.239 ± 0.086 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| hierarchy, T=256 | **+0.136** | 0.174 | 0.173 | 7/8 | unmeasured |
| hierarchy, T=512 | **+0.151** | 0.157 | 0.155 | 7/8 | unmeasured |

## Reading it

**The claim survives in size and direction.** +0.136 against the published +0.130
-- essentially unchanged -- on 7 of 8 seeds at both lengths. So the recipe result
does *not* dissolve it: the effect is not an artefact of undertraining, and both
arms rose together (hier 0.415 -> 0.514, flat 0.285 -> 0.378).

**And it is still not established.** Both contrasts land just inside their own
\textsc{mde} (0.136 vs 0.173; 0.151 vs 0.155). At this seed variance n=8 cannot
resolve an effect of this size. That was equally true of the published number,
which was cross-batch and never carried an \textsc{mde} at all -- so this is not a
downgrade, it is the first honest power statement about a claim that has been
quoted as settled.

**What it would take.** sd 0.174 at n=8 gives \textsc{mde} 0.173; reaching
\textsc{mde} < 0.136 needs about **n=13**, and n=16 would give 0.122 with margin.
That is 16 more runs at ~75 min each -- the cheapest remaining way to move a
standing claim from "quoted" to "measured".

## Status of the claim

Neither retracted nor confirmed. It reproduces at the recipe that matters, at the
size originally reported, with the direction on 7/8 seeds — and it has never been
measured at sufficient power, then or now. Cite it as *directional, n=8,
unmeasured*, not as +0.130.
