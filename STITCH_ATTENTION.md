# Stitch: attention-map probe

CSCG reports the stitching experiment qualitatively ("Predictive performance on the stitching of the two rooms is perfect") and evaluates it by inspecting the learned transition matrix. MapFormer exposes no such matrix, so this ports the *evaluation style* -- inspect the learned structure -- to attention.

`join_share` = attention (per token) on phase-A tokens at the JOIN cell, over the sum of that and the same for the CONFOUNDER cell. The two cells emit the **same symbol**. Layer 0 is the clean readout: its keys are the token embedding rotated by the path-integrated angle and nothing else, so two same-symbol tokens differ there only in position.

**The statistic is the PAIRED DIFFERENCE, whose floor is exactly 0.** Both arms are scored on the same two cells against a literally identical phase-A prefix, so visit counts, recency and token identity cancel between them. The per-arm 0.5 is weaker: within a single episode one look-alike happened to be seen more often or more recently, so 0.5 holds per arm only in expectation over episodes.

Approach paths are random walks of EQUAL length confined to room A in both arms, so neither arm re-enters room B or crosses the junction before the measurement.

The bootstrap CI below resamples EPISODES and so understates the uncertainty that matters; the per-seed line under each table is the honest n=3 replication and should be read first.

## MapWM-Flat  (n=3 seeds, 200 episodes each)

| layer | join arm | confound arm | paired diff | episodes with diff>0 | B-share join | B-share confound |
|---|---|---|---|---|---|---|
| 0 | 0.5772 | 0.4466 | **+0.1306** [+0.1047, +0.1560] | 0.653 | 0.4776 | 0.4581 |
| 1 | 0.5622 | 0.4258 | **+0.1364** [+0.0976, +0.1762] | 0.713 | 0.5012 | 0.4965 |
| 2 | 0.5920 | 0.3876 | **+0.2044** [+0.1634, +0.2451] | 0.763 | 0.5760 | 0.5151 |

**Transitive tail** (layer 0). One action sequence, legal from both look-alike cells, replayed by both arms: from the join it enters room-B-only cells (4.8 of 6 steps on average), from the confounder it cannot leave room A. Value = B-share, massB/(massA+massB) over explore-phase observation tokens, read at each step's ACTION token (before that step's observation is visible). Unlike `join_share` this has no analytic floor -- the confound arm is its empirical floor, and the arms' tokens are matched only from the shared prefix, not step for step. Mean approach-path length: join 5.9, confound 5.9 steps.

| tail step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| join arm | 0.4868 | 0.4810 | 0.4838 | 0.4811 | 0.4823 | 0.4889 |
| confound arm | 0.4637 | 0.4740 | 0.4590 | 0.4626 | 0.4603 | 0.4580 |
| diff | +0.0231 | +0.0070 | +0.0247 | +0.0185 | +0.0221 | +0.0309 |

**Retrieval concentration** (layer 0): mean attention per token on the explore-phase tokens of the cell the agent is standing on, divided by mean attention per token over the SAME PHASE. **1.0 = no retrieval.** `B-share` above sums over all 256 phase-B tokens and is dominated by diffuse background, so it cannot see a few sharp peaks; this is the sensitive version. The baseline is taken within phase because phase B sits nearer the query than phase A: normalising against both phases together makes every room-B cell look retrieved from recency alone, for PlainFlat as much as for MapFormer.

| tail cell | concentration | n |
|---|---|---|
| join arm, room-B-only cells (**the transitive case**: reached through room A, only ever seen in phase B) | **2.98x** | 2625 |
| join arm, room-A cells | 4.53x | 723 |
| confound arm, room-A cells (within-room retrieval — the yardstick) | 3.32x | 3501 |

per-seed, join arm room-B cells: 0: 2.59x, 1: 1.56x, 2: 4.79x
per-seed, confound arm room-A cells: 0: 2.65x, 1: 3.27x, 2: 4.04x

### Per-seed, layer 0 (n=3 — read this before the CI)

| seed | join arm | confound arm | paired diff | tail diff | held-out revisit acc |
|---|---|---|---|---|---|
| 0 | 0.5554 | 0.4521 | +0.1033 | +0.0057 | 0.9712 |
| 1 | 0.5887 | 0.4392 | +0.1495 | +0.0367 | 0.9707 |
| 2 | 0.5874 | 0.4486 | +0.1388 | +0.0207 | 0.9739 |
| **mean ± sd** | **0.5772 ± 0.0189** | **0.4466 ± 0.0067** | **+0.1306 ± 0.0242** | **+0.0210 ± 0.0155** | |

## PlainFlat  (n=3 seeds, 200 episodes each)

| layer | join arm | confound arm | paired diff | episodes with diff>0 | B-share join | B-share confound |
|---|---|---|---|---|---|---|
| 0 | 0.4956 | 0.5009 | **-0.0053** [-0.0184, +0.0083] | 0.510 | 0.3845 | 0.3768 |
| 1 | 0.4597 | 0.4586 | **+0.0011** [-0.0063, +0.0085] | 0.552 | 0.4283 | 0.4241 |
| 2 | 0.5029 | 0.4730 | **+0.0298** [+0.0069, +0.0526] | 0.607 | 0.5665 | 0.5121 |

**Transitive tail** (layer 0). One action sequence, legal from both look-alike cells, replayed by both arms: from the join it enters room-B-only cells (4.8 of 6 steps on average), from the confounder it cannot leave room A. Value = B-share, massB/(massA+massB) over explore-phase observation tokens, read at each step's ACTION token (before that step's observation is visible). Unlike `join_share` this has no analytic floor -- the confound arm is its empirical floor, and the arms' tokens are matched only from the shared prefix, not step for step. Mean approach-path length: join 5.9, confound 5.9 steps.

| tail step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| join arm | 0.4082 | 0.4135 | 0.4157 | 0.4143 | 0.4203 | 0.4220 |
| confound arm | 0.4061 | 0.4127 | 0.4147 | 0.4140 | 0.4204 | 0.4221 |
| diff | +0.0022 | +0.0008 | +0.0009 | +0.0003 | -0.0001 | -0.0001 |

**Retrieval concentration** (layer 0): mean attention per token on the explore-phase tokens of the cell the agent is standing on, divided by mean attention per token over the SAME PHASE. **1.0 = no retrieval.** `B-share` above sums over all 256 phase-B tokens and is dominated by diffuse background, so it cannot see a few sharp peaks; this is the sensitive version. The baseline is taken within phase because phase B sits nearer the query than phase A: normalising against both phases together makes every room-B cell look retrieved from recency alone, for PlainFlat as much as for MapFormer.

| tail cell | concentration | n |
|---|---|---|
| join arm, room-B-only cells (**the transitive case**: reached through room A, only ever seen in phase B) | **1.16x** | 2625 |
| join arm, room-A cells | 1.53x | 723 |
| confound arm, room-A cells (within-room retrieval — the yardstick) | 1.41x | 3501 |

per-seed, join arm room-B cells: 0: 1.25x, 1: 1.11x, 2: 1.13x
per-seed, confound arm room-A cells: 0: 1.50x, 1: 1.24x, 2: 1.50x

### Per-seed, layer 0 (n=3 — read this before the CI)

| seed | join arm | confound arm | paired diff | tail diff | held-out revisit acc |
|---|---|---|---|---|---|
| 0 | 0.4961 | 0.5011 | -0.0050 | -0.0004 | 0.5963 |
| 1 | 0.5134 | 0.5030 | +0.0104 | +0.0037 | 0.6063 |
| 2 | 0.4773 | 0.4986 | -0.0213 | -0.0012 | 0.5715 |
| **mean ± sd** | **0.4956 ± 0.0180** | **0.5009 ± 0.0022** | **-0.0053 ± 0.0158** | **+0.0007 ± 0.0026** | |


## What this does and does not show

Shows: the model tells apart two cells that emit the same symbol and sit in an identical shared prefix, and it does so in the right direction at both of them. And it retrieves a room-B memory of a cell it reached through room A's frame nearly as sharply as it retrieves a within-room memory. That is CSCG's negative control and its transitive claim, on the only structure MapFormer exposes.

Does not show:

- **That the discrimination is specifically path integration.** The two arms differ in the observations along their approach as well as in position, so the model may be localising from that content. Position is one component of the context it uses, not demonstrably the whole of it. What the PlainFlat row establishes is that index positions plus the same content are not enough at this training budget.
- **A clean architecture comparison.** PlainFlat reaches 0.57-0.61 held-out revisit accuracy against MapWM-Flat's 0.97. It is the much weaker model, so 'PlainFlat shows no effect' is entangled with 'PlainFlat did not learn the map'. Parameters are matched (600,212 vs 600,660); capability is not.
- **A stable transitive magnitude.** The per-seed transitive concentrations spread widely, and on one seed the stitched retrieval is well below that seed's own within-room yardstick. Every seed is above the PlainFlat range, so the direction replicates; the size does not.
- **Anything CSCG measured.** CSCG reports this experiment with no number at all, so none of these values can be compared to a published one. This is a port of their evaluation style, not a reproduction of their result.

