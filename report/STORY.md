# STORY: what the results report argues, and how it is organised

Editor's decision document, written 2026-09-13 from the six inventories (`report/inventory/A-F`),
the brief, and the framing documents. The inventories win on facts. Where a number is quoted, the
inventory ID and the primary file are named. Where two sources disagreed, the primary file was
opened, and that is recorded in Section 6. This document does not write the report.

Status labels follow `INVENTORY_BRIEF.md`: CITABLE, DIRECTIONAL, POWERED NEGATIVE, EXPLORATORY,
PRIOR-ART REPLICATION. "Unmeasured" means inside the MDE, never "null".

---

## 1. The thesis

### 1.1 Candidate storylines

**(a) "What a content-dependent phase must do to be a map", organised by axes.**
- Spine: sign (E11, E12), rank (E1, E2, E5), the task setting the value (E24-E26), and add-ons that
  do not pay (E15, E18, E19).
- Load-bearing results:
  - Abs - Signed -0.280 loss-matched at T=1024 (E11, `SIGN_ABLATION.md`).
  - r=4 over r=2 +0.085 at T=1024, 8/8 (E1, `RANK_SWEEP.md`).
  - Index cannot count, +0.750 (E24, `RECENCY_RESULTS.md`).
  - The content gate by intervention, +0.594 (E26, `RECENCY_GATE_ABLATION.md`).
- Weakest link: the headline axis is not ours.
  - Sign is a prior-art replication (Sarrof, Grazzi, Selective RoPE Sec. 4.2).
  - Counting is CoPE's claim.
  - Rank is shown on MapWM on the torus only. It does not transfer measurably to MapPoPE (E8), and
    its geometric account is refuted (E7).
  - The recency half of the crossover is unmeasured.
  - Every accuracy effect lives at OOD length, which is the project's unexplained universal
    signature.
- What gets cut: the environment line, the EM/WM line, correction, hierarchy.
- Verdict: tidy but thin. It is the existing `axes_measured.tex` again, and most of its novelty is
  denied by the corpus.

**(b) "When does a structural position code pay", organised by environment and task.**
- Spine:
  - The paper-task 2x2 (A3).
  - Match-Query (C09, C12) and parity (C18).
  - Rotation actions and allocentric recoding (D11, D12).
  - MiniGrid (D6, D8).
  - MiniWorld map extent and the aliasing falsification (D19-D21).
- Load-bearing results:
  - Position +0.461 vs encoding +0.003 (A3).
  - Rotate +0.050 vs allocentric +0.488, n=8 (D11).
  - Allocentric plus r=4 beats index by +0.034 (MDE 0.014, 8/8) on MiniGrid (D8).
  - MiniWorld grid 32 +0.178, n=5 (D20).
- Weakest links:
  - The headline 2x2 uses the 16-epoch LinearLR recipe, under which the index arms are known to
    be undertrained. RoPE reaches 0.799 at T=128 under the converged torus recipe (E11).
  - The map-extent threshold is a post-hoc pooling at n=3, and two of its points exist only as raw
    JSON (D21).
  - The MiniGrid effect is small.
- What gets cut: the EM/WM line, most of the phase-axis mechanism work.
- Verdict: closest to the stated research goal (transfer to a redrawn map). On its own it says
  *when* and not *what the integrator needs*.

**(c) "EM vs WM: kernel sharing, expressivity vs learnability".**
- Spine: F1 (-0.375), existence (F9), holdability (F10, F11), search (F12, F14, F16, F25),
  per-pair origins (F17-F19), phase freedom (F8), map-task ties (F21, F23), and the qualified
  paper-task length advantage (F20).
- Load-bearing results:
  - EM_P0 - WM -0.375 (MDE 0.154, 0/8).
  - Frozen install 1.000, 8/8.
  - Per-pair total +0.215 = pathway +0.124 + freedom +0.091 at n=48.
  - Freedom, not capacity, abandons the per-token rewind (0.948 -> 0.189).
- Weakest links:
  - One task (varying-k recency).
  - Every contrast is a fit contrast (r(loss, acc) -0.936 to -0.986), so "search, not
    representation" rests on existence constructions, not on loss-matching.
  - "Holds" was shown with `w_in` content columns pinned.
  - WM at the 4x budget was never run.
  - Phase freedom's mechanism is unidentified.
- What gets cut: environments, correction, hierarchy, loops.
- Verdict: the most genuinely ours (no paper in the corpus isolates shared vs per-pair kernels) and
  directly about the where/what conjunction (MapEM's Hadamard is TEM's `g (x) x`). It is also the
  youngest line, and a report built only on it would rest on one task.

**(d) "A learned path integrator becomes a transferable map under three conditions: the input must
make each token's displacement fixed, the increment must be able to cancel with a well-conditioned
basis, and training must find the solution."**
- This combines (b), (a) and (c), each as one condition, with the negatives as a fourth part: what
  bolted-on machinery adds.
- Load-bearing: the CITABLE core of each of the above, restricted to results established by
  intervention or at n>=8 in one batch.
- Weakest link: it is a conjunction of separately established claims on different tasks, not one
  experiment. No single task shows all three conditions binding at once. The "conditions" framing
  must not be read as sufficient conditions, or as an exhaustive list.
- What gets cut: every n<=3 line that does not bear on a condition (April baselines, TEM variants,
  PC/NoDrop/GSF/Cascade, early hierarchy, hex probes, n=1 MiniGrid and continuous-nav work).

### 1.2 Choice

**Storyline (d), with (b) as the entry point and (c) as the part that is most ours.** The reasons:

1. It is the only storyline that serves the stated goal end to end.
   - The goal is a relational where, kept separate from the what, that transfers.
   - (b) says when the structural where transfers at all.
   - (a) says what the integrator must be able to represent for that.
   - (c) asks what happens when the where is fully factorised from the what, as in TEM and MapEM,
     and the answer is "representable, but harder to find".
2. The three conditions share one mechanism statement.
   - Attention reads the interval sum of the per-token increments.
   - Condition 1 is about whether a fixed per-token increment can equal the displacement.
   - Condition 2 is about whether the interval sum cancels, so that it measures net displacement.
   - Condition 3 is about whether gradient descent reaches increments that do so.
   - That makes it a story, not a list.
3. It lets the powered negatives do real work.
   - The correction line's benefit does not grow with drift (B5, B6).
   - The explicit what/where gate separates and buys nothing (E18).
   - The aliasing account is falsified with the sign inverted (D20).
   - Together they say the where is not improved by filtering it, gating it or pooling it. The one
     addition that pays, recursion, pays through optimisation, which ties back to condition 3.
4. It is modest where it must be: novelty is placed on the navigation regime, the rank bottleneck,
   the environment conditions, and the shared-vs-per-pair kernel isolation. Everything else is
   labelled replication.

### 1.3 The thesis (to open the report)

> On tasks where the observation map is redrawn at test, the transferable part of MapFormer is
> its accumulated, content-driven position phase, not the choice of rotary encoding. That phase
> pays only under three conditions, each established by intervention. The input must make a
> token's displacement fixed, the increment must be signed and well-conditioned so that it can
> cancel, and training must find the solution. Fully factorising where from what, as MapEM's shared
> kernel does, costs nothing measurable in what can be represented on the tasks tested. On a
> varying-offset recall task it detectably changes what training finds, and per-pair position
> freedom recovers part of that cost. Explicit correction, gating and pooling add no measurable
> accuracy on these tasks. All of this is transfer to a new instance of a known structure; transfer
> across a change of structure is not measured.

---

## 2. Claim hierarchy

Five main claims, one supporting reproduction section, and one cross-cutting open problem (the
length axis). Headline items are CITABLE. "Ours" and "prior art" are marked per claim.

### Claim 1: Path integration, not the rotary encoding, carries transfer to a redrawn map

**Headline evidence (CITABLE):**
- **A3** (paper task, n=8, one batch, 16 ep; `INDEX_BASELINE_PAPER_TASK_n8.md`, `BASELINE_TABLE.md`).
  - Fresh-map accuracy: MapPoPE-Flat 0.994 +/- 0.017, MapEM-os 0.987 +/- 0.009, MapWM-Flat 0.967 +/- 0.039.
  - Index arms: Plain-Flat 0.534 +/- 0.040, RoPE 0.530 +/- 0.043, PoPE-Flat 0.509 +/- 0.001.
  - Measured floor 0.506.
  - Position main effect +0.461. Encoding +0.003 (MDE 0.029, 5/8).
  - The encoding result is a POWERED NEGATIVE against an encoding effect comparable to position;
    it is not a null.
- **C12 Q1** (Match-Query 128^2, n=8, warmup+cosine; `LOOP_HEADROOM.md`).
  - Path-int 1-layer 0.456 vs index 0.108: +0.348 (MDE 0.215, 8/8).
  - Chance 0.0625.
- **C09** (Match-Query 64^2, n=5 vs n=5; `MATCH_QUERY_RESULTS.md`, `MATCH_QUERY_SCALE.md`).
  - 0.730 +/- 0.247 vs 0.154 +/- 0.018. No seed overlap (worst path-int 0.398, best index 0.178).
  - Context destruction: 0.918 -> 0.074 / 0.076.
  - Corrected never-moved floor 0.0893.
- **C18** (parity, n=8; `ALGORITHMIC_RESULTS.md`). Path-int - index +0.316 / +0.326 / +0.167 /
  +0.083 / +0.041 at L=16..256, 8/8 at every length. The copy control failed.
- **E24** (recency, n=8; `RECENCY_RESULTS.md`).
  - Path-int - index +0.750 (MDE 0.030, 8/8).
  - The index arms are at a capability limit: 2x budget moves RoPE 0.251 -> 0.242.
  - PRIOR-ART REPLICATION of CoPE.

**Supporting:**
- A6: the paper task passes context destruction. Every destroyed condition falls below the 0.506
  floor (n=3, EXPLORATORY by n, unambiguous in size).
- A7: index models exceed the floor only at recurrence interval 1-2, i.e. out-and-back retraces
  (EXPLORATORY).
- A12: family tree, path-int over index +0.205 (MDE 0.118, 3/3), n=3, EXPLORATORY.

**Ours vs prior art:**
- Ours: that path integration and not the encoding is the axis on MapFormer's own task, and the
  Match-Query separation.
- Replications:
  - That structural codes beat index codes at contextual counting is CoPE's claim.
  - That a signed cumsum is a parity register is Selective RoPE Sec. 4.2 / Grazzi.
- MapFormer's own paper already shows path integration solves its task. Our addition is the index
  and PoPE controls and the destruction gate.

**Strongest referee objection: the index baselines are undertrained.**
- The 2x2 is at 16 epochs LinearLR.
- Under the converged torus recipe (300 ep cosine lr 1e-3), RoPE reaches 0.799 +/- 0.018 at T=128
  (E11, `SIGN_ABLATION.md`).
- At that recipe, signed - RoPE loss-matched is -0.021 at T=128 (MDE 0.013), but +0.123 at T=512
  (MDE 0.075) and +0.195 at T=1024 (MDE 0.107), 0/12 negative.
- Does the data answer it? **Partly.**
  - On Match-Query at the better recipe the index arm is 0.108 (C12), because the blind query
    removes the retrace route.
  - On recency, doubling the budget moves nothing (E24).
  - On the torus at training length, a well-trained index model closes much of the raw gap, and
    after loss-matching all of it.
  - The report must say that the torus separation at a converged recipe is an OOD-length effect.
    The 16-epoch table must be presented as the paper's recipe, not as the converged one.
- Second objection: a plain transformer does path integrate through attention (compositional plain
  0.216 vs floor ~0.072, C02; MiniGrid commanded, index arms best, D6). That is answered by Claim 2,
  which is the boundary.

### Claim 2: The environment decides whether it pays, because a fixed per-token increment must be able to name a displacement

**Headline evidence (CITABLE):**
- **D11** (torus knob sweep n=8 and allocentric recoding; `KNOB_SWEEP_n8.md`,
  `ALLOCENTRIC_RECODING.md`).
  - Baseline +0.438. Rotate (turn/forward actions) +0.050 (0.558 +/- 0.026 vs 0.508 +/- 0.006).
  - Allocentric recording of the same dynamics: 0.996 +/- 0.005 vs 0.508 +/- 0.006, **+0.488**.
  - Answer-stream gates are identical to three decimals under both records.
  - Established by intervention on the action record alone.
- **D8** (allocentric MiniGrid-DoorKey-16x16, n=8; `MINIGRID_EM.md`).
  - Vanilla_r4 - RoPE +0.034 (MDE 0.014, 8/8) at T=1024.
  - At r=2 it is -0.010 (MDE 0.056), unmeasured.
- **D20** (MiniWorld, fixed grid 32, converged 400 ep; `ALIASING_CONTROLLED.md`).
  - n_obs=16: +0.178 (sd 0.094, MDE 0.118, n=5).
  - The aliasing hypothesis is falsified with the sign inverted. Pre-registered outcome B fired:
    the effect exceeded 0.150 at all three aliasing levels.
  - The less-aliased endpoint is +0.305 at n=3, 800 ep, with no results file (see Section 6).

**Supporting:**
- D6 (MiniGrid commanded actions, 8-cell factorial, n=8).
  - Index arms are best: PoPE-Hier 0.955 +/- 0.003 and PoPE-Flat 0.953 +/- 0.003 at T=1024.
  - MapWM-Flat 0.823 +/- 0.088.
  - Main effects: encoding +0.076, hierarchy +0.048, position -0.021.
  - The arm means are CITABLE levels. The main effects are DIRECTIONAL (no sd recorded).
- D12 (H=12 headings). Every path-int seed is above every index seed at every budget (weakest
  path-int 0.661, strongest index 0.555). The direction is CITABLE as existence; the magnitude is
  EXPLORATORY and non-monotone in budget.
- D19: at grid 8 both arms solve the task (-0.010); the earlier crossover is withdrawn.
- D21: map extent over visits per cell.
  - -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells.
  - EXPLORATORY: post-hoc pooling at n=3.
  - "Distinct cells visited" is falsified (CITABLE as a falsification).
- D5: frequency learning is not the position effect (+0.004 / -0.008, n=3, EXPLORATORY).

**Ours:** all of it. None of the prior-art corpus evaluates navigation.

**Strongest referee objections.**
1. "Allocentric recoding hands the model the answer."
   - Answer: dynamics are byte-identical, gates are identical, and the index arm stays at the floor
     under both records. The recoding supplies an integrable input, not the target.
   - The report must still say that it needs a known heading, and that Habitat's navmesh slides
     69-91% of forward moves (D13), which no experiment models.
2. "The MiniWorld threshold is post hoc."
   - Answer: yes. It stays EXPLORATORY and goes in the text as a boundary to test, not a finding.
   - The aliasing falsification does not depend on it.
3. "MiniGrid +0.034 is small."
   - Answer: it is small and detectable, on a benchmark where the commanded-action arms leave little
     headroom. The claim is the sign flip conditioned on r=4, not the size.

### Claim 3: The increment must cancel, and cancel cleanly: sign and rank (and the task sets their value)

**Headline evidence (CITABLE):**
- **E11 sign ablation** (torus, 6 arms x 12, one batch, identical parameters; `SIGN_ABLATION.md`).
  - Abs - Signed loss-matched: -0.215 (MDE 0.055, 12/12) at T=512 and -0.280 (MDE 0.061, 12/12)
    at T=1024.
  - Signed - RoPE +0.123 / +0.195. Abs - RoPE -0.092 / -0.085, unmeasured.
  - Every constrained arm has worse training loss on 12/12 seeds.
  - PRIOR-ART REPLICATION in the navigation regime, with a one-operation isolation.
- **E12**: opposition score, signed 0.106-0.130 vs monotone 1.849-1.981 (`SIGN_PROBE.md`).
- **E1 rank** (torus, n=8; `RANK_SWEEP.md`).
  - r=4 over r=2 +0.085 at T=1024 (t 3.57, 8/8); +0.038 at T=512.
  - A step: r=8/16/32 give +0.091 / +0.079 / +0.095.
  - T=1024 seed sd falls 0.064 -> 0.012. Cost: 384 parameters. **Ours.**
- **E2**: at r=2, opposition 0.4950 and |cos(N,E)| 0.7833; at r=4, 0.0922 and 0.1754. The actions
  still occupy a 2-plane (energy 1.0000). A description of learned codes; the causal route is not
  tested.
- **E26** (recency gate ablation, intervention): uniform_content 0.7826 vs uniform_all 0.1886,
  +0.594 at 8/8, magnitude-matched.

**Supporting:**
- E5: the paper's Fig. 4 C3 limitation (|cos| 0.779) falls to 0.174 at r=4 with no regulariser.
- E24/E25, the crossover.
  - Monotone - Signed on recency at T=2048: -0.004 raw (MDE 0.055), +0.026 loss-matched (MDE 0.027),
    unmeasured.
  - The unconstrained arm's alpha is 0.591 on the torus and 0.967 on recency (se 0.009). The
    constrained arms sit near 1.0 on both.
  - The crossover interaction (~+0.28) is cross-batch arithmetic: DIRECTIONAL.
- Scope limits, each main text:
  - E8: rank does not transfer measurably to MapPoPE (+0.019, 5/8, unmeasured).
  - E7: packing-geometry account refuted (r=2 posts its best score at D=5, 0.896; deficit +0.110 /
    +0.153 / +0.055 does not grow with D). POWERED as a falsification of the prediction's shape.
  - C13: on Match-Query, rank alone +0.005 (MDE 0.091); with the loop +0.154 (MDE 0.084, 8/8).
  - D8: the allocentric MiniGrid flip needs r=4.

**Ours vs prior art:**
- Ours: rank, and its conditioning diagnosis.
- Replication in a new regime: sign (Sarrof 2405.17394; Grazzi 2411.12537; Selective RoPE Sec. 4.2).
- CoPE's: counting. Our addition is the finding that an unconstrained MapFormer learns a CoPE-like
  content gate, by intervention.
- The two-slot taxonomy and content-dependent rotation are GRAPE / Puranik / Mamba-3 and are
  background only.

**Strongest referee objection: every accuracy effect here is at OOD length.**
- At T=128 the signed arm is 1.000 +/- 0.000 and r=4 is 1.000. Rank's +0.085 is at 8x training
  length.
- "Helps at OOD length" is also what the InEKF, forget gate, PoPE and EM show.
- Does the data answer it? **Partly.**
  - Sign has a training-length effect in the loss (12/12).
  - Opposition and alpha give a mechanism-level description that tracks both sign and rank
    (r(opposition, alpha) = +0.9995, E13).
  - The route from a growing accumulator to OOD failure is not established, and the one imported
    account (critical dimension) is refuted (E13 P2).
  - The report must carry this as the open length-axis problem (Section 3, Sec. 9).

### Claim 4: Factorising where from what with a shared kernel changes what training finds, not what can be represented

**Headline evidence (CITABLE, all on varying-k recency unless stated):**
- **F1** (`RECENCY_EM_RESULTS.md`).
  - EM_P0 - WM -0.375 (sd 0.156, MDE 0.154, 0/8) at T=1024.
  - Epochs to loss < 0.5: WM 8/8, EM_P0 0/8.
- **F2**: MapEM's `A_X (*) A_P` equals the bilinear form on content (x) position (TEM's `g (x) x`),
  exact algebra. MapWM's kernel has per-pair amplitudes and phases, so MapWM is not additive. The
  kernel-level construction selects the answer 1423/1423.
- **F9** (`WARM_RESULTS.md`): rewind installed and frozen, 1.000 +/- 0.000 on 8/8 at T=1024 and
  T=2048; +0.400 over scratch (MDE 0.125).
- **F10/F11**: installed at 8x scale, trainable, content -> Delta leak closed: 1.000 +/- 0.000,
  8/8 >= 0.95, vs 4/8 with the leak open. The paired +0.059 (MDE 0.063) is unmeasured at ceiling;
  seed count is the registered readout.
- **F12**: of failed cells, 0.000 carry a rewind in every arm. 93-97% of solved large-k cells go
  through the query token's own wrapped rewind, with the A_X sign tracking peak vs trough (1.000 /
  0.000). This is description.
- **F14**: with one shared k=64, EM_P0 reaches 0.985 +/- 0.043 (7/8), in 54 epochs vs WM's 99.
- **F16**: at a fixed budget, m4 - m64 +0.665 (MDE 0.213, 8/8). At matched queries per token,
  +0.045 (MDE 0.166), unmeasured.
- **F19** (n=48; `PAIRSPLIT_RESULTS.md`).
  - Per-pair origins total +0.215 (MDE 0.068, 42/48) = pathway +0.124 (MDE 0.071) + freedom +0.091
    (MDE 0.066).
  - Freedom replicates on fresh seeds (+0.089, MDE 0.069).
- **F18**: the mechanism, by intervention. Solved cells via per-token rewind: EM_P0 0.964,
  EMPairConst 0.948, EMPair 0.189. Phase spread 0.000 / 0.000 / 1.448.
- **F8**: phase freedom in k0 +0.146 (MDE 0.086, 22/24), fresh seeds +0.113 (MDE 0.111). Mechanism
  unidentified.

**Supporting:**
- F25: available rewind quality 0.992 (failed) vs 0.995 (solved) tokens, achieved 0.359 vs 0.544.
  Description, 8 checkpoints.
- F21/F23: ties on map tasks once EM's init is fixed.
  - MiniGrid EM_P0_r4 - Vanilla_r4 +0.0035 (6/8, no MDE stated).
  - Vocab n_obs=256 trimmed +0.0000 (MDE 0.169).
  - DIRECTIONAL.
- A8 / F6 / F7: the paper's separate-q0/k0 conjecture is task-dependent.
  - Worse on the torus: sep - P0 -0.154 (MDE 0.130, 0/8, n=8).
  - Worse on Match-Query: -0.358, n=3.
  - Better on recency: +0.128 pooled n=24, fresh seeds +0.073 unmeasured.
- F20 (paper task, extended length, 50 ep cosine).
  - EM - WM floor-normalised +0.186 (MDE 0.185, 8/8) at l=1024 and +0.287 (MDE 0.194, 8/8) at
    l=2048. r(loss, acc) = -0.461, so not a loss gap.
  - **The pre-registered convergence gate failed, so P1 is not read.** EXPLORATORY in the claim
    hierarchy; reported in the text with the gate stated.
- F27: [11]'s capacity result does not transfer to MapEM (no separate memory network). [11]'s
  N-back corresponds to fixed k, where EM learns faster (F14).

**Ours:** the shared-vs-per-pair isolation, the existence / holdability / search ladder, the
per-token search description, and the per-pair decomposition. [11] and TEM are the frame, prior art.

**Strongest referee objections.**
1. "One task, and every contrast is a fit contrast (r = -0.936 to -0.986)."
   - Partly answered. The "search, not representation" reading does not rest on loss-matching.
     It rests on existence (F9), holdability (F11) and availability (F25), each shown directly.
   - The one-task limit stands.
   - The map side (F20-F23) shows no detectable EM cost, so the claim is scoped to varying-offset
     retrieval.
2. "Holdability was shown at a pinned point in weight space."
   - Not answered. It is stated as a scope limit.
3. "WM at the 4x budget was never run, so -0.375 may be convergence rate."
   - Not answered. The report says -0.375 is the fair shared-budget comparison and makes no claim
     about asymptotic ability.

### Claim 5: Added machinery does not add transfer accuracy; only recursion pays, and it pays through optimisation where the base model is unreliable

**Headline evidence:**
- **Recursion helps where there is headroom (CITABLE): C12** (Match-Query 128^2, n=8).
  - Loop on path-int +0.414 (MDE 0.277, 7/8); loop on index +0.099 (MDE 0.045, 8/8).
  - Interaction +0.315 (MDE 0.281).
  - Loop vs 3 real layers +0.099 (MDE 0.252), unmeasured. The loop matches depth, it does not beat it.
- **C13**: r=4 + loop x4 is 0.986 +/- 0.020 (min 0.941), interaction +0.149 (MDE 0.112).
- **Recursion's torus win is convergence (CITABLE as a decomposition): C24/B2** (n=12).
  - Looped - Vanilla at T=128: raw +0.052 (MDE 0.048, 12/12), loss-matched +0.006 (MDE 0.017).
- **C19** (parity, n=16).
  - Loop - L1: index +0.056 (MDE 0.014), path-int +0.070 (MDE 0.056).
  - Difference +0.014, not detectable: additive on parity, super-additive only on Match-Query.
- **State correction (the project's original extension) is not inference.** POWERED NEGATIVE,
  **B5/B6** (Match-Query with stochastic transitions, where drift exists and observations carry
  true position).
  - Level15 - Vanilla at p=0.10: +0.038 (MDE 0.068) in take 1, +0.005 (MDE 0.065) in take 2.
  - The pre-registered "grows with drift" prediction was refuted twice (change +0.003, -0.141).
  - Noisy-task floor ~0.15 (B7).
- **B1** (n=5): Level15 - Vanilla loss-matched +0.062 (t 3.08) at T=512 and +0.124 (t 3.83) at
  T=1024. CITABLE and OOD-length only. No named component is load-bearing; every component
  contrast is unmeasured.
- **Explicit what/where gate: E18.**
  - Separation 4.16x, 8/8 (CITABLE).
  - Torus T=512 gain +0.004 (MDE 0.008): a POWERED NEGATIVE against any gain the size of rank's
    +0.038.
  - Recency -0.039 raw (MDE 0.079), unmeasured.

**Supporting:**
- B2 (n=12): the filter main effect at T=1024 is raw +0.129 (MDE 0.103), loss-matched +0.083
  (MDE 0.083, borderline). The filter-plus-loop combination is below the filter alone at OOD
  (0.830 vs 0.878).
- E15: Selective RoPE's generator is not better than MapFormer's.
  - Parity -0.009, unmeasured.
  - Torus T=512 +0.031 (MDE 0.030), T=1024 t 1.36.
  - Per-knob attribution is confounded (every single-knob arm also deletes omega).
- E19/E20: the forget gate at r=2 is +0.081 (MDE 0.080, 7/8), vanishes at r=4 (-0.002), needs a
  live lambda (frozen -0.016, MDE 0.118). Mechanism unidentified.
- C05: hierarchy on compositional transfer +0.136 (MDE 0.173, 7/8). DIRECTIONAL.
- C04: the recipe effect on the same task, +0.160 (MDE 0.140), is larger than any architectural
  effect measured there.
- C21: hierarchy on parity at L=512, +0.060 (MDE 0.075, 12/12) and +0.071 (MDE 0.081, 11/12).
  DIRECTIONAL by the MDE rule, sign-consistent.
- C22/C26: hierarchy is an efficiency property.
  - -22.9% time and -19.9% memory at L=2048; +12.2% time at L=16.
  - enwik8 +0.0032 bpc worse at 1.23x throughput, n=1.

**Ours vs prior art:**
- Ours: the correction, gate and hierarchy measurements.
- "Recursion substitutes for depth" is Mixture-of-Recursions' premise. Our addition is the
  interaction with path integration on one task, and the convergence decomposition.

**Strongest referee objection: the correction could still help on a regime you did not test (true
landmarks, very long T).**
- Answer: the landmark evidence is either void (April lm200) or tied by a filter-free capacity
  control (B8, +0.010, t=0.79, n=3). The claim is scoped to the tasks tested.
- Second objection: "the loop helps via optimisation" is an interpretation.
  - Answer: at training length it is a decomposition (loss-matched +0.006). On Match-Query it is
    stated as associated, since the loop's contribution is mostly to the floor (8/8 seeds >= 0.77).

### Supporting section: reproduction of MapFormer (not a main claim)

- A4/A5: our WM and EM are consistent with paper **v1** Table 2 within about one sd, and below v4.
  - At n=8, 50 ep cosine: WM IID 0.968 +/- 0.051, EM 0.985 +/- 0.023.
  - v4 lists 1.00.
  - WM's shortfall does not move with budget (0.969 at 16 ep, 0.968 at 50 ep).
- E5/E6: Fig. 4 claims C1-C3 reproduce. C4 does not, on either backbone (MapWM 0.57, MapEM 0.90).
- A14 (CITABLE, engineering measurement): the parallel-scan claim holds. Forward+backward growth
  over L=128 -> 2048 is 2.5x (Vanilla), 14.4x (MapEM-NC), 120.4x (TEMFaithful).
- A12/A13 (n=3, EXPLORATORY): non-commutativity buys +0.013 (MDE 0.008) on a family tree with
  measured non-commutativity 1.000. Path integration buys +0.205 there.
- The omega sign typo in paper eq. 17/18 and our fix belong in methods.

---

## 3. Section outline of the report

Target: about 28-32 pages plus appendices. Each section gives its one-sentence point, the
experiments it uses, and a rough length.

**Abstract** (Section 7 below).

**1. Introduction (2 pp).**
- Point: a learned where kept separate from the what should transfer. We measure when MapFormer's
  where does, on tasks where every evaluation redraws the map.
- Content:
  - Research goal (TEM factorisation) and the transfer-measurement framing.
  - MapFormer in one paragraph.
  - Prior art, positioned honestly: GRAPE / Puranik / Vetcha (two-slot taxonomy); Mamba-3 and
    Selective RoPE (content-dependent rotation); Sarrof / Grazzi (sign); CoPE (contextual counting);
    TEM and [11] (EM/WM); MoR (recursion); LieRE (nearest rank neighbour); HGRN / HGRN2 and E29
    (language negatives).
  - Contribution list with an ours / replication column.
- Figure 1: schematic of the accumulator, the kernel, and the three conditions.

**2. Setup and methods (3.5 pp).**
- 2.1 Models (0.75 pp).
  - MapWM, MapEM (single p0 vs separate q0/k0), and the kernel algebra (F2, F4).
  - Index baselines: RoPE (architecture-matched), PlainFlat, PoPE-Flat, MapPoPE.
  - The sign knob, the rank knob, loops, Level 1.5.
  - The omega sign-typo fix. Canonical RoPE schedule check (D23, powered negative).
- 2.2 Tasks (1 pp). Table 1: task, what it tests, chance, measured floor, train/eval lengths, gate
  status.
  - Paper task and its OOD protocol (floors from `PAPER_TASK_FLOORS.md`).
  - Match-Query 64^2 and 128^2 (C09, C10). Parity (C18). Recency (E23).
  - Knob torus (D10, D11). MiniGrid-DK-16x16 (D6, D8). MiniWorld OneRoom (D19, D20).
  - Compositional motif and family tree are appendix only.
- 2.3 Validity gates (0.5 pp).
  - Action/answer-stream n-gram gates run before training.
  - Context destruction with both shuffle and on-manifold resample (A6, B7, C01, C09); they order
    differently by task.
  - Measured floors. Manipulation checks.
  - Appendix: the two audits that voided task families (C32, C33).
- 2.4 Statistics (0.75 pp).
  - Paired MDE = 2.8 sd/sqrt(n); "unmeasured" below it.
  - One batch per comparison; bitwise determinism checks are not replication.
  - Fresh-seed replication split. The first eight seeds overestimated effects in F7, F8 and F19.
  - Rule 9 r(loss, acc) printed per batch. Loss-matched residuals are reported beside raw results,
    with the mediator caveat (F7 / audit #7).
  - Pre-registration: verdicts reported as registered even when a gate fails (F20) or a
    discriminator was mis-designed (E11 H1).
  - Readouts are designed to be invariant to model symmetries (periodicity, sign gauge).
- 2.5 Recipes (0.5 pp).
  - Paper recipe: 16 ep LinearLR. Torus recipe: 300 ep warmup+cosine lr 1e-3.
  - C25: lr 1e-3 cuts Vanilla T=512 sd 0.096 -> 0.028, but does not transfer to Match-Query or the
    compositional task.
  - C04: the recipe effect exceeds architecture effects on compositional.
  - Which table uses which recipe.

**3. Reproducing MapFormer (1.5 pp).**
- Point: the reimplementation matches v1, sits below v4, and the parallel-scan claim holds. Two
  paper conjectures do not survive on map tasks.
- Uses: A4/A5 (Table 2: paper v1, v4, ours), E5/E6 (Fig. 4 table), A14 (timing table), A8 (brief,
  with pointer to Sec. 7), A12 (one sentence).

**4. Claim 1: the accumulated phase, not the encoding, carries transfer (2.5 pp).**
- Point: on four tasks an index code sits at or near the floor on a redrawn map while a
  path-integrated code solves it. The encoding contributes little by comparison.
- 4.1 The paper task 2x2 (A3; Table 3), with A6 destruction and A7 retrace residual (one small
  table or inline).
- 4.2 Blind continuation (C09 n=5, C12 Q1 n=8).
- 4.3 Beyond navigation: parity (C18) and recency (E24 index arms; CoPE replication).
- 4.4 The recipe objection, stated and answered with E11 Signed - RoPE at the converged recipe.
- Figure 2: accuracy vs length for signed / monotone / index (E11) or per-offset recency (E24).

**5. Claim 2: the environment must make displacement a function of the token (3 pp).**
- Point: under turn/forward actions the advantage vanishes and recording realised displacement
  restores it. On MiniWorld the effect needs a large map and does not scale with aliasing.
- 5.1 The knob sweep and allocentric recoding (D11; D10 in appendix). Table 4.
- 5.2 Heading resolution (D12, existence only; B23 noise at undertrained budget in appendix).
- 5.3 An external benchmark (D6 levels; D8 +0.034 with r=4; D7 appendix).
- 5.4 MiniWorld: convergence first (D19), aliasing falsified (D20), map extent as an open threshold
  (D21, EXPLORATORY). Figure 3: effect vs occupied cells, with n and budget annotated.
- 5.5 What is not modelled: Habitat slides (D13), one sentence.

**6. Claim 3: sign, rank, and the task (4 pp).**
- Point: the increment must be able to cancel. Monotone increments cannot, a rank-2 bottleneck
  cancels badly, and on a counting task the same constraint is harmless.
- 6.1 Sign (E11, E12). Table 5 plus opposition. Labelled replication.
- 6.2 Rank (E1, E2, E5). Table 6. Scope: E8, C13, D8. Negative: E7.
- 6.3 The task sets the value (E24, E25, E26), including the rewind scope correction (F2: recency
  does not need a clock).
- 6.4 One description, two severities (E13 P3 alpha/opposition). Refuted import (E13 P2).
- Figure 4: opposition vs alpha across arms.

**7. Claim 4: a shared kernel is representable but harder to find (5 pp).**
- Point: MapEM's shared position kernel ties MapWM on map tasks. On varying-offset recall it
  trails by 0.375, and the gap is search.
- 7.1 Two ways to compose where and what (F2, F4; Hadamard = TEM conjunction).
- 7.2 Map tasks.
  - F21/F23 ties after init fixes.
  - F6/A8 separate-origin task dependence.
  - F20 paper-task length result with the failed gate stated.
- 7.3 The recency gap (F1).
- 7.4 Existence and holdability (F9, F10, F11). Table 7 as a ladder.
- 7.5 Search (F12, F14, F16, F25; F13 in appendix). Figure 5: per-token rewind anatomy and the
  m-by-budget grid.
- 7.6 Per-pair origins (F17, F18, F19). Table 8: the n=48 decomposition plus mechanism readouts.
- 7.7 Phase freedom (F8; mechanism unidentified).
- 7.8 Relation to [11] (F27, one paragraph).

**8. Claim 5: what added machinery buys (3 pp).**
- Point: of correction, gating, alternative generators, forget gates, hierarchy and recursion, only
  recursion measurably adds accuracy, and it does so where the base model is unreliable.
- 8.1 Recursion (C12, C13, C19, C24; C25 converged fractions). Table 9.
- 8.2 State correction (B1, B2, B5, B6, B7). The powered negative leads.
- 8.3 Gates and generators (E18 lead; E15, E19/E20 brief).
- 8.4 Hierarchy (C05, C21, C22; C26 brief). Directional on transfer, an efficiency property on
  compute.

**9. The length axis (1.5 pp).**
- Point: rank, sign, correction, the forget gate, PoPE and EM all show their effect at OOD length,
  and no account covers them.
- Uses: E13 (alpha tracks sign and rank only), E14 (forget / PoPE / Level15 leave alpha
  unmeasured; InEKF does not bound the accumulator), E10 (PoPE: octave prediction refuted, length
  trend 3/3), F26 (collisions: a switch, 53% of EM's own drop, one arm).
- Stated as the main open problem, not a claim.

**10. Limitations (1.5 pp).** See Section 5 of this document for the list:
- transfer within a structure only;
- no language result;
- one varying-offset task;
- OOD-length concentration;
- recipe dependence of index baselines;
- n=3 boundaries;
- pinned-weight holdability;
- Habitat realism;
- Match-Query cross-batch non-reproducibility;
- borrowed benchmarks that do not discriminate (E27, E28).

**11. Conclusion (0.5 pp).**

**Appendices** (purpose per item in Section 4):
- A: gates and floors per task.
- B: April baselines and TEM.
- C: family tree and non-commutativity.
- D: compositional task and hierarchy details.
- E: MiniGrid and MiniWorld supplementary factorials.
- F: rank and generator probes.
- G: EM kernel supplementary (N5, DOF, SPREAD, S2).
- H: correction supplementary.
- I: loops supplementary.
- J: language pointers.
- K: in-flight MONOTONE.

---

## 4. Placement of every surviving experiment

Legend: **M** main text, **App** appendix (purpose), **Cut** (reason). Duplicates across
inventories are marked "= X" and placed once.

### Inventory A

| ID | placement | purpose / reason |
|---|---|---|
| A1 | App B | n=3 first pass; source of the paper-task separate-q0/k0 row (0.898) |
| A2 | Cut | exploratory n=3, superseded by A4/A5 |
| A3 | M Sec 4.1 | headline 2x2 |
| A4 | M Sec 3 | Table 2 at n=8, paper recipe |
| A5 | M Sec 3, 7.2 | = F20; converged-recipe rerun, gate failed |
| A6 | M Sec 4.1 | paper-task context destruction |
| A7 | M Sec 4.1 (inline) | why index exceeds floor |
| A8 | M Sec 7.2 (brief), App G | separate q0/k0 on map tasks, n=3; = F24 |
| A9 | App G | kernel-geometry falsification; sign gauge |
| A10 | Cut | superseded by A11 (old recipe, n=3) |
| A11 | M Sec 7.2 | = F21; EM capacity claim unsupported |
| A12 | M Sec 3 (one sentence), App C | non-commutativity small; n=3 |
| A13 | App C | depth-7 check; n=3, no absolute accuracies |
| A14 | M Sec 3 | timing |
| A15 | App B | LSTM / CoPE / MambaLike context; n=3 ungated |
| A16 | App B | TEM-t baselines; n=3 |
| A17 | App B | TEMFaithful clean/noise; n=3 cross-batch |
| A18 | App B | CSCG stitching port; n=3 |
| A19 | Cut | superseded OOD protocol; paper's rescaling unverified (= B19, D24) |
| A20 | Cut | ungated, no floor, n=3 |
| A21 | App D | gate licensing the compositional task |

### Inventory B

| ID | placement | purpose / reason |
|---|---|---|
| B1 | M Sec 8.2 | Level 1.5 decomposition n=5 |
| B2 | M Sec 8.1-8.2 | = C24; filter x loop n=12 |
| B3 | App H | refinement on depth axis dead; loop-vs-Kalman needs replication (n=3) |
| B4 | Cut | premise-invalid test, unmeasured |
| B5 | M Sec 8.2 | powered negative, take 1 |
| B6 | M Sec 8.2 | powered negative, take 2 |
| B7 | M Sec 8.2 (inline), App A | floor ~0.15 for noisy Match-Query |
| B8 | App H | lm200 destruction gate plus capacity tie; withdraws Kalman reading of landmarks |
| B9 | Cut | n=3, several arms unconverged, interpretation withdrawn |
| B10 | Methods footnote | licenses April clean/noise rows (determinism) |
| B11 | Cut | LinearLR recipe, superseded by B1/B2 |
| B12 | App H | rule-5 budget example; ceiling |
| B13 | App H | correction without measurements; n=3 |
| B14 | App H | = C07; likelihood not accuracy; n=3 |
| B15 | Cut | n=3 unmeasured, duplicate of A12 row |
| B16 | Cut | single seed; mechanism diagnosed only on void lm200 |
| B17 | Cut | n=3, built for void lm200 claims |
| B18 | Cut | no detectable effect; cross-batch |
| B19 | Cut | = A19 / D24 |
| B20 | Cut | n=1 |
| B21 | Cut | n=1 hex probes; Sorscher conditions absent, so no test |
| B22 | Cut | n=1 continuous nav |
| B23 | App E | = part of D12; actuation noise at undertrained budget |
| B24 | App F | drift probe: 11x ratio, does not explain accuracy |
| B25 | Cut | diagnostic on a voided task |
| B26 | Cut | landmark torus half of a voided family |

### Inventory C

| ID | placement | purpose / reason |
|---|---|---|
| C01 | App D | compositional task validity |
| C02 | App D | old-recipe multiseed; context for C04/C05 |
| C03 | App D | H3 falsified (segmentation, frame reset); n=3 |
| C04 | M Sec 2.5, 8.4 | recipe exceeds architecture |
| C05 | M Sec 8.4 | hierarchy directional |
| C06 | App D | = D25; dissociation sweep, n=3 |
| C07 | App H | = B14 |
| C08 | Cut | exploratory hit rates; gate above chance; cross-batch |
| C09 | M Sec 4.2 | Match-Query separation |
| C10 | App A | scale, aliasing boundary, long blind queries; n=3 |
| C11 | Cut | undertrained; single diagnostic run |
| C12 | M Sec 4.2, 8.1 | loop x path integration |
| C13 | M Sec 6.2, 8.1 | rank x loop |
| C14 | App I | MoR upper bound; powered negative |
| C15 | App I | index loop buys horizon; n=3 |
| C16 | Cut | budget-limited; superseded by C15 |
| C17 | Cut | stale anchors |
| C18 | M Sec 4.3, 8.1 | parity |
| C19 | M Sec 8.1 | loop vs depth frontier |
| C20 | App D | with C21 (L=16 cannot exercise pooling) |
| C21 | M Sec 8.4 (brief) | hierarchy at L=512 |
| C22 | M Sec 8.4 | compute benchmark |
| C23 | App I | sampled loop count; directional |
| C24 | = B2 | |
| C25 | M Sec 2.5 | recipe power |
| C26 | App J | enwik8 hierarchy, n=1; efficiency measured |
| C27 | Cut | predates deterministic val; n=1; superseded |
| C28 | App J | enwik8 PoPE x path integration; underpowered |
| C29 | Cut | design flaw, unconverged, n<=3 |
| C30 | Cut | training-length confound |
| C31 | Cut | off-story (event vs place); n=3 |
| C32 | App A | audit that voids hier-goal (methods) |
| C33 | App A | planner-task audit (methods) |
| C34 | Cut | n=1 ungated |

### Inventory D

| ID | placement | purpose / reason |
|---|---|---|
| D1 | Cut | n=1, no floor |
| D2 | Cut | n=1 |
| D3 | Cut | superseded by D6 |
| D4 | Cut | superseded by D6 |
| D5 | App E | frequency-learning confound check; n=3 |
| D6 | M Sec 5.3 | MiniGrid 8-cell n=8 |
| D7 | App E | allocentric MiniGrid flip; n=3 cross-batch |
| D8 | M Sec 5.3, 6.2 | = F22; allocentric + r=4 beats index |
| D9 | M Sec 7.2 | = F23; EM init fix tie |
| D10 | App E | four n=3-only knobs |
| D11 | M Sec 5.1 | knob n=8, allocentric |
| D12 | M Sec 5.2 | 12 headings, existence |
| D13 | M Sec 5.5 (one sentence), App E | Habitat measurements |
| D14 | App E | fixed-map MiniWorld; effects below floor |
| D15 | Cut | unconverged; "liability" reading withdrawn |
| D16 | Cut | unconverged recipe; superseded by D19-D21 |
| D17 | App A | measured run-to-run floor (with unconverged caveat) |
| D18 | Cut | unmeasured, unconverged; NoPE n=1 |
| D19 | M Sec 5.4 | converged grid 32 / grid 8 |
| D20 | M Sec 5.4 | aliasing falsified |
| D21 | M Sec 5.4 | map-extent threshold, EXPLORATORY |
| D22 | Cut | unconverged, unmeasured |
| D23 | App A | index RoPE schedule check; powered negative |
| D24 | Cut | = A19 |
| D25 | = C06 | |
| D26 | App F | gate record for E7 |

### Inventory E

| ID | placement | purpose / reason |
|---|---|---|
| E1 | M Sec 6.2 | rank sweep |
| E2 | M Sec 6.2 | action geometry |
| E3 | Cut | descriptive; confounded by omega deletion |
| E4 | App F | truncation is not a sufficiency test |
| E5 | M Sec 3, 6.2 | Fig. 4 on MapWM |
| E6 | M Sec 3 (brief), App F | Fig. 4 on MapEM; C4 discrepancy |
| E7 | M Sec 6.2 | D x r geometry refuted |
| E8 | M Sec 6.2 | rank does not transfer to MapPoPE |
| E9 | Cut | cross-batch compilation; unconverged cells |
| E10 | M Sec 9 (brief), App F | PoPE length vs octaves |
| E11 | M Sec 4.4, 6.1 | sign ablation |
| E12 | M Sec 6.1 | opposition probe |
| E13 | M Sec 6.4, 9 | alpha / opposition; refuted import |
| E14 | M Sec 9 | accumulator scope; failed positive control |
| E15 | M Sec 8.3 (brief), App F | SRoPE generator; confound |
| E16 | App F | gate-as-suppressor falsified |
| E17 | Cut | kernels at random baseline; off-story |
| E18 | M Sec 8.3 | explicit gate |
| E19 | M Sec 8.3 | forget gate |
| E20 | M Sec 8.3 (inline), App F | frozen-lambda control |
| E21 | App F | transient-aid refuted |
| E22 | Cut (Limitations mention) | no result: batch deleted, must re-run |
| E23 | M Sec 2.2 | recency task and gates |
| E24 | M Sec 4.3, 6.3 | recency batch |
| E25 | M Sec 6.3 | alpha per task |
| E26 | M Sec 6.3 | gate ablation by intervention |
| E27 | App J / Limitations | Flip-Flop does not discriminate |
| E28 | App J | MQAR feasibility floor |
| E29 | M Sec 1, 10; App J | language prior art; enwik8 pointer |
| E30 | App K | MONOTONE, in flight |

### Inventory F

| ID | placement | purpose / reason |
|---|---|---|
| F1 | M Sec 7.3 | recency EM vs WM |
| F2 | M Sec 7.1, 6.3 | kernel algebra, rewind construction |
| F3 | Cut | correlational, null-shaped, falsifier fired |
| F4 | M Sec 7.1 | phase spread |
| F5 | App G | |rho| scoped to frozen low-amplitude kernel; gauge |
| F6 | M Sec 7.2 (torus sep - P0 only), App G | recency half superseded |
| F7 | App G | n=24 extension; magnitude unmeasured |
| F8 | M Sec 7.7 | phase freedom |
| F9 | M Sec 7.4 | existence |
| F10 | M Sec 7.4 | install scale |
| F11 | M Sec 7.4 | leak closed |
| F12 | M Sec 7.5 | per-token wrapped anatomy |
| F13 | App G | init gradient, ruggedness snapshots |
| F14 | M Sec 7.5 | fixed k, curriculum |
| F15 | App G | superseded in size by F16 |
| F16 | M Sec 7.5 | queries per token |
| F17 | M Sec 7.6 | per-pair origins, mechanism readouts |
| F18 | M Sec 7.6 | capacity control |
| F19 | M Sec 7.6 | n=48 decomposition |
| F20 | = A5 | |
| F21 | = A11 | |
| F22 | = D8 | |
| F23 | = D9 | |
| F24 | = A8 | |
| F25 | M Sec 7.5 | availability |
| F26 | M Sec 9 | collisions |
| F27 | M Sec 7.8 | [11] comparison |

---

## 5. Negative results that earn main-text space

Ranked by how much of the story they carry.

1. **The correction does not scale with drift** (B5, B6; POWERED NEGATIVE at MDE ~0.065-0.068).
   - This is the project's original extension, tested where its premise holds, with the prediction
     refuted twice.
   - It carries Claim 5 and tells the reader the where is not improved by filtering it. With B1,
     it also leaves "stabilisation at OOD length" as the only surviving description.
2. **Aliasing does not drive the position effect, and the sign is inverted** (D20). It prevents the
   most natural reading of the environment line, and it forces the map-extent question.
3. **An explicit what/where gate separates 4.16x and buys nothing** (E18). This directly answers
   the research goal's obvious engineering move: build the separation in. It corroborates that
   MapFormer's bottleneck already separates.
4. **Monotone increments beat an index code nowhere** (E11 Abs - RoPE unmeasured). This is what
   makes the sign result a condition rather than a tweak.
5. **The rank deficit is not packing geometry** (E7). It stops the rank result being over-read as
   an account of the paper's 5D Table 6.
6. **Rank does not transfer measurably to MapPoPE** (E8, unmeasured). This scope limit is needed
   so "use r=4" is not overgeneralised. Report as unmeasured, not negative.
7. **Hierarchy on transfer is unpowered and costs retrieval** (C05 directional; retrieval losses
   in C29 are exploratory, so appendix only). Main text says "directional, n=8, unmeasured" plus
   the efficiency measurement.
8. **The critical-dimension import is refuted** (E13 P2), and **the InEKF does not bound the
   accumulator** (E14 P1). These belong in Sec. 9: they are why the length axis is open.
9. **Separate q0/k0 is not a general improvement** (A8, F6, F7): worse on map tasks, directionally
   better on recency. Main text in Sec. 7.2 as task dependence.

Negatives that go to appendices:
- MoR routing (C14).
- Motif segmentation and frame reset (C03).
- Selective RoPE's gate-as-suppressor (E16).
- The Flip-Flop and MQAR non-discrimination (E27, E28; one sentence in Limitations).
- Magnitude freedom (F7/F8 M3).
- Transient-aid forget gate (E21).
- Refinement on the depth axis (B3).

---

## 6. Tensions, gaps, and what must be resolved first

### 6.1 Tensions across lines, and how the report reconciles them

1. **The index baseline is at the floor on the paper recipe but not on the converged recipe.**
   - RoPE 0.530 (A3, 16 ep) vs 0.799 at T=128 (E11, 300 ep cosine lr 1e-3), still with final loss
     0.7844 against the signed arm's 0.0002.
   - Reconciliation: the torus separation at a converged recipe is an OOD-length effect (+0.123 /
     +0.195 loss-matched). Match-Query and recency keep the index arm near floor at the better
     recipe (C12, E24).
   - Present the 16-epoch table as the paper's recipe, and E11 as the converged check.
2. **EM ties WM on map tasks, degrades more slowly with length on the paper task, and trails badly
   on varying-k recency.**
   - The ties are at training length after init fixes (F21, F23). The length advantage is monotone
     in l and qualified by a failed gate (F20). The recency gap is a search result (F1-F19).
   - Reconciliation: these are different axes (length vs per-query offset). The report must not
     say "a shared kernel is better when the offset is fixed", which the sources explicitly demote.
3. **Rank.** r=4 helps MapWM on the torus (E1) and is needed for the allocentric MiniGrid flip (D8).
   It is unmeasured on MapPoPE (E8), does nothing alone on Match-Query and a lot with the loop (C13),
   and one Vanilla_r4 seed collapses on vocab 256 (A11).
   - Reconciliation: rank is a conditioning repair that pays where the r=2 basis binds. The
     MapPoPE reason (64 vs 32 channels) is untested and must be called untested.
4. **Loops.** Super-additive on Match-Query (C12), additive on parity (C19), convergence-only on
   the torus (C24).
   - Reconciliation: the interaction is task-specific. The common thread is reliability of
     optimisation, stated as associated.
5. **The encoding axis.**
   - Negligible on the paper task (A3, +0.003).
   - MapPoPE beats MapWM at l=2048 by +0.116 (A4, 16 ep).
   - Encoding is the largest MiniGrid main effect under commanded actions (+0.076, D6).
   - Reconciliation: the claim is "position, not encoding, carries transfer on map tasks with an
     integrable input". PoPE's length benefit joins the length axis (Sec. 9). Its MiniGrid benefit
     is in the regime where path integration is misspecified.
6. **The OOD-length signature is shared** by rank, sign, correction, forget gate, PoPE and EM.
   - Only sign and rank have a common description (E13).
   - The report must present this as an open problem, not let each section claim "helps at length"
     as its own mechanism.
7. **The crossover's recency half versus the rewind construction.**
   - "Monotone is harmless on recency" (unmeasured) and "recency does not need a clock" (F2, F9) are
     consistent, but the text must use the scoped wording.
   - MONOTONE Experiment 1 tests exactly this for EM (see 6.4).

### 6.2 Source disagreements to resolve before numbers go in the report

Listed in order of risk.

1. **A5 vs F20 status.**
   - Inventory A labels the paper-task EM - WM length result CITABLE; inventory F labels it
     EXPLORATORY.
   - Primary file (`PAPERTASK_RESULTS.md`, checked): the gate FAILED, P1 is not read, and the
     numbers are "reported but not interpreted as a verdict".
   - **Decision: EXPLORATORY in the claim hierarchy.** Report the numbers with the gate stated.
   - The user may wish to re-register the gate and re-read. Open question 7 in
     `THEORY_NARRATIVE.md`.
2. **MiniWorld +0.305 (n_obs=256, 800 ep) and grid 16 +0.015 exist only as raw JSON**
   (`runs/alias_follow/n256_800/`, `runs/alias_follow/g16/`), with no generated results file,
   no sd and no MDE (D20, D21).
   - These are two of the three threshold points and the less-aliased endpoint of the falsification.
   - **A results file must be generated by the existing aggregator before either number is quoted.**
   - The falsification's direction survives on the 400-ep table (+0.178 / +0.310 / +0.374),
     although those runs are not all converged.
3. **Match-Query cross-batch reproducibility.**
   - `LEVEL15_MEETS_GATED_matchq.md` reproduces Vanilla to four decimals.
   - `REFINE_RESULTS.md` and the CLAUDE.md 2026-09-05/06 note say Match-Query does not reproduce
     across batches.
   - The same arm reads 0.456 (C12) and 0.416 (C13).
   - **Decision:** cite only within-batch contrasts on Match-Query, and state the non-reproducibility
     in Methods. Whether it depends on recipe (LinearLR vs fast-attn cosine) is unresolved.
4. **RESULTS_INDEX lists landmark-based files as "other current"** (cross-scale, per-scale omega,
   multiclass, EXTRAHEAD_CONTROL), while the archive voids their siblings.
   - The inventories exclude them.
   - **Decision: exclude.** Flag for the user, since RESULTS_INDEX is stale on this.
5. **RESULTS_INDEX EM/WM block.** It quotes "map tasks tie within 0.004", struck in favour of raw
   +0.035 / +0.070 / +0.085 from the deleted-log 16-epoch batch. Use `PAPERTASK_RESULTS.md` only.
6. **Signed_r4 torus alpha:** 0.518 (`LOCALISATION.md`) vs 0.591 (`RECENCY_H2.md`). Quote each only
   beside its own comparison, or lead with opposition, which has no disagreement.
7. **Sign-ablation parameter count:** 205,785 (prereg) vs 204,757 (training log). Use 204,757.
8. **Forget gate at r=2:** +0.086 loss-matched with no MDE (`FORGET_GATE.md`) vs +0.081 raw with
   MDE 0.080 (`FORGET_CONTROL.md`, the deterministic rerun). Quote +0.081 (MDE 0.080) as primary.
9. **Per-token rewind fraction for EM_P0:** 0.459 / 0.427 + 0.541 (`SEARCH_RESULTS.md`, two
   readouts) vs 0.962 / 0.964 (`PAIRORIGIN` / `PAIRCONST`). These are different readout
   definitions. The report must define the one it quotes. Recommended: the PAIRCONST readout for
   the mechanism table, and "93-97% of solved large-k cells" for the anatomy.
10. **MQ noise gate numbers:** CLAUDE.md vs `MATCH_QUERY_GATES_P010.md` (600 episodes). Use the file.
11. **Paper appendix location for separate q0/k0:** App. A.4 in several files vs A.7 (CLAUDE.md
    correction). Fig. 4 discussion: Sec 5.4 vs App. C.3. **Check against the v4 PDF before
    citing.**
12. **LOOP_HIER_PARITY seed count:** tables n=12, scope text n=16. Use n=12.
13. **enwik8 metrics:** final checkpoint vs mean of last 5 (C28, E29). Appendix only; pick one and
    state it.
14. **PAIRSPLIT decomposition** garbled in `EM_WM_STATE.md`. Use `PAIRSPLIT_RESULTS.md`.
15. **The MiniWorld noise floor 0.150** (D17) was measured on mostly unconverged runs and then
    applied to converged conditions. Say so wherever it is used.

### 6.3 Missing experiments that would most strengthen the story

1. **A converged-recipe paper-task 2x2** (RoPE / PoPE x index / path-integrated, r=4 on the
   path-integrated arms, 8 seeds, one batch, torus recipe).
   - It defends the headline table against the one objection E11 answers only for RoPE.
   - It ties Claim 1's table to the recipe used for Claims 3 and 5.
   - Cheap. The highest value per GPU-hour for this story.
2. **Transfer across a change of structure** (train on one topology or action space, evaluate on
   another).
   - This is the TEM claim the goal is about, and nothing in the repo measures it.
   - It would not repair a weak link. It would change what the thesis's last sentence can say.
3. **A pre-registered, converged intermediate map size in MiniWorld** (e.g. grid 24 at matched
   aliasing, n=8), plus results files for the existing points. It turns the map-extent threshold
   from a post-hoc pooling into a claim.
4. **WM at the 4x budget on recency, and one second varying-offset task.** These address Claim 4's
   one-task limit and the convergence-rate reading of -0.375.
5. **Re-registering the paper-task convergence gate** at a level WM can reach, and a fixed-offset
   task with length held constant. Either would decide whether F20 enters Claim 4.
6. **The forget-clock batch** (`FORGET_CLOCK_PREREG.md`), already registered and scripted. It is
   the only pending test of the length-axis candidate account.

### 6.4 Where MONOTONE (`MONOTONE_PREREG.md`, training now) slots in

**Experiment 1: does monotone EM still solve recency?** It goes in Sec. 6.3 and Sec. 7.5.
- **SUBTRACTION NEEDED** (P1 and P3 detectably negative).
  - Sign becomes load-bearing on recency *for the shared-kernel architecture*, although WM does not
    need it.
  - Sec. 6.3's "the task sets the value" gains an architecture term: whether a constraint is
    harmless depends on how the kernel meets content.
  - This strengthens Claim 4's link between factorisation and search, and is main text.
- **WRAP SUBSTITUTES** (P1 unmeasured with MDE <= 0.15, rewind route >= 0.5).
  - Monotone EM solves through forward wrapped shifts. This directly supports the periodicity
    reading behind F12 (wrapped per-token rewinds).
  - Main-text sentence in 7.5. The crossover's recency half stays "harmless" on a second
    architecture.
- **MONOTONE HELPS EM** (P1 detectably positive).
  - A tension. A constrained search space helps the shared kernel find its solution.
  - It supports the search framing (a smaller space is easier) and complicates "the increment must
    be signed" as unconditional.
  - Main text in 7.5, with Claim 3 scoped to map tasks explicitly.
- **UNRESOLVED**: appendix K, every number reported.

**Experiment 2: does the sign account transfer to Selective RoPE's generator?** It goes in Sec. 6.1
and 6.3.
- **Q1 detectably negative and Q2 not.**
  - Claim 3's sign result upgrades from one generator to two, on both halves of the crossover.
  - Still labelled replication in regime, but no longer MapFormer-specific. Main text.
- **Q1 falsified** (not negative with MDE <= 0.10).
  - The sign condition must be scoped to MapFormer's bottlenecked, omega-read generator.
  - Claim 3's wording changes from "the increment must be signed" to "MapFormer's increment must
    be signed". Main text, as a scope limit.
- **Q2 detectably negative.**
  - The recency half fails for this generator. "The task sets the value" is weakened to "for
    MapFormer's generator". Main text.
- **M-C3 reproduction control fails.** The unpaired comparison with stored `runs/sign` arms is
  dropped. Only in-batch contrasts are used.

---

## 7. Title and abstract

**Title:** *When a Learned Path Integrator Makes a Map: Controlled Measurements on MapFormer*

Alternatives, if a shorter name is wanted:
- *Integrable, Cancelling, Findable: Conditions for a Transferable Where in MapFormer*
- *The Where That Transfers: Measurements on MapFormer's Path Integrator*

**Abstract (about 215 words):**

> MapFormer builds a sense of place by accumulating a content-dependent, signed phase and applying
> it as a rotation in attention. We report controlled measurements of what that accumulated
> "where" buys when the observation map is redrawn at test, so that every evaluation measures
> transfer to a new instance of a known structure. On the paper's task, path integration rather
> than the rotary encoding carries the result: index codes sit near the blank floor and
> path-integrated codes solve it. Three conditions decide whether it pays.
>
> - **Integrable input.** Under turn-and-forward actions the advantage over an index code falls
>   from +0.438 to +0.050. Recording realised displacement restores it to +0.488.
> - **Cancelling increments.** Forcing the increment non-negative costs 0.28 at matched loss and
>   removes every measurable gain over an index code, extending a known result to navigation. The
>   paper's rank-2 bottleneck learns a skewed basis, and rank 4 repairs it for 384 parameters.
> - **Findable solutions.** MapEM, whose shared position kernel is TEM's conjunction of where and
>   what, represents a varying-offset recall task exactly yet trails MapWM by 0.375 from scratch.
>   We trace the gap to per-token search, and per-pair position origins recover 0.215 of it.
>
> Explicit state correction, token gating and hierarchical pooling add no measurable accuracy.
> Weight-shared recursion does, where the base model trains unreliably. Transfer across a change of
> structure, and language modelling, are not tested.

Number sources for the abstract:
- +0.438 / +0.050 / +0.488: D11, `KNOB_SWEEP_n8.md`.
- 0.28: E11, Abs - Signed loss-matched at T=1024.
- "Removes every measurable gain": E11, Abs - RoPE unmeasured.
- r=4 at 384 parameters: E1.
- 0.375: F1.
- 0.215: F19, n=48.
- "Near the blank floor": A3 and C12.
- "No measurable accuracy": B5/B6 (powered negative), E18 (powered negative at T=512), C05
  (directional, unmeasured).

If the user wants the abstract to carry n and MDE, add "(n=8)" after 0.488, "(12/12 seeds)"
after 0.28, and "(n=48)" after 0.215.
