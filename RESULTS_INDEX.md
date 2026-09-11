# Results index (regenerated 2026-09-06; EM/WM line added 2026-09-11)

**For the EM/WM + position-kernel line (2026-09-09..11), read `EM_WM_STATE.md`** --
its section is below, after the clock/map crossover.

Entry point to the repository. `CLAUDE.md` is the chronological log; this file is
the current state. **48 void files live in `archive/void/`** — nothing there is
citable, and each carries a banner naming the evidence that invalidated it. The
DIAGNOSTICS that established those verdicts are kept here, because they are current
results.

## The two documents

| file | what it is |
|---|---|
| `positional_review.pdf` | **the presentable one.** 25pp review of the positional-encoding arena: the one-parameter-group classification, the log-polar unification, the relational map over eight axes, where MapEM / TEM-t / the fast-weight family sit, and the measurements. No corrections, no process. |
| `mapformer_math.pdf` | the working record. Same content plus every correction, retraction and audit finding, kept so the same errors are not made twice. |
| `papers/` | all 40 cited sources, read first-hand. `txt/` is tracked and greppable; `bash papers/fetch.sh` restores the PDFs. `papers/INDEX.md` records which claim each reading checks. |

---

## THE HEADLINE

**Path integration is what makes in-context cognitive maps work, and it is not the
positional-encoding scheme.**

A 2x2 on **the paper's own task**, matched recipe, seeds and parameters (within
0.4%), against a **measured** always-predict-blank floor of **0.506**:

| encoding | index position | path-integrated |
|---|---|---|
| **RoPE** | 0.514 +/- 0.004 | 0.989 +/- 0.011 |
| **PoPE** | 0.509 +/- 0.004 | **1.000 +/- 0.001** |

Both index cells sit at the floor; both path-integrated cells solve it. Position
moves the result ~0.46, the encoding ~0.003. It holds under the paper's own
OOD-d / OOD-s protocol (`PAPER_OOD_WITH_POPE.md`, `INDEX_BASELINE_PAPER_TASK.md`).

Supported by: the paper's task and its OOD protocol; **Match-Query** (0.730 +/-
0.247 n=5 vs index 0.154, chance 0.0625, no seed overlap, survives context
destruction 0.918 -> 0.074); **family tree** (+0.115 over index, map-destroyed
condition lands on the 0.163 hub floor); and an explained residual — index models
exceed the floor only at recurrence interval 1-2, i.e. out-and-back retraces
readable from the action tokens as content (`REVISIT_DISTANCE.md`).

**Two boundaries.** The effect tracks **map extent**, not aliasing: at matched
aliasing it is -0.010 / +0.015 / +0.305 for 32 / 128 / 512 occupied cells, a
threshold rather than a gradient (`ALIASING_CONTROLLED.md`, `VISITS_TEST.md`). And
it does not survive **rotation-based actions**: under turn/turn/forward it falls
from +0.438 to +0.050, which allocentric recoding restores to +0.488
(`KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md`).

---

## The clock/map crossover -- a mechanism's value is decided by the TASK

`RECENCY_RESULTS.md`, `RECENCY_GATE_ABLATION.md`, `RECENCY_H2.md`; pre-registered
with a pre-launch amendment in `RECENCY_PREREG.md`; gates `RECENCY_GATES_K64.md`.
6 arms x 8 seeds, one batch. Task: retrieve the k-th most recent CONTENT symbol
with uncounted filler interleaved, so k is a contextual position (the answer sits
129.7 +/- 10.3 tokens back at k=64). Chance 0.0625; 2x budget moves nothing.

- **The crossover.** Forcing the increment to be monotone costs **-0.280** on the
  torus (12/12 seeds) and **-0.004** here (4/8, inside MDE); loss-matched -0.215
  vs +0.026. Interaction ~ **+0.28**. First result here showing a mechanism has a
  MATCH rather than a quality.
- **alpha is diagnostic, not descriptive.** The UNCONSTRAINED arm learns
  alpha 0.591 on the torus and **0.967** here (delta +0.376, se 0.009) while every
  CONSTRAINED arm sits at ~1.0 on both -- the control that attributes the shift to
  the choice rather than the task's statistics.
- **Mechanism established by intervention**, not correlation (gate strength does
  NOT predict accuracy, r = -0.34). Magnitude-matched: a constant increment on
  content scores 0.783, on every token 0.189 -- **+0.594 at 8/8**. An
  unconstrained MapFormer DISCOVERS CoPE's gate.
- **A fixed index code cannot count contextually**: 0.234 vs 0.96-1.00, **+0.750
  at 8/8**, larger than the navigation effect. This REPRODUCES CoPE's published
  claim in another architecture -- an anchor, not a new result.
- **Two of my four pre-registered hypotheses were REFUTED**, and one condition I
  pre-stated as decisive is confounded (a control changing only theta's scale
  collapses just as hard). Both recorded.

**The external check was run, and it is a null that our own data predicted.**
Flip-Flop LM (Liu et al. 2023), 6 arms x 8 seeds: every contrast unmeasured
(path-integrated minus index -1.23pp against an MDE of 3.99), all arms at 0.00%
in-distribution error. Our per-offset curve said so in advance -- index is 0.99 at
k=1 and only collapses from k=8, and **Flip-Flop only ever asks k=1**. The
benchmark varies distance-to-write but never ORDINAL DEPTH, which is the property
that separates a content-gated counter from an index. `FLIPFLOP_RESULTS.md`.

**Scope correction (audit 2026-09-10):** recency does not *require* a clock. A signed
accumulator whose query token rewinds the count by k-1 makes the retrieval offset zero,
and a single-`p0` EM kernel then solves the task exactly (`AUDIT_2026-09-10.md` #2). The
crossover's recency half shows a monotone increment is HARMLESS there; it does not show
recency needs one.

## EM vs WM and the position-kernel theory (2026-09-09..11) -- read `EM_WM_STATE.md`

- **The only EM/WM difference measured: recency (k-back), single-`p0` EM - WM = -0.375**
  (MDE 0.154, 0/8). By the registered readout, WM gets below loss 0.5 on 8/8 seeds and EM
  on 0/8. Map tasks tie within 0.004. `RECENCY_EM_RESULTS.md`.
- **It is a SEARCH problem.** Rewind installed and frozen: **1.000** on 8/8 seeds at T=1024
  and T=2048 (`WARM_RESULTS.md`). Installed trainable at 8x weight scale: **0.941**
  (+0.298 over the 1/64 install, 7/8). Found from scratch: **0/40**
  (`UNFREEZE_RESULTS.md`, `MAGONLY_RESULTS.md`).
- **Phase freedom in `q0/k0` is real**: +0.146 against a matched-optimiser control
  (MDE 0.086, 22/24; fresh seeds +0.113, 14/16). It does not act through the rewind.
  Magnitude freedom (-0.015) and initial coherence (-0.004) are null.
  `MAGONLY_RESULTS.md`, `D5_RESULTS.md`.
- **MapWM is NOT additive** (it rotates content Q,K). Thm 3 and its corollary are
  withdrawn. `|rho|` (+0.292, 8/8, torus) holds only for a frozen kernel ~100x below
  learned amplitude; the kernel's sign is a gauge. `AUDIT_2026-09-10.md`, `N5_RESULTS.md`.
- **Leakage test (landed 2026-09-11):** with the content -> Delta leak closed, the 8x-installed
  trainable rewind scores 1.000 (8/8), the frozen level; EM's recency deficit is entirely search.
  `NOLEAK_RESULTS.md`.

| file | status |
|---|---|
| `EM_WM_STATE.md` | **current summary**. Start here |
| `AUDIT_2026-09-10.md` | **current**. Supersedes the six files it names; its Tier-1 "sharpened" bullet is corrected in-file |
| `THEORY_KERNEL.md` | theory as written; **Thm 3 and corollary withdrawn, Thm 1 scoped, Thm 2 scoped**. Read the top block |
| `TALE_OF_TWO_ALGORITHMS.md` | **stale headline** (predates the recency batch); capacity argument stands |
| `REC_EM_PREREG.md`, `RECENCY_EM_RESULTS.md` | verdicts stand; the n=8 `sep - P0` size is superseded by D5 |
| `N5_PREREG.md`, `N5_RESULTS.md` | corollary refuted; `\|rho\|` scoped to a frozen low-amplitude kernel |
| `DOF_PREREG.md`, `DOF_RESULTS.md` | recency half superseded by D5; torus half n=8, provisional; D4 was determinism |
| `D5_PREREG.md`, `D5_RESULTS.md` | **current** n=24 sizes (read with the audit block) |
| `MAGONLY_PREREG.md`, `MAGONLY_RESULTS.md` | **current** |
| `WARM_PREREG.md`, `WARM_RESULTS.md` | W1-W3 current; **W4 reading withdrawn** (top block) |
| `UNFREEZE_PREREG.md`, `UNFREEZE_RESULTS.md` | **current**; U4's exhaustiveness withdrawn (top block) |
| `NOLEAK_PREREG.md`, `NOLEAK_RESULTS.md` | **current**. L1 accuracy half met (8/8 at 1.000), slope half not; L2 split; leakage is the whole accuracy residual |

## What else is citable

**The sign of the increment is load-bearing, and it is worth more than any other
ingredient per parameter.** At matched training loss, signed path integration beats
an index code by +0.123 / +0.195 at T=512/1024 (12/12 seeds); a monotone increment
beats an index code **nowhere**. The learned code shows why: opposition score 0.11
signed against 1.85-1.98 monotone, where 2.0 means opposite actions are identical.
CARoPE's published parameterisation reaches 1.98. Zero parameters.
`SIGN_ABLATION.md`, `SIGN_PROBE.md`, pre-registered in `SIGN_ABLATION_PREREG.md`.
*Scope: a replication of Sarrof / Grazzi / Selective RoPE §4.2 in a new regime.*

**Use r=4, not the paper's r=2.** +0.085 at T=1024 (8/8 seeds, t=3.57) for 384
parameters — a step at r=2, flat to r=32. The cause is a skewed basis, not missing
capacity: at r=2 opposite actions fail to cancel by half the action scale and
north/east are nearly parallel (|cos| 0.78 against 0.17 at r=4). It also cuts the
seed sd from 0.064 to 0.012, which sets every detectable effect size downstream.
`RANK_SWEEP.md`, `ACTION_GEOMETRY.md`, `PAPER_FIG4_REPRO.md`.

**MapPoPE-Flat is the strongest configuration on this benchmark.** 1.000 / 0.995 /
0.996 (IID / OOD-d / OOD-s, l=512). PoPE is inert without path integration (0.509)
and beats everything else with it. `PAPER_OOD_WITH_POPE.md`.
*Caveat found 2026-09-07: `MapFormerWM_PoPE` defaults to `bottleneck_r=2`, so every
MapPoPE number here is at the rank this project independently shows is
under-provisioned. r=4 has never been applied to it; a batch is running.*

**The parallel-scan claim holds, and the reason is now mechanical.** 2.6-3.3x
scaling against 14.5x (MapEM-NC) and 120x (TEMFaithful) over a 16x length increase.
TEM's cost is the missing group law: its ReLU and its non-commuting per-action
matrices each independently destroy interval-relativity, and MapFormer is exactly
the surviving configuration. `TIMING_BENCHMARK.md`, `positional_review.pdf` §7.

**The loop composes with path integration where there is headroom.** On
Match-Query, `r=4 + loop x4` is 0.986 +/- 0.020 with 8/8 seeds >= 0.941 at 204,757
parameters — the best arm measured on that task, with a positive interaction of
+0.149. `MQ_RANK_2X2.md`, `LOOP_HEADROOM.md`.

**Two of the paper's stated-but-unmeasured conjectures are refuted.** Separate
q0/k0 (+0.358 against it on Match-Query, four tasks) and the value of
non-commutativity (+0.005-0.014 for 34x the cost, below plain MapWM).
*Correction 2026-09-10: on recency (k-back) the separate form is BETTER, +0.128 at n=24
(`D5_RESULTS.md`). On fresh seeds alone that is +0.073 (9/16, MDE 0.130), unmeasured. So
the sign depends on the task, and "refuted" holds only for the four map tasks
(`AUDIT_2026-09-10.md` #3).*

**CSCG's stitching negative control reproduces in MapFormer's attention.** Paired
difference +0.131 +/- 0.024 (floor exactly 0) against index -0.005 +/- 0.016.
`STITCH_ATTENTION.md`.

---

## Live negatives — do not re-run these

- **Level 1.5 / the InEKF is stabilisation, not inference.** At n=5 no individual
  component is load-bearing (`L15_ABLATION.md`), a filter-free capacity control
  ties it on lm200 (`EXTRAHEAD_CONTROL.md`), and — the sharpest test — its benefit
  does **not** grow with the drift it exists to correct: +0.003 and -0.141 across
  two recipes when stochastic transitions are added (`MQ_NOISE_2X2*.md`). The
  earlier "measurement-driven" framing in this file is withdrawn.
- **Refining theta across depth does nothing**, in the regime built for it: flat at
  zero with no slope in noise, and the learned gate declines to fire
  (`NOISE_REFINE.md`).
- **Hierarchy** helps compositional transfer and long-horizon aggregation and costs
  precise retrieval; an aggregate-task "win" was a training-length confound.
  Oracle room-aligned pooling did not help; frame-reset made both metrics worse.
- **Hex emergence** does not follow from architecture, correction stacking, or the
  Sorscher conditions as implemented here.
- **PC and Kalman are duals, not complements** — gradient descent finds the
  degenerate joint optimum unless the two are fully isolated.
- **MoR-style depth routing has nothing to route on here**: an oracle per-stratum
  router buys +0.007 against a seed sd of 0.152.
- **The octave prediction for PoPE wrapping is refuted** (+0.077/+0.095/+0.079,
  flat in grid size); the length half holds 3/3. `POPE_WRAPPING.md`.
- **An explicit content gate separates what from where, verifiably, and buys
  nothing.** `Delta = sigmoid(W_g x) * (W_out W_in x)` reaches a **4.16x**
  action-vs-observation gate ratio (8/8 seeds, against a 1.35x floor set from
  Selective RoPE's gate) -- and improves accuracy nowhere: +0.004 torus, -0.039
  recency (both unmeasured), while fitting worse. MapFormer's linear bottleneck was
  already separating adequately, which **corroborates the paper's design** rather
  than improving it. The inference that produced this batch was mine and was wrong:
  "the gate is load-bearing" does not imply "the model needs help building one".
  `GATED_RESULTS.md`.
- **The forget gate's +0.086 has no identified mechanism**: anti-correlated with
  lambda, 5/8 seeds learn lambda<0, and the frozen-lambda control lands on Vanilla.
  `FORGET_GATE.md`, `FORGET_CONTROL.md`.

---

## Standing rules (each bought by a failure)

1. **n-gram on the ACTION STREAM ALONE, orders 1-5**, before any demonstration task.
2. **Context-destruction ablation on trained models.** hier-goal 0.912 -> 0.913
   (void); Match-Query 0.918 -> 0.074 (passes).
3. **Never compare a fresh variant to a stored baseline.** Retrain every arm in one
   batch.
4. **Report the measured chance rate**, and check which column it belongs to.
5. **Verify the training budget before reading a weak number as a negative** — but
   two budget points make a line, not a trend.
6. **Three seeds is not a point estimate**, and this applies to your own fresh
   numbers, not just other people's.
7. **A gate must CALL the task code, not reimplement it.**
8. **A retraction must be applied to the GENERATORS, not just recorded.** Grep the
   code for retracted numbers, not only the docs.
9. **Check whether accuracy is just the training loss.** r has been as strong as
   -0.996; loss-match when |r| > 0.5 and report both.
10. **Verify convergence and the LR schedule.** LinearLR from step one cannot escape
    a plateau late; one arm moved 0.448 -> 0.990 on the same task.
11. **"Null" requires power.** Report the MDE; say "unmeasured" otherwise. And check
    whether the cell could have gone the other way — a baseline at 1.000 +/- 0.000
    cannot show a deficit of any size.
12. **Put the seeds on the comparison you are CLAIMING.**
13. **Balance the GPU picker to the less-loaded device**, and do not interleave job
    types against an alternating picker.
14. **Do not infer held-out accuracy from training loss** (0.03 loss -> 0.674 acc).
15. **Set pre-registered branch boundaries against the measured noise floor.**
16. **Never edit a running bash script**; kill and relaunch.
17. **Check a mechanism's PREMISE applies to the task before testing it.**
18. **Check whether the knob is a RUNTIME argument** before specifying a sweep.
19. **Split a hypothesis before testing it**, so a failed test kills only what it hits.
21. **Read the CODE, not its COMMENT.**
23. **Verify WHAT a probe measures, not just that it ran.**
24. **Post-hoc truncation is not a sufficiency test.**
25. **Check whether a "failure to reproduce" is the paper's own reported result.**
26. **`pgrep -f` / `pkill -f` match the AUTHOR's shells too.** Filter by
    `ps -o comm=` and require a real interpreter.
27. **A failed `git add` stages NOTHING** — one bad pathspec kills the whole call,
    and ` M` with a leading space in `git status --short` is UNSTAGED. Verify
    `git show HEAD:<file>`, not the absence of a crash.
28. **Case matters in verification greps.** A case-insensitive search for `rope`
    matches `p-rope-rty`; one for `Undefined` misses LaTeX's `undefined`.

From the EM/WM line. These are numbered **27-33 in `CLAUDE.md`**, which collides with 27-28
above; the labels here avoid the clash. Details in `EM_WM_STATE.md` Sec 7.

- **C27. A same-seed rerun is determinism, not replication.** Report fresh seeds alone
  beside any pooled estimate.
- **C28. Check for a sign or scale GAUGE before registering a contrast.**
- **C29. Existence before mechanism**: construct a solution in the class before calling a
  deficit a class limit.
- **C30. Report the registered primary readout** even when the verdict is obvious.
- **C31. A parameterisation change is an optimiser change** (Adam's relative step size).
- **C32. Existence, then stability**: warm-start frozen AND trainable.
- **C33. Install a warm start at the scale training would use.**

Two method notes that are not rules but cost real time:

- **`shuffle` and `resample` are not interchangeable.** Permuting slots destroys the
  walk's autocorrelation and puts the input off-manifold; substituting a stream from
  an independent episode does not. Report both.
- **An ablation landing BELOW the floor** means the model fails confidently rather
  than hedging. Check with an on-manifold resample before blaming the manipulation.

---

## VERIFIED — safe to cite (24)

- `ABLATE_COMPOSITIONAL.md`   (new 2026-08-18)
- `ABLATE_FAMILY_TREE.md`     (new 2026-08-18)
- `EM_COMP_SAMEBATCH.md`
- `FAMILY_TREE_GATES.md`
- `FAMILY_TREE_RESULTS.md`
- `INDEX_BASELINE_PAPER_TASK.md`  (new 2026-08-18)
- `LEVEL15_MEETS_GATED_matchq.md` (new 2026-08-18)
- `LEVEL15_MEETS_GATED_paper50.md` (new 2026-08-18)
- `LM200_CORRECTED_MULTISEED.md`  (RECLASSIFIED — see note below)
- `MATCH_QUERY_EM.md`
- `MATCH_QUERY_GATES.md`
- `MATCH_QUERY_LONGQ.md`
- `MATCH_QUERY_RESULTS.md`
- `MATCH_QUERY_SCALE.md`
- `NOISE_CLEAN_REVALIDATION.md`
- `PAPER_OOD_PROTOCOL.md`
- `PAPER_OOD_WITH_POPE.md`     (new 2026-08-18)
- `PAPER_TASK_ABLATION.md`     (new 2026-08-18)
- `PAPER_TASK_ACCURACY.md`
- `PAPER_VALIDATION.md`
- `REVISIT_DISTANCE.md`        (new 2026-08-18)
- `STITCH_ATTENTION.md`        (new 2026-08-18)
- `TIMING_BENCHMARK.md`
- `VOCAB_SWEEP_MULTISEED.md`

**Note on `LM200_CORRECTED_MULTISEED.md`.** It is the corrected, fresh,
multi-seed lm200 leaderboard under current code (Level15 0.990 +/- 0.005 vs Vanilla
0.742 +/- 0.075) and the numbers stand. Its *interpretation* does not: a
filter-free capacity control ties it (`EXTRAHEAD_CONTROL.md`), so the gap is not
evidence for the Kalman mechanism.
Its one outstanding gap is that lm200 has never had a context-destruction
ablation (rule 2).

### Added since 2026-08-18

- `SIGN_ABLATION.md`, `SIGN_PROBE.md`, `SIGN_ABLATION_PREREG.md`
- `RANK_SWEEP.md`, `ACTION_GEOMETRY.md`, `PAPER_FIG4_REPRO.md`, `RANK_TRUNCATION.md`
- `MQ_RANK_2X2.md`, `LOOP_HEADROOM.md`, `L15_LOOP_2X2.md`, `LOOP_SAMPLED.md`
- `DXR_RANK_THRESHOLD.md`, `ND_GATES.md`
- `SELECTIVE_ROPE.md`, `GATE_PROBE.md`, `CONV_KERNEL_PROBE.md` (per-knob rows are
  attributable only up to initialisation — see the caveat in the review)
- `FORGET_GATE.md`, `FORGET_CONTROL.md`, `LAMBDA_TRACE.md`
- `POPE_WRAPPING.md`, `ALIASING_CONTROLLED.md`, `VISITS_TEST.md`
- `L15_ABLATION.md`, `NOISE_REFINE.md`, `MQ_NOISE_2X2.md`, `MQ_NOISE_2X2_C2.md`
- `RECIPE_POWER.md`, `LOOP_DEPTH_STRATA.md`, `ROPE_CANONICAL.md`
- `MINIGRID_FULL_2X2X2.md`, `KNOB_SWEEP_n8.md`, `HABITAT_BUILD.md`
- `papers/INDEX.md` — the source corpus and what each reading verifies

## DIAGNOSTICS — what invalidated things (10)

- `AP_KERNEL_DIAGNOSTIC.md`
- `CORRECTED_LM200_LEADERBOARD.md`
- `HIERGOAL_ABLATION.md`
- `HIERGOAL_CLOSEDLOOP.md`
- `LAP_GATES.md`
- `LAP_TRANSFER.md`
- `LAP_TRANSFER_NOREWARD.md`
- `LM200_CORRECTED_MULTISEED.md`
- `MAP_QUERY_GATES.md`
- `PLANNER_TASK_AUDIT.md`

## PARTIAL — lm200 rows void, clean/noise VALID (14)

- `CAPACITY_PERREGIME.md`
- `GSF_FULL_RESULTS.md`
- `LEVEL15BETA_RESULTS.md`
- `NOBYPASS_RESULTS.md`
- `NODROP_PARETO_RESULTS.md`
- `RESULTS_PAPER.md`
- `TEM_NOISE_FFN_RESULTS.md`
- `TEM_RESULTS.md`
- `TEM_T_MULTISEED.md`
- `TEM_T_RESULTS.md`
- `V3_RESULTS.md`
- `V4_CONTROL_RESULTS.md`
- `V4_MULTISEED.md`
- `V4_RESULTS.md`

## CONTAINS RETRACTED CLAIMS (3)

- `REPORT_ADDENDUM.md`
- `REPORT_v2.md`
- `RESULTS_SUMMARY_2026-05-10.md`

## SUPERSEDED / corrected (4)

- `CAPACITY_CONTROL.md`
- `COMPOSITIONAL_RESULTS.md`
- `MAP_QUERY_RESULTS.md`
- `VOCAB_SWEEP_RESULTS.md`

## Known-open, ranked

1. **Level15 / correction line vs the gated tasks, beyond n=3.** The Match-Query
   comparison is underpowered (sd 0.14-0.26; every arm has a 1.000 seed).
   It establishes "no advantage", not "a deficit".
2. **lm200 context-destruction ablation** — the +24.8pp result has never been
   through rule 2. Blocked: no lm200 checkpoints on the koopman machine.
3. **`DOG_RESULTS_FIXED.md` has never been produced.** The existing
   `DOG_RESULTS.md` used all-zero DoG targets (unnormalised Gaussians cancel at
   d=0), so the Sorscher hex test is VACUOUS, not negative.
4. **`STOCHASTIC_TRANSITION_RESULTS.md` never landed** (queued 2026-05-01).
5. **RoPE on Match-Query** — its index controls are PlainFlat and PoPE-Flat, not
   the architecture-matched RoPE that made the paper-task result tight.
6. **Map-Query** — room query is learnable at 7.6x chance but on ONE variant,
   ONE seed; the multi-seed table is the 25-epoch undertrained one, and
   `train_map_query.py` still defaults to 25 epochs.
7. **Schema task** — still NOT READY; needs the redesign stitch received.
8. **Vanilla (WM) on the family tree** — it has EM and MapEM-NC but no plain WM.

## other current (79)

- `AGGREGATE_EXTRAS.md`
- `AGGREGATE_MULTISEED.md`
- `AGGREGATE_TASK_RESULTS.md`
- `BOUNDED_MEMORY.md`
- `BOUNDED_MEMORY_RESULTS.md`
- `BUMP_TOKEN_RESULTS.md`
- `CASCADE_MULTISEED_RESULTS.md`
- `CASCADE_REPRO_TEST.md`
- `CASCADE_ZEROSHOT_S0.md`
- `CLAUDE.md`
- `CLOCK_SCAN.md`
- `CLONE_ANALYSIS_LEVEL15PC.md`
- `CLONE_TRANSFER_NOBYPASS.md`
- `CNAV_HEX_Level15.md`
- `CNAV_HEX_Level15EM.md`
- `CNAV_HEX_Vanilla.md`
- `CNAV_HEX_VanillaEM.md`
- `CNAV_RESULTS.md`
- `COMPOSITIONAL_EXPERIMENT.md`
- `COMPOSITIONAL_MULTISEED.md`
- `DAGGER_DK6_RESULTS.md`
- `DAGGER_EMPTY_RESULTS.md`
- `DAGGER_RESULTS.md`
- `DETAILED_RESULTS.md`
- `DOG_RESULTS.md`
- `DOORKEY_BC_RESULTS.md`
- `EM_FIX_COMP.md`
- `EM_HOPFIELD_CROSSSCALE.md`
- `EM_P0_COMP.md`
- `EM_P0_PAPER.md`
- `EXTRAHEAD_CONTROL.md`
- `FAMILY_TREE_D7_GATES.md`
- `GENERALIZATION_REPORT.md`
- `HIERGOAL_LONGT.md`
- `HIER_ATTN_LONGT.md`
- `HIPPOCAMPAL_ANALYSIS.md`
- `HIPPOCAMPAL_GRID.md`
- `HIPPOCAMPAL_GRIDL15PC.md`
- `HIPPOCAMPAL_GRID_FREE.md`
- `HIPPOCAMPAL_HIDDEN.md`
- `HIPPOCAMPAL_HIDDEN_GRIDFREE.md`
- `HIPPOCAMPAL_LEVEL15PC.md`
- `HOPFIELD_NOMAINAP_RESULTS.md`
- `HOURGLASS_README.md`
- `LEVEL15EM_CROSSSCALE.md`
- `LONG_SEQ_clean.md`
- `MATCH_GATES_128_16.md`
- `MATCH_GATES_64_16.md`
- `MATCH_GATES_64_4.md`
- `MINIGRID_DK16_RESULTS.md`
- `MINIGRID_DOORKEY_CACHED.md`
- `MINIGRID_DOORKEY_LONGT.md`
- `MINIGRID_DOORKEY_RESULTS.md`
- `MINIGRID_DOORKEY_ROPE_DIAG.md`
- `MINIGRID_MEMORY_RESULTS.md`
- `MULTICLASS_MULTISEED_RESULTS.md`
- `MULTICLASS_RESULTS.md`
- `MULTISEED_FOLLOWUP.md`
- `MULTISEED_FOLLOWUP_RESULTS.md`
- `NUMBERLINE_RESULTS.md`
- `OMEGA_RESCALE_clean.md`
- `OOD_GRID_RESULTS.md`
- `PERSCALE_OMEGA_RESULTS.md`
- `PER_VISIT_clean.md`
- `PUBLICATION_VENUES.md`
- `README.md`
- `RECURSIVE_RESULTS.md`
- `REPORT.md`
- `ROUTE_ATTN_RESULTS.md`
- `R_T_DISTRIBUTION_3WAY.md`
- `SESSION_2026-05-01.md`
- `SESSION_HIERARCHICAL_CASCADE.md`
- `SPACETIME_HIER_RESULTS.md`
- `TEM_BACKGROUND_BASELINES.md`
- `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `TOPOLOGY_RESULTS.md`
- `VECTOR_NAV_V2_RESULTS.md`
- `ZERO_SHOT_TRANSFER_clean.md`
- `ZERO_SHOT_TRANSFER_clean_brokeninit.md`
