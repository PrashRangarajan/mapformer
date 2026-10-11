# Critical review of the 2026-10-04 .. 10-10 experiments (2026-10-11)

Independent, adversarial review requested by the owner ("what is the purpose of all of this?"). Read-only: no existing
file modified, no GPU job. Two CPU checks were run from committed artifacts (scripts in the session scratchpad, not
committed; reproduced inline below so they can be re-run).

Terms. **Path model / MapWM**: MapFormer, rotary angle theta = omega * cumsum(W_out W_in emb(token)). **Index model /
RoPE**: rotary angle = token index. **Step**: a token's contribution to theta. **Score rule**: how the attention logit is
formed from content and phase (MapWM rotates content q, k by theta; PoPE / the gain arms multiply non-negative content
magnitudes by a content-free cosine kernel). **SOLVED**: final-5% training loss < 0.05. **MDE**: minimum detectable
effect. **Floor**: best trivial predictor. **Re-score**: multiplying attention outputs by 1/(1-p) at eval.

---

## Summary of the most severe problems (most severe first)

1. **The registered "STATE BOUND TO PLACE" (TW_STATECHANGE C) is passed by a rule with no state semantics.** The floor
   it beats (F2 0.649) is location-FREE. A location-aware, verb-agnostic rule -- "answer with the last noun told while
   at this cell, ignoring which verb" -- scores **1.000 on T2drop** (0.841 if aside nouns count), computed on the
   batch's own eval stream (check 1 below). DirOnly, which has no step on any state word, scores 0.926 on T2drop
   (`TW_STATECHANGE_ANALYSIS.txt`), i.e. it would also pass C. C shows path integration + "the most recent noun at this
   phase wins"; it does not show the model represents that a drop changes the world. The stratum that does require state
   semantics, T2take (the naive rule scores 0.000 there), was demoted to "no decision" because the constant answers it.
2. **Overclaims of mechanism built on one uncorrected secondary p-value.** SCORE_RANK's "PoPE's score repairs short-gap
   revisits (+0.115, p 0.036), not wrap-only ones" is one of three strata secondaries (Holm-adjusted p ~0.11). It became
   "two separate defects, two separate fixes ... the rank defect is the periodic code" (`SCORE_RANK_RESULTS.md:51-54`,
   `CLAUDE.md:79`, report `language_summary.html:372`) and then the premise of GAIN_PHASE's "two defects, two fixes"
   (`GAIN_PHASE_RESULTS.md:64`, `CLAUDE.md:81`, report line 559). The strata do not even isolate wrap: rank-2 plain
   revisits at gap >= 128 are also at 0.75-0.79 vs rank 4's 1.000 (`SCORE_RANK_ANALYSIS.txt`), so "long gap" and
   "periodic code" are not separated. The test that would separate them (RANK_NOWRAP) was postponed.
3. **GAIN_PHASE's "two defects" never showed the second defect on its own task.** D2 (score fix alone) is NO DIFFERENCE
   under both steps (+0.0011, p 0.95; +0.0054, p 0.135). The only demonstrated defect is the leak (~1% accuracy). Its
   rotary-side step effect replicated only on SOLVED, and SOLVED is a loss threshold that Amendment 3 itself measured to be
   ~97% leak (`GAIN_PHASE_PREREG.md` Amendment 3 item 1) -- so the SOLVED firing largely restates "NormStep removes a
   leak that dominates the loss". On accuracy, LEAK's +0.0107 did not replicate (+0.0045, p 0.46).
4. **TinyStories prose contains two errors that a readout check exposes** (check 2 below). (a) `mr` is listed as ticking
   the clock "~2x faster" (`TINYSTORIES_PILOT_RESULTS.md:24`); its step is **anti-parallel to '.'** (cosine -1.00, -0.98,
   -0.99 on s0-s2): it steps BACKWARDS and cancels the abbreviation period. That is the one learned cancellation in the
   table, and it contradicts "nothing cancels" (line 28) -- the opposition test only looked at pre-chosen semantic pairs.
   (b) "spatial words (up, down, in, out, inside, back, home ...) 0%" (line 27; report line 489): the probe assigns
   up/down/in/out/into/over/under/there/here to the FUNCTION class first (`tinystories_probe.py`: `SPA = ... - FUNC`), so
   the "spatial" class held only away/back/home/inside/outside. With the full list the share is 1.4%, against a
   frequency share of 2.5%. The class "shares" are frequency shares: motion verbs are 1.5% of tokens and step at 0.62-0.67x
   the mean token; "motion words do not move it" is "motion words are rare and step a little less than average".
5. **Factual error in the task description.** `TW_STATECHANGE_RESULTS.md:11` ("She picked up the cat.") and
   `CLAUDE.md:72` ("'picked up / dropped the X' sentences") describe a phrase the prereg deliberately excluded
   (`environment_tw_statechange.py:13-15`: take verbs took/grabbed/lifted; "picked UP" excluded because UP is a direction
   word). A reader would infer the model handled a direction word in a non-movement sense; it was never tested.
6. **Ceiling-bound "precision" claims.** MAPPOPE_PAIR's COUNT CI [-0.0004, +0.0008] "excludes anything above 0.0008, about
   3% of the total" (`MAPPOPE_PAIR_RESULTS.md:29-30`, `CLAUDE.md:79`) is computed where both arms are 16/16 SOLVED at
   0.9995 / 0.9997: the maximum possible gain was 0.0005. The bound is the ceiling, not a measurement. GAIN_GRAIN's
   non-inferiority (S vs P one-sided p < 0.0001, printed 90% CI "[+0.0004, +0.0004]", degenerate) is likewise guaranteed
   once both arms sit within 0.002 of 1.000; it escaped the registered "(at ceiling)" label by single seeds at 0.997-0.998.
7. **Stale claim not corrected.** `MAPPOPE_PAIR_RESULTS.md:47` still says "**PoPE's score rescues rank 2**" with no
   CORRECTED block, after SCORE_RANK found NO RESCUE where rank 2 actually fails (T=1024). At T=128 rank 2 does not fail
   (MapWM r2 18/20 SOLVED in GAIN_GRAIN), so there was nothing to rescue there either.
8. **Drift away from the project's own plan, at ~2 days of two-GPU wall time.** The 2026-10-04 plan
   (`docs/theory/2026-10-04/00_PLAN.md` section 2) ranked, after the re-score, the cheap causal tests of the one novel
   result: 1D ring rank 1 vs 2 (~1.2 h), rank escape test (~2 GPU-h), forbid drift (~6 h), no-wrap grid (~3-4 h), and
   the baseline table "required by any reviewer". None was run. Instead: MAPPOPE_PAIR (~4 h) -> SCORE_RANK (~13.5 h,
   01:12-14:39 on 10-05) -> GAIN_GRAIN (~4.5 h) -> GAIN_PHASE (~14 h, 19:04 10-08 to 09:05 10-09) -> TW_STATECHANGE
   (~6 h) -> TinyStories (~3 h) -> TW_LANDMARK pilots (~4 h) -> TW_AMBIG and RANK_NOWRAP prepared, not run. Each was
   motivated by a post hoc observation or a descriptive secondary of the one before.

---

## A. Purpose

The question the project serves, in one line: **when does a learned, content-dependent rotary phase let a transformer
build a relational "where" separate from "what" (TEM's factorisation), and is that a capability or only robustness?**
The citable answer before this week was: path integration helps in distribution on navigation, substitutes for depth,
and the per-head rank of the content-to-angle map must exceed the task dimension (the one result with a claim to
novelty; sign is a replication). This week did not advance that core. MAPPOPE_PAIR and SCORE_RANK touched it
(SCORE_RANK is the best-motivated experiment of the week: a clean 2x2 at matched init that cleanly said NO RESCUE),
but the follow-ons are rabbit holes: GAIN_GRAIN came from a self-constructed "brain-constraint scorecard"
(`docs/theory/2026-10-05/neuro_design.md` section 1, weights chosen by the authors) and tested on a task where every
arm is at 0.990-0.9999; its descriptive "3x faster" secondary then motivated GAIN_PHASE (14 h), where the speed-up did not
replicate (x1.03) and the score fix did nothing measurable. TW_NORMSTEP, TW_STATECHANGE and TW_LANDMARK extend a
scripted 58-65-word grammar at one layer, where the index baseline is at the constant floor by construction, so every
"path needed" verdict was predicted at power 1.00 before launch. TinyStories was a reasonable cheap reality check and
its honest headline (MapWM = RoPE) is the most useful language result of the week. The line is drifting: the number of
amendments per batch (GAIN_PHASE 3, RANK_NOWRAP 4, TW_LANDMARK 4) and the commit count (83 since 2026-10-03) now
outrun the scientific content per GPU-hour.

Publishable and novel? Given the project's own prior-art list: the taxonomy and content-dependent rotation are GRAPE /
Mamba-3 / Selective RoPE; non-negative content magnitudes on a content-free position kernel is PoPE's design, and a
scalar version (GainScalar) is a special case of PoPE / separable content x position attention -- not novel, and only
shown "as good" at ceiling; the separation of what/where in path-integrated transformers is TEM-t and MapFormer Fig. 9.
The candidate paper remains the per-head rank result (with the escape test, a no-wrap test and baselines), plus sign
and depth substitution as supporting replications. Nothing from this week is a headline for it; SCORE_RANK is a useful
negative control paragraph.

Per experiment, does it advance the question?
| experiment | motivation | advances the core question? | verdict on value |
|---|---|---|---|
| Dropout re-score | necessary hygiene | indirectly (no verdict changed) | well-motivated; keep as a methods note, but see B1 |
| TW_NORMSTEP | predicted NormStep failure mode | no; A unmeasured below 0.096 by design | low value; the useful finding (aside cap on DirOnly) was post hoc |
| MAPPOPE_PAIR | a confound flagged three times | marginally (which part of MapPoPE matters, on a near-ceiling task) | reasonable, cheap |
| SCORE_RANK | direct test of a rank hypothesis | yes (negative) | well-motivated, but 13.5 h after a 40-epoch pilot that suggested the opposite |
| GAIN_GRAIN | brain scorecard design | no; ceiling, NO HEADROOM | rabbit hole |
| GAIN_PHASE | combine two fixes from the scorecard | no; one defect shown, ~1% effect | rabbit hole, 14 h |
| TW_STATECHANGE | language world beyond the walk | weakly; C is passed by a naive location rule | medium; headline overstated |
| TinyStories pilot | reality check | yes (negative: no benefit in natural text) | valuable and cheap; prose errors |
| TW_LANDMARK | cue competition | not as designed (see C) | stop |

---

## B. Validity, per experiment

### B1. Dropout-scale re-score (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`, `rescore_hook.py`)
- Correct as an engineering check: scale-none reproduces every registered number exactly; the hook is narrow (known
  layer classes, skipped classes reported). Numbers match `rescore_other_compare_out.txt` and
  `rescore_torus_compare_out.txt` (text world 0.9692 -> 0.9934; TW_NORMSTEP DirOnly - MapWM -0.0006 -> -0.0240).
- It is an ad hoc correction, not a better estimator: it LOWERS multi-layer and looped models (LOOP_RANK 4L 0.9901 ->
  0.9726), so it is not "the true accuracy". The principled fixes are an MC-dropout average (already implemented in
  TW_LANDMARK's Amendment 1) or retraining with attention dropout 0. The models that need it are below ceiling and
  depend on a scale they will never see at eval: that is a property of the training recipe worth fixing, not an
  under-report to correct.
- Forking paths: after the re-score made the DirOnly - MapWM contrast detectable in TW_NORMSTEP (power 0.85 re-scored vs
  0.18 eval mode, `TW_STATECHANGE_PREREG.md` power table), TW_STATECHANGE registered the re-scored accuracy as primary.
  Legitimate as a prereg, but every "learned steps beat the oracle" claim (CLAUDE.md:71-72, report lines 146 / 205-206)
  rests on a scoring rule chosen after seeing that it produces that difference; in eval mode it is NO DIFFERENCE in both
  batches. Say "after a post hoc eval correction" each time.
- Minor: the report says "up to 0.16" (line 206), the audit "up to +0.17" (0.1702 on NormStepNB).

### B2. TW_NORMSTEP
- A was effectively unpowered by design (MDE 0.096, bimodal per-seed accuracy known from the text world). ~3 GPU-h for
  a verdict that could only read "unmeasured" unless NormStep failed catastrophically.
- B: the Amendment-1 readout (drift over optional words) was built to be gauge-invariant to a per-move clock and fired
  (+0.057 rad vs a 0.05 floor, p 0.027; paired p 0.047). The theory review then showed 0.126 of 0.133 rad is ASIDE
  sentences (`word_drift_split_out.txt`). So the registered readout did not measure the thing it was named for; the
  CORRECTED block is honest. Note that the aside offset is itself a count of aside sentences carried into later phase
  (a net offset that is not undone), so "aside offset, not a clock" is a relabel; "not a word-count clock from beta" is
  what is supported.
- The DirOnly finding ("100% of its errors are aside nouns", 2 of 8 seeds) is a ONE-LAYER fact: an aside noun's key at
  one layer carries only its identity and phase, so with no step it must collide with the cell. A 2-layer DirOnly can
  mark aside nouns from context. "'Only the action words should move' is the wrong target in language"
  (`TW_NORMSTEP_RESULTS.md:66-67`, theory T3, report line 205) generalises from one layer and one scripted grammar.

### B3. MAPPOPE_PAIR
- Clean construction (equivalence to tied MapPoPE-Flat 0.0, bitwise reproduction of paper2x2). Registered SCORE RULE
  follows from the data. r(final loss, acc) -0.963 and MapWM 10/16 SOLVED: SCORE is "PoPE's score converges more
  reliably at 300 epochs", i.e. likely speed (GAIN_GRAIN: P 74 vs W 132 median epochs). Not said in the results file.
- COUNT CI and the "3%" claim are ceiling artefacts (summary item 6). The OOD COUNT (+0.003 at T=1024, p 0.82) is the
  only informative reading of COUNT and it is out of distribution.
- `MAPPOPE_PAIR_RESULTS.md:47-50` "PoPE's score rescues rank 2" is stale (summary item 7).
- GAIN_GRAIN's fresh-seed replication shrank SCORE from +0.024 to +0.009 (MapWM 18/20): the MAPPOPE_PAIR seeds 10-25 were
  a harder draw. The CLAUDE.md row still leads with 16/16 vs 10/16.

### B4. SCORE_RANK
- Best-designed batch of the week (2x2 at matched init, positive control, pre-specified n from a power table).
  Registered NO RESCUE follows (3/12 vs 3/12; accuracy +0.098, p 0.089, MDE 0.155).
- Note the premise moved: MapWM r2 solved 3/12 here vs 0/8 in RANK_MI and "0, 2, 2 of 8" in CLAUDE.md's rank table. The
  pooled rank-2 solve rate is higher than the headline table conveys; the table should carry it.
- Overclaims (summary item 2): the strata are secondaries (3 strata, smallest p 0.036, Holm ~0.11); "no score rule
  changes the code" (`:37`) generalises from one alternative score; "the rank defect is the periodic code" is not
  separated from "long gaps" (plain gap >= 128 also fails at rank 2: 0.749 / 0.792 vs 1.000).
- "SOLVED iff a clean head 39/40" is presented as an out-of-sample pass (`:55-56`). The thresholds were set post hoc on
  80 rank runs and the plan itself calls the solved -> clean direction "close to circular" (`00_PLAN.md` T1). GAIN_GRAIN
  shows it fails at T=128 (5-16/20). It is a description of end states, not a test.
- Cost discipline: the 40-epoch pilot "made RESCUE the likely outcome" on a different schedule, and 13.5 h were spent
  on n=12. Fine as a decision, but it is the pattern of item 8: the expensive batch answered a question raised post hoc.

### B5. GAIN_GRAIN
- (a) SCALAR GAIN SUFFICES is registered with the NO HEADROOM qualifier, correctly. It means: on a task where MapWM
  itself solves 18/20 and every gain arm is 0.9995-0.9999, nothing is worse. It is not evidence that per-frequency gain
  is unnecessary anywhere it could matter. The CLAUDE.md row title ("One non-negative gain per token is as good as
  MapPoPE's per-frequency gain") drops the qualifier from the headline position.
- Printed 90% CI "[+0.0004, +0.0004]" for S - P and "[+0.0002, +0.0002]" for M - P (`GAIN_GRAIN_ANALYSIS.txt`) are
  zero-width: the inversion of the shift test is degenerate at the ceiling. A zero-width CI should have triggered a
  check; the non-inferiority p "0.0000" comes from the same machinery.
- (b) NON-NEGATIVE WORSE (12/20 vs 19/20): softplus(q.k) also halves the gradient at init and adds a positive mean
  (stated). The secondary reading "the form of the gain matters, not the sign (scalar product gain is best)" (CLAUDE.md
  row) rests on S - N, a bundle contrast (factorised vs dot product, delta 0 vs learned offsets, 32k fewer parameters,
  different score scale) and on "best" among arms that differ by <= 0.0004 at ceiling.
- "Trains ~3x faster" is descriptive, confounded by the new score scale (2/sqrt(d_h)), the learned non-negative
  spectrum and 32k fewer parameters; it did not replicate at rank 4, T=1024 (GAIN_PHASE D5, x1.03).
- The remap-probe readings for S, E and N are by construction (the prereg says so); they are not findings.

### B6. GAIN_PHASE
- Engineering and pre-registration are thorough (three audits; leak readout L_ms validated on 24 LEAK checkpoints with
  positive controls; the leak-free convergence gate is a good fix). The numbers in the results file match
  `GAIN_PHASE_ANALYSIS.txt`.
- D3 SEPARATE DEFECTS is close to by construction: the leak is defined in the step map (S_id reads only W_out W_in), the
  gain score does not touch the step map, and nothing in the gain score penalises an identity step more than the rotary
  score does. The informative possibility (training unlearns the step under the gain score) was unlikely a priori. The
  generalisation "no score rule removes it" (`:30`) is from one alternative.
- D1 "NORMSTEP HELPS UNDER BOTH SCORES": the rotary half fires on SOLVED only, and SOLVED here is mostly the leak (summary
  item 3). Accuracy did not replicate. The label should have carried "[rotary: SOLVED only; SOLVED is ~97% leak]".
- D4 "THE GAIN-PHASE MAP WORKS" is fair on accuracy (+0.0099 vs MapWM, p 0.0002), but the comparison to NormStep is NO
  DIFFERENCE, and "the only arm with no weak seed" (`:49-52`, report line 191) is a post hoc variance observation on n=8.
- Practical size: the whole leak line is a ~0.5-1% accuracy effect on unseen-object revisits in a one-layer model whose
  MapWM arm is still descending at 900 epochs (budget-scoped). 14 h for this is poor value.

### B7. TW_STATECHANGE
- A PATH NEEDED: predicted at power 1.00; RoPE 1L sits at the constant floor on 8/8, as on the text world. A one-layer
  index model cannot integrate direction words at all, so this is a sanity check, not a finding; "Index RoPE at the
  same depth gets neither" (`:59`) is true by construction of the depth.
- C STATE BOUND TO PLACE: see summary item 1. Check 1 (CPU, the batch's eval stream via
  `tw_statechange_readouts.walks(make_env(HELDOUT))`, 200 walks, 5375 revisit targets):

  | verb-agnostic rule: answer = last noun told at this cell | T1 | T2drop | T2take | T3 | all |
  |---|---|---|---|---|---|
  | saw slots + state-clause nouns + aside nouns | 0.852 | 0.841 | 0.000 | 0.841 | 0.762 |
  | saw slots + state-clause nouns (no asides) | 1.000 | 1.000 | 0.000 | 1.000 | 0.897 |

  So T2drop measures "location + recency among nouns at that place", which any path model that retrieves by phase gets
  nearly for free (DirOnly 0.926). The models' real distinction from this rule is T2take (MapWM 0.997, DirOnly 1.000,
  NormStep 0.872): saying "nothing" after a take requires knowing that take empties the cell. That was the
  semantically informative stratum, and it carried no decision. The report's "the right answer differs from everything
  ever seen at that place, so the model must know both where it is and what was dropped there" (line 458) is misleading:
  the answer is the noun told at that place moments after arrival.
- B OFF THE MAP PLANE: threshold 0.10 move per clause was set at the TOP of the aside calibration range (Amendment 1
  admits OFF means "up to ~3x a median aside"); the power table shows a clause at 3x the aside shift still reads
  OFF/MIXED. The result (0.028 / 0.027, not distinguished from asides) is fine as description; "state changes do not
  move the map" means "no more than asides do", and B cannot separate state clauses from asides (stated).
- D2 DirOnly WORSE depends on the re-score (flagged); in eval mode NO DIFFERENCE.
- "This is the factorisation TEM claims, now with state changes" (`:60-61`) overstates a one-layer phase-coincidence
  mechanism on a scripted grammar with at most one change per cell.
- Prose error "She picked up the cat." (`:11`) and CLAUDE.md:72 (summary item 5).

### B8. TinyStories pilot
- Honest framing (pilot, n=3, no test). The headline MapWM = RoPE (1.5053 vs 1.5050) stands.
- Check 2 (CPU, the three committed MapWM checkpoints, the probe's own definitions):

  | seed | cos(step mr, step '.') | cos(step mr, mean step) | motion: freq share / step share / mean step vs average | spatial, full list: freq / step share | r(log freq, step size) |
  |---|---|---|---|---|---|
  | 0 | -1.00 | -0.98 | 0.015 / 0.009 / 0.62x | 0.025 / 0.014 | +0.33 |
  | 1 | -0.98 | -0.96 | 0.015 / 0.009 / 0.63x | 0.025 / 0.014 | +0.30 |
  | 2 | -0.99 | -0.93 | 0.015 / 0.010 / 0.67x | 0.025 / 0.014 | +0.25 |

  `mr` (and `mrs`, cos -0.89 to -0.94) step backwards against the sentence tick: the model learned to cancel the period
  of an abbreviation. This supports a sentence counter more precisely than the prose does, and it refutes "nothing
  cancels" and "mr ~2x faster". The probe reports step NORMS, so direction was never looked at.
- "Discourse clock" is a generous label for: ~80-87% coherent token clock + extra ticks on punctuation, line and story
  boundaries. Alternatives the pilot does not exclude: (i) frequency (step size correlates with log frequency, r
  +0.25..+0.33; class shares are frequency shares); (ii) the `<eos>` jump as an attention-isolation separator: windows of
  512 words span ~2.6 stories (535M tokens / 2.72M stories ~ 197 tokens each), so a big phase jump at `<eos>` simply
  decorrelates the previous story; calling it "structurally like global remapping" (`:54-55`) is speculation (labelled);
  (iii) tokenizer artefacts (`mr` + `.`, quotes split from words).
- Content-word substitution cost (+0.018..+0.030 nats) and the uniform-clock cost (+0.07..+0.09) measure the trained
  model's reliance on its own step table under an out-of-distribution swap. They do not show that content-dependent steps
  are useful: RoPE, which has none, ties. "About 20x the MapWM - RoPE difference" (`:43-44`) compares a lesion cost with
  a between-architecture difference; the comparison has no meaning.
- Capacity: rank 4 SHARED across 8 heads and 6 layers gives the step a 4-dimensional latent for an 8192-word
  vocabulary. A map over story locations was given little room; "nothing pushes the step toward a map" is not
  separated from "the bottleneck could not hold one alongside a clock".

### B9. TW_LANDMARK (pilots only; see C)
- Pilot numbers in Amendments 2-4 match `runs/tw_landmark_pilot/analysis.txt`, `exist/eval_out.txt`,
  `budget3/eval_out.txt`.
- The cost line in the prereg (48 runs ~7.5-8 h wall) is for the abandoned 2-layer design. At 3 layers and 1200-1800
  epochs (budget pilot: 3.6-3.9 s/epoch at 2 per GPU), 48 runs is ~48 x 1500 x 3.7 s / 4 concurrent ~ 18-19 h wall, and
  RoPE 3L was still not SOLVED at 1800 epochs.

### Cross-cutting statistics
- Roughly 20-25 registered decisions this week, each at its own .05 with "no family-wise claim". Several fired narrowly
  (TW_NORMSTEP B p 0.027 at 0.057 vs a 0.05 floor; GAIN_GRAIN (b) p 0.026 / 0.020). At this rate ~1 false firing is
  expected per week; narrow firings should not anchor narratives.
- Secondaries become headlines: SCORE_RANK's short-gap stratum (item 2), GAIN_GRAIN's speed (B5), GAIN_PHASE's
  "no weak seed" (B6), TW_NORMSTEP's DirOnly cap (B2). Each was labelled secondary in its own file and lost the label in
  CLAUDE.md rows and the shared report.
- Ceiling: MAPPOPE_PAIR COUNT, GAIN_GRAIN (a), GAIN_PHASE GainPhase vs NormStep all compare arms within 0.002 of 1.000.
  The project's own rule 5 ("check the verdict cell could have gone the other way") was satisfied in letter (ceiling
  labels exist) but not in spirit (the labels were dodged by single seeds at 0.997).

---

## C. TW_LANDMARK: stop it

1. **Not identifiable as posed.** "Do names overshadow the map" needs a path model whose accuracy depends on which cue it
   uses. The path model is at ceiling without informative names (MapWM 3L r=1: uninf 0.9974, own 0.9972; 2L: 0.9996),
   so name benefit = acc_named - acc_uninf has at most ~0.003 of room: saturated. Strip (0.45-0.50) is out of
   distribution (the per-move gauge of the always-present "reached <name>" clause disappears; Amendment 1 D1). Conflicts do
   not identify name use (Amendment 3: the name-using RoPE 3L also follows names on <= 3% of conflicts). The registered
   in-distribution probe O-ID reads reliance at UNNAMED cells, where names cannot help; for an exact integrator at
   ceiling its null outcome (PERSISTS, 0.88 under the null) is close to certain.
2. **The architecture cannot express the biological question.** MapFormer's theta is a cumsum with no reset; a name can
   be a parallel retrieval route but can never correct theta (the prereg says so: "names can only substitute for the map,
   never correct it"). Cue competition between a landmark and path integration in the same representation is therefore
   absent by construction. What can be asked is whether a redundant second route is learned -- and with noise-free moves
   and an exact integrator, the path route is learned at 1 layer while names need 3 (index model). A redundant, harder
   cue is expected to be blocked (Kamin). The pilots already point that way: MapWM 3L r=1 has reliance 0.519 with
   INFORMATIVE names present (`budget3/eval_out.txt`, rel named), equal to its reliance with useless names (0.519) -- knocking
   out the path costs the same whether or not names are there, so the path model has no usable name route (n=1).
3. **Cost is now ~18-19 h, not 7-12**, for an outcome largely visible at n=1.
4. **If the owner wants a landmark result**, the cheap well-posed version is: make names NECESSARY, e.g. "teleport"
   moves rendered without a direction word followed by "reached <name>", so only a name re-anchors place; compare MapWM
   with an index model at matched depth; readout = accuracy after teleports. That tests whether a context-free cumsum can
   use landmarks at all (prediction: it cannot without a reset gate -- a falsifiable architectural limit, also relevant to
   the Jericho failure). CPU gate + one pilot seed per arm (~2 GPU-h) before any batch. Otherwise drop the line; a CPU-only
   readout of `rel_named` vs `rel_uninf` on the existing pilot checkpoints is the honest final note.

---

## D. TinyStories

- "Discourse clock" is a fair loose description only after the corrections in B8: it is a coherent token clock plus
  extra ticks at punctuation and boundaries, with one learned cancellation (`mr`/`mrs` against `.`). Alternatives:
  frequency (r +0.25..+0.33 between log frequency and step size; class shares are frequency shares); tokenizer
  artefacts (`mr` + `.`, split quotes); multi-story windows (the `<eos>` jump isolates stories within a 512-word window).
  A clean test of "discourse" vs "punctuation/frequency": regress per-token step on log frequency, part of speech and
  position-in-sentence, then see whether speech verbs and clause links stay above the fit. CPU, minutes.
- The content-word substitution cost is a lesion of a trained model, not a benefit; RoPE ties MapWM, so content
  dependence buys nothing measurable on this corpus (n=3). The report's "the variable rate is used, but buys nothing" is
  the right reading; "about 20x the MapWM - RoPE difference" should be dropped.
- "Motion words do not move it" should become "motion verbs step 0.62-0.67x the average token and are 1.5% of tokens";
  "spatial 0%" is a classification artefact and should be removed (report line 489, results line 27).
- Do not scale this up without a hypothesis that natural text has return-to-coordinate structure (00_PLAN T7 already
  argues it rarely does).

---

## E. Ranked recommendations

**STOP**
1. TW_LANDMARK as designed (C). Write a closing note; at most the teleport gate on CPU.
2. The gain-phase / brain-scorecard line (GAIN_GRAIN, GAIN_PHASE follow-ons, any GainScalar language arm): effects at
   ceiling or ~1%, no novelty over PoPE.
3. TW_AMBIG and further scripted-grammar one-layer batches: the index baseline is at the floor by construction, the
   grammar is yours, and every result so far is either predicted at power 1.00 or a one-layer artefact.
4. Promoting secondaries into CLAUDE.md rows and the report. Cap audit rounds for batches whose verdict cannot change the
   paper (three audits for a 1% effect is process for its own sake).

**FIX NOW (no GPU)**
- `TW_STATECHANGE_RESULTS.md:11`, `CLAUDE.md:72`: "picked up" -> "took / dropped". Add the verb-agnostic floor (check 1)
  to the C reading; reword "state bound to place" as "location + recency at the place; take semantics shown on T2take".
- `MAPPOPE_PAIR_RESULTS.md:47`: CORRECTED block (no rescue where rank 2 fails). Drop the "3% of the total" ceiling bound
  from the file and CLAUDE.md:79.
- `SCORE_RANK_RESULTS.md:35-37, 51-54`, CLAUDE.md:79, report line 372: "repairs short-gap revisits" is a secondary
  (Holm ~0.11); "periodic code" is not separated from long gaps.
- `GAIN_PHASE_RESULTS.md:64`, CLAUDE.md:81, report line 559: "two defects" -> "one defect (the leak); the score fix had no
  measurable effect here". D1 rotary: SOLVED is mostly the leak.
- `TINYSTORIES_PILOT_RESULTS.md:24-30`, report lines 242 / 489 / 495: `mr` cancels `.`; spatial share is an artefact;
  shares are frequency shares.
- Shared report: re-read end to end (rule 28) before the next republish.

**KEEP / DO NEXT (highest value per GPU-hour)**
1. **Rank escape test + 1D ring rank 1 vs 2** (`00_PLAN.md` steps 3-4): ~3-4 GPU-h total. These directly test whether
   the rank result is a search pathology (escape) or a basin chosen early, and whether the lemma T1 is causal. They are
   the cheapest experiments that can strengthen or kill the one novel claim.
2. **A trimmed baseline table at matched length** (torus T=1024: RoPE 2L/4L, an LSTM, an unfactorised full-rank angle
   map, MapWM r2/r4; n=4-6): ~6-8 GPU-h. Any reviewer of the rank paper will ask for it first.
3. **RANK_NOWRAP at 8 seeds** only if 1-2 leave the rank story standing (~13-15 h): it is the test that separates
   "periodic code" from "long gaps", which the current prose already assumes.
Then write the rank paper (rank + sign + depth substitution, with SCORE_RANK as a control paragraph). Total ~10-12
GPU-h before writing, versus ~54 h wall spent this week.

---

## Checks run for this review (CPU, read-only; from /home/prashr with PYTHONPATH=/home/prashr)

Check 1 (verb-agnostic location floor on the TW_STATECHANGE eval stream):
```python
from collections import defaultdict
from mapformer import tw_statechange_readouts as R
from mapformer.train_tw_statechange import make_env
W = R.walks(make_env(R.HELDOUT)); acc = {k: defaultdict(lambda: [0, 0]) for k in ("any", "noaside")}
for w in W:
    t = w["tok"].numpy()
    ev = [(s["pos"], s["cell"], int(t[s["pos"]]), "saw") for s in w["info"]]
    ev += [((a + 3) if k in ("take", "drop") else e - 2, c, int(t[(a + 3) if k in ("take", "drop") else e - 2]), k)
           for k, a, e, c in w["clauses"]]
    ev.sort()
    for s in w["info"]:
        if s["stratum"] == "first": continue
        prior = [x for x in ev if x[0] < s["pos"] and x[1] == s["cell"]]
        for rule in acc:
            p = prior if rule == "any" else [x for x in prior if x[3] != "aside"]
            g = p[-1][2] if p else None
            for st in (s["stratum"], "all"): acc[rule][st][0] += g == s["answer"]; acc[rule][st][1] += 1
```
Output: any: T1 0.852, T2drop 0.841, T2take 0.000, T3 0.841, all 0.762; noaside: T1 1.000, T2drop 1.000, T2take 0.000,
T3 1.000, all 0.897 (n: T1 4052, T2drop 498, T2take 555, T3 270).

Check 2 (TinyStories step directions and class shares): loads `runs/tinystories/p0/Vanilla_s{0,1,2}.best.pt` with
`train_tinystories.build`, step table `(action_to_lie(token_emb.weight) * omega)` as in `tinystories_probe.py`,
frequencies from the first 50M training tokens as the probe does; reports cos(step mr, step '.'), class frequency share
vs step share with the full SPATIAL list, and r(log freq, step norm). Output in the B8 table.
