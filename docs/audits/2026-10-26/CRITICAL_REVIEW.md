# Critical review, 2026-10-26: the Memory Maze proposal and the last week as a whole

Adversarial read, no experiment run, nothing else edited. Read: CLAUDE.md; `docs/IMAGE_TASKS_SURVEY_2026-10-25.md`;
`docs/HYAK_NOTES.md`; `IMAGE_TORUS_RESULTS.md` (+ prereg headings, both independent checks); `G3_SSM_RESULTS.md`;
`G3_SSM_A2_RESULTS.md`; `FACTORIAL_N8_RESULTS.md`; `TORUS_LRSWEEP_RESULTS.md`; `RANKRECIPE_RESULTS.md`; `RANKWD_RESULTS.md`
(header); `SIGNSEP_RESULTS.md`; `docs/paper/MAIN_DRAFT2.md` (all), `MAIN_APPENDIX2.md` (section heads, H4, A3, B),
`MAIN_DRAFT2_notes.md` (head); `DRAFT6_workshop.md`; `BIG_PICTURE_2026-10-24.md`; `CLAIM_MAP_clock_map.md`;
`.claude-memory/project_state.md`; `feedback_follow_on_drift.md`; `feedback_gpu_sharing_and_cost.md`; `KNOB_SWEEP_n8.md`
and `ALIASING_CONTROLLED.md` (heads). Line numbers refer to the files as of commit 9db4899. Memory Maze facts below that
are not in the repo are from my own knowledge of Pasukonis et al. 2022 and are marked (verify): the paper is not in
`papers/` (the only "memory maze" hit in `papers/txt/` is an unrelated Chen et al. 2023 title).

Severity: BLOCKER = do not launch / do not submit until fixed; MAJOR = a reviewer will raise it and it changes a sentence
the paper leans on; MINOR = wording, housekeeping, or a caveat that is already in the file but not where it is read.

---

## Part A. The Memory Maze proposal (survey Sec. 3b, lines 100-160; risks Sec. 4)

### Summary verdict

As designed, the primary contrast (P - R1 on long-gap retrieval) answers a question the project has already answered
three times (torus tokens, torus words, torus pixels): "does attention with the true position written into its rotary
phase beat attention that must integrate displacements in its residual stream?" The pose-difference step makes the
path model's phase ground-truth position by construction (A-B1), the retrieval target as written penalises exactly the
prediction a map makes (A-B2), and 15x15 is at or below the project's own map-extent threshold (A-B3). The one arm that
asks something new, E (raw egocentric actions), is listed as a secondary "expected to fail" (line 124). Invert the
design: make learned allocentric recoding the question, keep given odometry as the ceiling, and gate on the dataset
before any GPU hour. The survey's own risk list (Sec. 4, items 1, 2, 5, 7) already names each of these; the fix is to
let them decide the design rather than sit beside it.

### BLOCKER

**A-B1. The pose-difference step is the answer, not a variable (lines 115-124).** With d_t = agent_pos[t] -
agent_pos[t-1] fed through a linear W_in, theta_t = omega W_out W_in (pos_t - pos_0): the phase is the world-frame
position up to a learnable linear map and a scale. Nothing has to be discovered about which tokens step or by how much;
the cumsum is exact for a linear input (the survey says so, risk 6). The index arms get the same d_t as tokens and must
build the sum in the residual stream, which `FACTORIAL_N8_RESULTS.md` already shows four index layers do not do within 900
epochs. The expected outcome is therefore known, and a reviewer will read "one path layer beats index RoPE on Memory
Maze" as "a transformer given its GPS position as a rotary code beats one that is not". The IMAGE_TORUS ActOnly result
has the same character (Part B, B-M4); Memory Maze would be the third instance.
*What is non-trivial and testable here:* (i) **learned allocentric recoding**: feed the egocentric action and the
heading as one joint token (6 actions x 8 heading bins = 48 tokens; or action token plus a heading token with a step map
that can multiply them) so the step map must learn the table d(action, heading) = speed x (cos h, sin h) -- the
project's rotation negative (`KNOB_SWEEP_n8.md` rotate +0.050; allocentric recoding +0.485) says this is where path
integration actually fails, and nobody has shown a learned step map recovering it; (ii) **inferred self-motion**: a
small odometry head on (f_{t-1}, f_t) producing d_t, trained with pose-difference supervision (or end to end), with the
phase driven by the predicted d_t; the registered contrast is "learned odometry step vs given odometry step vs index".
(ii) is what "navigation from pixels" means to a reader; (i) is cheaper and already decisive against the project's own
negative. **Fix:** make (i) the primary (P_joint vs P_given vs R1/R4), keep given odometry (the current P) as the
ceiling arm, and say in the registration that P alone is a sanity check. If only the current P is affordable, write the
claim as "a given-odometry phase" and do not call it navigation from pixels.

**A-B2. The retrieval target penalises the map's own prediction and is solvable locally (lines 132-136).** Three
problems. (a) InfoNCE over all frames of the episode, true frame ranked first: at a pose revisit the frame f_s from the
earlier visit to the same pose bin is in the candidate set and is nearly identical to f_t (static maze, same cell, same
heading bin). A model that retrieves what it saw at this place -- the behaviour the map is supposed to produce --
predicts an embedding closest to f_s and is scored wrong when f_s outranks f_t. The score is then about sub-cell pose and
lighting noise, not the map. (b) Local continuation: consecutive 64x64 first-person frames at ~0.3 cell/step differ by a
small ego-motion; a two-frame model (f_{t-1}, d_t) -> f_t predicts the next frame without any memory. The survey lists
this floor for the glimpse torus (line 95) but not for Memory Maze (floors (i)-(viii), lines 141-149, have copy-current,
retrace and view-recognition only). It will be the strongest floor and may sit near the ceiling. (c) Perception
confound: with a frozen encoder every arm and every floor shares the embedding, which is good, but retrieval accuracy
then measures encoder discriminability as much as position; the pose-oracle ceiling (vi) controls the top, not the
floors. **Fix:** score the top-ranked frame by whether its *pose bin* matches the target's, or exclude earlier
same-pose-bin frames from the negatives (they are positives); add the trained two-frame continuation floor and the
"predict before the frame is seen" readout as the registered primary; report every floor under the same readout
(survey risk 8 says so already). The published layout probe is a weaker primary than this fixed retrieval target (a
one-layer MapWM has no per-step state summarising the maze; a 128-d token decoding a 13x13 layout is implausible and
the comparison to RSSM probes is between unlike objects) -- keep it as the comparability secondary.

**A-B3. 15x15 is at or below the project's map-extent threshold (lines 142-143, 169-170).** The citable threshold is
-0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells (CLAUDE.md, `ALIASING_CONTROLLED.md`, `VISITS_TEST.md`). A
15x15 labmaze has a wall border and corridors; the open interior is of the order of 80-110 cells (verify on the data),
i.e. at the 128-cell point where the effect was +0.015. Eight heading bins multiply pose states but not the number of
places a map must hold. The survey picks 15x15 "unless the gate says otherwise" but does not state the prediction its
own threshold makes: little or no path effect. Add to this that the official 15x15 episodes are 1000 steps of a policy
whose revisit kinds are unknown (risk 7), and the most likely outcome of the primary is a null that the project's own
data predicted (`feedback_borrowed_benchmarks.md` records two of these). **Fix:** the CPU gate must report open cells,
pose-revisit rate, gap distribution, retrace share and continuation-floor accuracy *before* registration, with a written
stop rule ("if open cells < 128 or non-local revisits < 10% of scored steps, do not run"). If the gate fails, the
options are larger custom mazes (memory-maze builds on labmaze; a custom size is plausible but needs checking) or
native-length 15x15 generation (~7.4 node-h per 1000 episodes, line 114), both with the heading-joint design of A-B1.

### MAJOR

**A-M1. Frozen encoder (lines 126-130).** A time-contrastive encoder trained on train-split frames learns temporal
adjacency, which in a static maze is pose adjacency: the embedding then identifies place and heading from one frame,
and floor (iv) (view recognition) rises toward the ceiling, leaving no room for a map. An autoencoder leaks less but
discriminates less. Whatever is chosen, pretraining must exclude the eval mazes, and the encoder probe (viii) must be
measured on eval-maze frames with the same pose bins as the readout. Decide the encoder by the gate, not by preference.

**A-M2. Baselines (lines 151-155).** With d_t given, an LSTM/GRU integrates by construction (one hidden unit per axis);
if it solves the task, "one path layer" is not a result about attention at all, and if it does not, that is the
interesting number. Also add the "GPS-PE" control: index attention given cumsum(d_t) as an absolute sinusoidal position
code -- if it matches P, the result is the position code, not path integration. R4 (4 index layers) is fine as the
depth control. Published RSSM/S5/TBPTT numbers are on other objectives; cite as context only (survey risk 9 is right).

**A-M3. Cost and infrastructure (lines 157-160; `HYAK_NOTES.md` lines 22-39, 47).** 200 GB of download onto filesystems
at 96-99% with a 30-day purge is the real risk, not the 32-64 A40-h. The plan needs: a quota check and a named target
before download; shard-wise download of the 15x15 set only (or a 1-5k episode subset for the gate and pilots); embedding
precompute written as a few large arrays (GPFS small-file penalty); per-epoch resumable runs for 8-9 h windows (the
notes say so); and a statement that cluster batches (torch 2.13) are never pooled with workstation batches (rule 12,
already in the notes). State the user's cost rule (`feedback_gpu_sharing_and_cost.md`) in the registration and offer a
trimmed variant (3 arms x 8 seeds) with its power.

**A-M4. Dataset facts are README-level (lines 108-112).** The paper (2210.13383) is not in the corpus and was not read
first-hand; the policy ("scripted waypoint policy under action noise chosen to form loops"), the per-step displacement,
collision handling and whether `agent_pos` is continuous are all (verify). Continuous pose means a "revisit" to a cell
bin is a different sub-cell view, which feeds A-B2(a). Rule 11 (gate must call the task code) applies: open one npz and
print these statistics.

**A-M5. Rule 14 before launch.** Construct the solution in the class: a linear W from d_t to angles with omega such
that 15 cells never alias (omega_max = 2 pi / Delta_max with Delta_max = maze size, as in the torus), install it frozen
and trainable at the trained weight scale, and check the pose-reconstruction R^2 of the cumsum on real trajectories
(risk 6). Cheap and it makes the given-odometry ceiling arm an actual ceiling.

### MINOR

- Survey Sec. 1 says `IMAGE_TORUS_RESULTS.md` is the first image result; `IMAGE_TORUS_RESULTS.md` line 196 says the survey
  "does not exist". Stale cross-reference; fix the results file.
- Lines 80-84 (prior-art risk): GTA / PRoPE / GridPE take given poses -- so would the proposed P. Say that P is in their
  class and that only the learned-recoding or inferred-odometry arms leave it.
- "(web survey)" rows (lines 64, 66, 69) are second-hand; before any novelty sentence, read Deng et al. 2307.02064 and
  Po et al. 2505.20171 first-hand (CLAUDE.md: grep the corpus, read first-hand).
- Line 136 "InfoNCE as the loss" with same-episode negatives: say the batch construction (per episode? per step?) --
  the floors depend on it.

### What result would be publishable; what would be a waste

Publishable: on a published first-person 3D benchmark, with a measured continuation floor and a measured
view-recognition floor, one path layer whose step map *learns* heading-conditioned displacement (A-B1(i)) retrieves
long-gap revisit views that neither floor nor four index layers reach, and an LSTM on the same inputs does not match it
-- that is a new statement (learned allocentric recoding in the rotary slot) with an external benchmark behind it. Also
publishable, as a negative with a mechanism: the learned-recoding step map fails and the given-odometry ceiling does
not, measured on the same data.

A waste: the current P vs R1 at 15x15 x 1000 steps. Either the gate shows the revisits are local and the map is below
threshold (a null the project predicted) or P wins, which is the torus result with a GPS phase and will be read as
such. SIGNROUTE-style "decisive" language should not be attached to it in advance.

### What I would do next for Memory Maze (ordered, with costs)

1. CPU gate on a 1-5k-episode 15x15 subset: open cells, revisit kinds, retrace share, continuation and view-recognition
   floors with a quick encoder, encoder-pose probe. ~1 day of agent time, 0 GPU-h (encoder: <1 A40-h). Write the stop
   rule first.
2. If the gate passes: register the inverted design (P_joint primary, P_given ceiling, R1, R4, LSTM/GRU, GPS-PE; n = 8),
   with the pose-bin-match readout and the continuation floor; independent check; pilots on outside seeds. 6 arms x 8 =
   48 runs x 1-2 A40-h = 48-96 A40-h, 6-12 ckpt array windows.
3. If the gate fails: decide between custom larger mazes with native-length episodes (generation ~10 node-h per 1000
   episodes) and dropping the benchmark in favour of the glimpse torus (line 90-98), which controls the same two risks
   at IMAGE_TORUS cost.

---

## Part B. The last week and the paper as a whole

### Summary verdict

The registered batches of the week are clean as batches: each has a prereg, an independent check, a results file that
re-derives its numbers and states its budget scope. The problems are upstream and downstream of them. Upstream:
`BIG_PICTURE_2026-10-24.md` Sec. 5 set a stopping point ("after (1)-(4) the main paper is [submittable]; everything after
that is revision material") and ranked the LSTM/GRU baseline above anything with images; IMAGE_TORUS (48 runs + 2
pilots, 11.4 h) and the Memory Maze line are not in that triage and the LSTM baseline is still unrun. Downstream: the
main draft's framing of the sign axis has not caught up with exp's no-cost, its main text omits the one registered test
that disconfirmed the route account's prediction (GAUGE, which draft 6 reports in its main text), and the pixel section
says things the results file explicitly withholds. None of this touches the four carrying results of BIG_PICTURE Sec. 1;
all of it is what a TMLR reviewer will find in an afternoon.

### BLOCKER

**B-B1. Draft 2 main text drops the registered disconfirmation of the route account (GAUGE).** `MAIN_DRAFT2.md` Sec. 4
Prop. 4(b) (lines 251-254) and Sec. 7.3 "Why the non-negative step does not find it" (lines 636-646) present the
through-zero route as the account of the |.| failure. The one registered test of Lemma 1's learnability prediction,
GAUGE (+1/+2: no zero crossing needed, |.| still 2/8, UNMEASURED vs signed 5/8; CLAUDE.md sign row; draft 6 Sec. 4.5
lines 196-215 and abstract line 14-16: "that does not explain the failure"), appears in draft 2 only in App. H4 (line
1503) and nowhere in Secs. 4, 7.3, 9 or 10 (grep "+1/+2", "GAUGE", "sign-coherent": no hit in the main text). The main
text thus states an account that the companion draft says is not confirmed, and hides the test. **Fix:** one paragraph
in Sec. 7.3 after "Why the non-negative step does not find it", with Table 2 of draft 6, and the sentence draft 6
already has: losing the through-zero route does not account for the difficulty; Lemma 1 is about representation. Add
to Limitations.

**B-B2. The sign framing is no longer what the evidence says; draft 2's title-level message must change as draft 6's
did.** After `G3_SSM_A2_RESULTS.md` the registered statement is "|.| and softplus increments pay; exp does not in the
SSM at 1e-3 (6/8 vs 7/8), reaching the map by wrapping on channels its initialisation puts past pi; exp pays in attention
(1/8 at both rates, SIGNROUTE)". Combine with MECHANISM (past-pi start rescues |.| 8/8 on the ring), TORUS_START (the
same start does not rescue on the torus), SIGNSEP (a route to the wrap does not carry |.| past pi; barrier half-read)
and GAUGE: the variable that predicts success is *where the initial angle sums sit relative to pi together with a
barrier at zero*, not the sign of the increment. Draft 6 renamed itself ("|s| or softplus(s)"); draft 2 still says "the
sign of the phase step" (Sec. 7.3 title, line 586), "sign gap" (abstract line 22), "a non-negative step has neither
route from a small start" (lines 18-19, 57-58), and the abstract (line 25) writes "the LRU-style exp step, which starts
many channels past pi, does not pay" -- a causal reading the results file declines ("the batch does not separate start,
multiplicative step and mixer", `G3_SSM_A2_RESULTS.md` lines 106-110). The practical consequence is worse: the
parameterisations people use (exp in the LRU, softplus in Mamba's Delta) have never been run on the headline task in
attention -- softplus was (SIGN_MATCHED, -0.218, 1/8), **exp never** (no exp arm in SIGN_MATCHED, GENERALITY or
TORUS_START). **Fix:** (a) rename Sec. 7.3 and the abstract's sentence to "barrier parameterisations of the step (|.|,
softplus)"; drop "which starts many channels past pi" from the abstract or write "consistent with its past-pi start,
not separated"; (b) run exp (and softplus at the default) on the observation-step-free T = 1024 torus at lr 1e-3 and
3e-3 beside signed and |.| anchors from TORUS_LRSWEEP's seeds -- 16-24 runs, ~7-10 h on two 4090s; this is the one sign
batch the paper still needs, because "sign of the increment" currently rests on a parameterisation nobody uses.

### MAJOR

**B-M1. Follow-on drift and the stopping point.** `BIG_PICTURE_2026-10-24.md` Sec. 5: needed = G-4, G-3 A2, rank-2
recipe search, FACTORIAL +4 seeds; nice-to-have = LSTM/GRU (if TMLR); "then stop". The week ran all four needed items
(good), then IMAGE_TORUS (not in the triage; 11.4 h + pilots), and now scopes Memory Maze (weeks of cluster work), while
the LSTM baseline (8 h, ranked, answers attack A5 directly) is "optional" in `project_state.md` line 19.
`feedback_follow_on_drift.md`: "for each proposed GPU job state which plan item / claim it serves". The image line
serves contribution 5's second half, which BIG_PICTURE Sec. 2 rates "medium-weak as a contribution, good as a
demonstration" for words; pixels inherit that rating. **Fix:** run the LSTM/GRU torus baseline before any Memory Maze
GPU hour; put IMAGE_TORUS and Memory Maze in a revised triage with the claim each serves written down.

**B-M2. IMAGE_TORUS: the registered primary was redefined after the pilot, and the main text does not say what the
original primary read.** `IMAGE_TORUS_PREREG.md` line 101 registered P1 = I_MapWM - I_RoPE (the paper's default model);
the stop rule fired on pilot 1 and Amendment 2 made ActOnly the path arm (results file lines 38, 177-181, disclosed as
"partly pilot-informed"). The original P1' reads **UNMEASURED, -0.033** (image MapWM 0.668 vs image RoPE 0.701; results
line 38): on pixels, the model every other section of the paper uses does *not* beat index RoPE. Draft 2 Sec. 8 (lines
878-881) says "(registered; partly pilot-informed: the path arm was seen on one pilot seed)" and never states that the
registered primary was swapped or that the default model fails. Table 8 row 3 carries the number without the word.
**Fix:** one sentence in Sec. 8 and in Limitations line 1024-1027: "The registered primary compared the default MapWM
with index RoPE and read unmeasured (0.668 vs 0.701); after a pilot stop rule the path arm was changed to the
action-driven step. The headline is for that arm."

**B-M3. Draft 2 line 892 contradicts the results file on "leak".** Draft: "This is the what-to-where rule above at full
strength." `IMAGE_TORUS_RESULTS.md` lines 80-81: "No seed of the batch is NOISY, so by Amendment 3 the word 'leak' does not
apply to this batch"; the P2 end states are COLLAPSED (encoder outputs a constant; the phase becomes a clock ticking at
every observation slot). The leak (LEAK_RESULTS) is identity-dependent offsets of order 1%; this is an optimisation
collapse of order 0.3. They share "a step that reads the observation" and nothing else. **Fix:** replace line 892 with
"a different failure of the same design choice: not a leak (no per-instance offsets) but a collapse of the encoder and
the phase into a clock". Also say that G (gain 1 vs 30.37) is a total effect on step input and residual stream together
(results line 217), so "images move the phase by 0.001-0.006 of a move" (line 890) is the measured consequence, not the
isolated lever (ActOnly at gain 30 shows the content scale is harmless, so the attribution to the step is sound; say
that chain in one clause).

**B-M4. What "path wins in pixels" adds beyond the token result, and how to say it.** In I_ActOnly the step map never
sees an image; its phase is bit-for-bit the token model's position code at the same seed and walk. What is new is (i)
the CNN is trained only through the revisit loss and ends above the standalone classifier (probe 0.985 vs 0.980, S3),
(ii) index RoPE learns a worse encoder (0.965) and (iii) content attention matches fresh instances at the retrieved
place. The honest headline is "a pixel front end does not break path integration, and the revisit loss alone trains
perception"; "Maps from pixels" (Sec. 8 heading line 869; abstract line 27-29; contribution 5) oversells it, and a
reviewer will say the position code was never asked to do anything new. The result's actual weight in the paper is the
same as the egocentric-actions paragraph (Sec. 9 line 973): a scope statement. **Fix:** retitle the subsection
"Pixel observations", cut it to one paragraph plus Table 8, move the P2 collapse to Limitations/design guidance, and
drop pixels from the abstract's last sentence or reduce it to "and survives a pixel front end".

**B-M5. RANKRECIPE's "search favours rank 2" is not what the grid did.** `RANKRECIPE_RESULTS.md` line 14 and
`MAIN_DRAFT2.md` lines 516-517: "The search favours rank 2 (it picks rank 2's best of three while rank 4 gets one)". The
two alternative recipes are lr 3e-3 and wd 0.2. `TORUS_LRSWEEP_RESULTS.md` (same task family, observation-step-free rank
4, same 900 epochs): the *signed rank-4* arm solves **0/8 at 3e-3** (all descending). So one of the two "rescue" recipes
is one at which the control class itself fails within budget; the other raises decay, the direction RANKWD shows rank 4
needs less of, not more. The search therefore did not offer rank 2 a plausible rescue; it offered two recipes that hurt
everything. The claim that survives is weaker than "survives a three-recipe search": "rank 2 fails at the default and at
two harsher recipes". **Fix:** reword lines 14 and 69-73 of the results file and draft 2 lines 513-518; in Limitations
(line 1014-1017) add "the alternatives were not recipes at which rank 4 is known to learn". Do not run more recipes
unless a reviewer asks; the 1800-epoch result already covers "slower".

**B-M6. "Survives" and "needed" in headings where runs are still descending (rule 3-4, CLAUDE.md "survives wording where
budget-scoped").** `RANKRECIPE_RESULTS.md` title line 11 "SURVIVES A THREE-RECIPE SEARCH" with 12/24 rank-2 runs
DESCENDING; `RANKWD_RESULTS.md` "DECAY NEEDED" with 5/7 wd-0 failures still descending; `TORUS_LRSWEEP_RESULTS.md` "COST
AT BEST RATES" with 5/8 |.| descending (the qualifier is in the next clause, good). Each file states the budget scope in
its body; the headings and the CLAUDE.md rows ("needs", "survives") do not. **Fix:** "within 900 epochs" in every such
heading and CLAUDE.md row; draft 2 line 517 "Rank 4 in turn needs weight decay" -> "within 900 epochs rank 4 learned only
with weight decay".

**B-M7. Related work omissions BIG_PICTURE marked as must.** `BIG_PICTURE_2026-10-24.md` Sec. 2: Larson 2026 (2609.12259,
rank law: "must be cited and reconciled"), Howe 2026 (2609.18966), Heo et al. 2026 (2603.05573). None is in
`MAIN_DRAFT2.md` or `MAIN_APPENDIX2.md` (grep). The rank claim is in direct tension with Larson's result; a reviewer who
knows it will ask why. **Fix:** three sentences in Sec. 9 Related work and one in Sec. 7.2 (why a sparse revisit loss
and a shared readout might need D + 1 where dense state supervision needs D). CPU only.

**B-M8. Length and the theory load.** Main text 18,000 words (`wc -w`, including captions and references), eleven
figures, eight propositions, one conjecture, three remarks, a 2,400-line appendix. BIG_PICTURE A4 said "cut, do not add";
since then Secs. 7.3 and 8 grew by the SSM, SIGNSEP, G-4, A2 and pixel paragraphs. The SIGNSEP paragraph (lines 688-703)
reports an UNMEASURED 2x2 in 16 lines; the exp paragraph (lines 665-676) in 12. TMLR has no page limit but reviewers do.
**Fix:** rule 28 end-to-end read with a target of ~10-11k words: SIGNSEP to three sentences; exp to four; Props. 5-6 and
Conjecture 1 to a remark; Sec. 7.4 and 8's pixel part to half; move Table 4 and Fig. 5 to the appendix.

**B-M9. Missing baselines for the one-layer claim.** No recurrent baseline anywhere (Limitations line 1036 admits it for
the SSM only). An LSTM/GRU on the torus at T = 1024 with the same action/observation stream either solves the map (then
"one layer of attention" is the wrong unit and the result is "a map needs an integrator") or does not (then the
one-layer result sharpens). Either way it is the first thing a TMLR reviewer asks for (BIG_PICTURE A5). ~8 h.

### MINOR

- Abstract line 25 and Sec. 7.3 line 666-671: the exp account ("starts many channels past pi") is stated as if separated;
  the results file (A2 lines 106-110) and Limitations (line 1033-1034) say it is not. Align the abstract with Limitations.
- `G3_SSM_A2_RESULTS.md` lines 58-62: at 1e-3 the signed SSM anchor solves 7/8 with partial codes (T = 512 0.764 +/- 0.273)
  and exp's T = 512 equals it (0.753). "NO SIGN COST" is read at a rate where the signed class itself generalises poorly;
  say so beside the exp sentence in draft 2 (line 667) and draft 6 (line 137).
- "State-space model" for an unnormalised linear-attention kernel with a per-head RMS norm and no decay (G3 lines 11-17):
  the results file is explicit; draft 2 line 657-660 should keep "diagonal complex SSM without decay or gate" in the
  first mention, not only in Limitations.
- `SIGNSEP_RESULTS.md` is UNMEASURED; the registered half-reading "BARRIER BLOCKS" is promoted to draft 2's Sec. 9 line
  915 ("giving |.| a collective route to the wrap with the barrier left in place does not (0/8)") without the word
  "half-reading" and without F's 4/8. Add both or cut the clause.
- `FACTORIAL_N8_RESULTS.md`: clean. Draft 2 Fig. 1 caption line 103 "n = 4 shown" with the fresh seeds in App. F7: fine;
  consider plotting both batches.
- `TORUS_LRSWEEP_RESULTS.md` lines 108-109: the signed prediction (7-8/8 at 3e-3) was missed; the file says so. Draft 2
  line 605-606 states it; good. The decay-coupling caveat is in Limitations; good.
- CLAUDE.md sign row (line 75) is a single ~4,000-character cell. It is no longer readable as a table row; split into a
  short row plus a pointer to `CLAIM_MAP_clock_map.md`, which already carries the per-batch "allowed / not" lines.
- `.claude-memory/project_state.md` has three "## Running" sections (lines 11, 72, 83) and twenty "Last results" blocks;
  its own header says "live state only". Collapse to: running, next, last three results, open decisions. The rest is
  `docs/LOG.md` material.
- `IMAGE_TORUS_RESULTS.md` line 196: "the survey the prereg names ... does not exist" -- it does now; update.
- `RANKRECIPE_RESULTS.md` has no "Scope" or "Checks" section matching the other files (no experiment_audit line, no
  host); add for uniformity.
- Draft 2 line 1005-1008 says the torus exchange rate is "a lower bound measured at n = 4 and replicated on four fresh
  seeds (n = 8 over two batches, pooled only descriptively)" -- consistent with FACTORIAL_N8; good. Keep "more than four"
  and never "about five".

### What a hostile TMLR reviewer says first

1. "Everything is one or two layers on synthetic walks of a few thousand tokens, and MapFormer already showed path
   integration helps navigation. Your contribution is that their benchmark was 93% solvable by a local rule -- a useful
   correction, one figure." (Answer: the floors, the strata, the matched-depth exchange rate; keep contribution 1 and 2
   first.)
2. "Every negative is 'not within budget', and your own tables show runs still descending in almost every failing cell."
   (Answer exists: 1800 epochs, 3x ring budget, held solutions; put the STALLED/DESCENDING counts in the main tables, not
   the prose.)
3. "There is no recurrent baseline. An RNN given actions integrates trivially." (B-M9.)
4. "The 'sign' result has an exception your own theory predicts (exp starts past pi), so the real variable is where the
   initialisation puts the angles. Call it that." (B-B2.)
5. "The theory came after the data; its one registered mechanism prediction was PARTIAL, the route-vs-barrier 2x2 was
   unmeasured, and the main text omits the +1/+2 test that disconfirmed Lemma 1's prediction." (B-B1; cut Sec. 4 to
   Props. 1-4 and 7.)
6. "Rank D + 1 depends on weight decay 0.05 on all parameters, your latent is shared where MapFormer's is per head, and the
   recipe search gave rank 2 two recipes at which your own rank-4 model fails." (B-M5, B-M6.)
7. "The pixel section changed its primary after a pilot and the default model does not beat RoPE on pixels." (B-M2.)

### Missing before submission (ordered, with costs)

1. CPU, 1-2 days: B-B1 (GAUGE paragraph), B-B2(a) (rename and abstract), B-M2, B-M3, B-M4, B-M5, B-M6 wording, B-M7
   citations; then the rule-28 end-to-end read with the word target (B-M8).
2. GPU ~8 h: LSTM/GRU torus baseline at T = 1024 (B-M9), registered, with the retrace floor and strata.
3. GPU ~7-10 h: exp and softplus on the observation-step-free T = 1024 torus at 1e-3 and 3e-3 beside signed and |.|
   anchors (B-B2(b)); this decides whether the paper's sign sentence may mention the LRU/Mamba parameterisations at all.
4. Optional, if a reviewer asks: a 3x-budget torus batch for rank 2 and |.| (~1 GPU-day; BIG_PICTURE A1).
5. Memory Maze: the CPU gate of Part A only; no GPU until it passes and the design is inverted (A-B1).

Total new GPU before submission: ~15-18 h on the workstation, all tied to named reviewer attacks; the Memory Maze line is
a separate paper's question and should be costed as one.
