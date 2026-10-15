# Mock review of Draft 3 (`docs/paper/DRAFT3_workshop.md`), 2026-10-15

This is a mock workshop review followed by a verification pass against the repository. No existing file was edited, and
no GPU was used. Sources checked: every file in `DRAFT3_changes.md`'s source table, plus `RING_BUDGET`, `WRAP_COPE`,
`TEXTWORLD` (its AUDIT and CORRECTED blocks), `SIGN_MATCHED_PREREG.md`, `docs/audits/2026-10-12/ring_dial_audit.md` (M1),
`stats_core.classify_run`, `docs/paper/figs/*_out.txt`, `FIGS_NOTES.md`, the appendix checks, Liu et al.'s Fig. 9 table
(`liu2023.txt:1720-1751`) and the three main-text figures, which I viewed.

---

## Part 1. Review

**Venue assumed.** A NeurIPS/ICLR workshop on positional encoding, sequence models or the geometry of representations.

### Summary

The paper asks whether the sign of a content-dependent rotary phase step matters. Grazzi et al. and Mamba-3 already show
that rotations by non-negative angles represent Z_m and parity, so in the rotary slot a non-negative step loses no
representational power per channel. The authors argue that sign still matters for **learning** maps where opposite moves
must cancel. The setting is 1-layer MapFormer-style models on a 64x64 torus and a 32-cell ring, at matched train/test
length, with pre-registered batches of n = 8.

1. **Sign cost.** A |.|-parameterised step solves 0/8 on the torus (accuracy -0.177) and 1-3/8 on the ring, against
   8/8 for the signed step. Softplus does no better.
2. **Existence.** An exact non-negative solution, built at a trained scale, is held under training on the ring and
   re-found on the torus. From scratch the same class solves 0/8, so the failure is one of search.
3. **Gauge lemma.** A per-channel reflection makes each channel's direction a gauge. |.| therefore removes only the
   relative signs of tokens that step in the same channel, and for an inverse pair it removes exactly the through-zero
   cancelling branch.
4. **Ring mechanism.** Signed models escape the |.| start through zero by a collective move of the latents (1/8 when
   the latents are frozen, against 7/8). |.| models learn 8/8 from a start whose angle sums exceed pi. That start does
   not help on the torus.
5. **Published methods.** Faithful CARoPE and CoPE fail these maps. CoPE also fails on a line that never wraps.
6. **Depth.** One signed path layer beats 1-4 index layers on the torus.

### Strengths

- **S1. It separates representability from learnability properly.** The rule-14 design is the paper's best idea. The
  authors construct an exact non-negative solution at a trained weight scale, freeze it, continue training it, and
  contrast that with training from scratch. They also explain why the naive construction misleads: it is
  ill-conditioned (latent cosine -0.99) and Adam loses it within ~10 epochs. Many papers would have stopped there and
  reported "fragile".
- **S2. The statistics are unusually careful for a workshop.** Batches are pre-registered with n = 8 fresh seeds per arm
  and in-batch controls. Fisher and permutation p-values are given with CIs. Results below threshold are reported as
  "unmeasured". The sign cost is replicated across 3 ring batches and 3 torus batches, with and without observation
  steps.
- **S3. The interventions are causal, not only probes.** The start x constraint 2x2 changes only the initial move angles.
  The frozen-latent test (T2) has its own control, F2: freezing the latents costs nothing from the default start.
- **S4. Faithful baselines with a sharp control.** The CoPE line control (no wrap, still fails) removes the obvious
  account of CoPE's failure. Proving that CARoPE cannot cancel (P6), while CoPE can represent ring cancellation,
  usefully corrects MapFormer's own claim that CoPE "cannot represent an action and its inverse".
- **S5. The theory is checked numerically, bitwise where possible.** Each statement in the appendix is labelled proved,
  conjecture or account, and the run-level mechanism is explicitly called an account (Remark R).
- **S6. The negative is reported.** The torus failure of the start lever is a registered negative, and the paper states
  it in the abstract instead of hiding it.

### Weaknesses (most severe first)

**W1 (major): the title and abstract claim more generality than the evidence supports.** The title asserts a property of
"rotary position". The evidence covers:
- one layer, 2 heads, d = 128;
- a rank-4 latent shared across heads (itself a deviation from MapFormer);
- one optimiser recipe per task (AdamW with warm-up and cosine decay, one learning rate);
- two synthetic navigation tasks.

Learnability claims depend far more on the recipe than representability claims do, yet the paper varies no learning
rate, optimiser or weight decay (for training from scratch) and no depth. The only budget test is 3x on the ring. In the
headline torus batch, 3 of 8 |.| runs were still DESCENDING at 900 epochs (`SIGN_MATCHED_RESULTS.md` caveat). The
abstract's closing budget caveat is honest, but the title reads as a general law.

**W2 (major): the mechanism is a ring-only account, and parts of the ring data fit it loosely.**
- *The start lever does not transfer.* The past-pi start does not rescue |.| on the torus (0/8), and it drags the
  signed model from 8/8 to 2/8. The draft explains this as "a random large-angle code is far from any map for either
  sign". That is an untested account, but it is written as fact.
- *The rescued models do not use the wrap branch much.* AO_high solves 8/8, but only 23% of its channels cancel at the
  end (SO_high: 34%). The per-channel gradient flow in Remark R captures ~28% of channels started from U(1.7, 4.5). That
  is close to 23%, which suggests the |.| rescue may be per-channel capture of a minority of channels, not a collective
  move to "the wrap branch". The draft says the flow "reproduces neither" result; for the |.| side that looks
  questionable.
- *The |.| start is not uniformly below pi.* Table 2 calls the |default| start "small", but per-channel sums at the |.|
  start run from 0.01 to 6.58 rad, with angles up to 5.5 rad (`FIGS_NOTES.md`). Many channels therefore already sit past
  pi, and P3 alone does not explain why they are not captured.
- *A competing account is not discussed.* 44% of AO_absmag's channels end with |a(+1)| <= 0.05, i.e. their steps
  collapse to zero. That points to P4's sticky wall (the step settles at zero) rather than to "not reaching the wrap",
  and the draft never raises it.
- *The start-size confound is only partly controlled.* The large start also moves 13-22x faster per Adam step, and the
  sub-pi control matches only ~5x.
- *T2 covers only the signed side.* It is the strongest mechanistic piece, but it says nothing about why |.| fails.

"We identify the reason" (Sec. 1) therefore over-claims. "We identify a route the signed model uses and the |.| model
lacks" is supported.

**W3 (major-moderate): the theory is thin, and the paper frames the wrong variable.**
- *Theorem 1 is elementary.* It is the identity J R(phi) J = R(-phi) applied per channel. The corollaries are
  bookkeeping, P3 is plane geometry, and P4 is known (as the draft says). Calling Theorem 1 "the exact core" invites a
  "trivial" verdict; it reads better as a lemma that organises the experiments.
- *"Non-negative" describes the parameterisation, not the function class.* In the rotary slot the step
  Delta' = Delta mod (2 pi / omega) is non-negative, has the same function as the signed step, and has the same gradient
  almost everywhere. A non-negative step can therefore be exactly as learnable as a signed one. What the paper actually
  measures is the cost of parameterisations with a barrier or reflection at zero (|.|, softplus, sigmoid rates), which
  cut the circle at zero (P3). Saying this outright would sharpen the claim and pre-empt the obvious objection.

**W4 (moderate): the representability text in Sec. 2 is muddled.**
- Sec. 2 says that when observations also step, exact representation "is not guaranteed", so only the
  observation-step-free arms carry the claim. But representing a *task* solution does not require reproducing a given
  signed table that has observation steps. The full |.| arms can set observation steps to zero, which is P1(a) with W_in
  annihilating the observation embeddings; the appendix states that this holds generically when V <= d_model.
- The ring_dial audit (M1) constructed exactly this in the real modules, and CLAUDE.md's sign row cites it. All |.| arms
  can represent the task, and the caveat weakens the headline torus contrast for no reason.
- Corollary (i) as written in the main text ("realises a signed angle table exactly iff every row is sign-coherent")
  contradicts P1(a), which gives exact mod-2-pi reproduction of opposite-sign rows. It needs the qualifier "as real
  numbers, without wrapping, with the same latent".

**W5 (moderate): on the ring the evidence is mostly a loss threshold.**
- Ring accuracy gaps are -0.03 (full models) and -0.08 to -0.11 (observation-step-free). SOLVED is "final-5% training
  loss < 0.05", and its sensitivity to that threshold is not shown.
- The post hoc strata paragraph helps, but it is post hoc, and its wrap-specific part disagrees between batches.
- The torus (-0.177, -0.104, -0.120) carries the accuracy evidence. The ring carries the mechanism.

**W6 (moderate): Secs. 4.6 and 4.7 dilute the paper.**
- *4.6 is about path vs index position, not sign.* It uses n = 4, and Liu et al. are quoted selectively. The draft cites
  C_4 (median below 90% at 1-15 layers), but in the same table C_5, C_7 and C_8 have medians >= 99% from 3 layers
  (`liu2023.txt:1722-1730`). On larger cyclic groups, Liu's data point the other way.
- *4.7 reports two unmeasured contrasts.* Both sections cost ~600 words that the mechanism needs.
- *The long-gap and wrap-only index accuracies (0.37-0.53) sit at or below the 0.507 blank floor* (Fig. 3). The draft
  does not say so.

**W7 (minor): the torus may confound sign with rank slack.**
- The signed cancelling code needs latent rank 1 per inverse pair (z(-x) = -z(+x)): rank 2 on the torus.
- The wrapped |.| code needs the sum column 2 pi k / omega in addition: rank 2 on the ring and at least 3 on the torus.
- With r = 4, the |.| class has less slack. The project's own results show that rank slack decides torus learnability
  (signed per-head r = 3 solves 6/8, r = 4 solves 8/8).
- The ring, with slack for both classes, still shows the cost, so this cannot be the whole story. On the torus it is an
  uncontrolled factor.

**W8 (minor): baselines and coverage are narrow.**
- *No recurrent or SSM model.* A Mamba-3-style complex SSM with a softplus Delta vs a signed theta would be the natural
  cross-architecture check.
- *No LRU-style exp(.) parameterisation.*
- *CARoPE and CoPE are tested only at 1 layer*, outside the language-model regime they were designed for. The claim is
  scoped to "these maps", which is fair.
- *Softplus has an initialisation confound.* `SIGN_MATCHED_PREREG.md`: "init confound, as in the original". The draft
  reports softplus -0.218 without this caveat.

**W9 (minor): presentation.**
- Terms are used before they are defined: |.|, AO / observation-step-free, MapWM, MapEM and Theta appear in Sec. 2;
  "wrap-only" and "small-beta potential" are never defined in the main text.
- The symbol S means the step table in Corollary (i) and the angle sum in P3.
- Fig. 2 panel titles use internal arm names (SO_absmag, AO_mid, AO_high).
- The draft contains internal provenance and process language (Part 2.4).

**W10 (minor, but checkable by a reviewer with the code): the protocol over-claims.** Sec. 3 says every contrast ran on
"n = 8 fresh seeds" and was "code-checked independently before launch". Neither holds for every number in the paper
(Part 2.2).

### Questions to the authors

1. Does the default-start sign cost survive (a) learning rates 3e-4 and 3e-3, (b) no weight decay from scratch, and (c)
   2-4 layers? A single robustness cell would address W1.
2. Do you agree the claim is about parameterisations with a barrier at zero rather than about sign? Delta mod (2 pi /
   omega) is non-negative and function-identical to signed. How do exp(.) (LRU) or sigmoid-times-scale steps behave?
3. AO_high solves with 23% cancelling channels. What do the other 77% do? Does Remark R's per-channel capture fraction
   (~28%) predict the past-pi rescue channel by channel?
4. In AO_absmag, 44% of channels collapse to |a| <= 0.05. Is the |.| failure the sticky wall (P4) rather than the wrap
   being out of reach (P3)? Separating these two from the existing checkpoints would cost only CPU time.
5. Corollary (ii) and (i)/(iii) make a falsifiable prediction: no sign cost when the stepping tokens of a channel can be
   sign-coherent. Examples are a single stepping token, or generators {+1, +2} with no inverse pair. Has it been tested?
6. On the torus, does |.| at r = 8 (more slack for the wrapped code) close part of the gap?
7. How do the SOLVED counts change at loss thresholds of 0.02 and 0.1, or under an accuracy criterion?
8. Why was the torus continuation run at lr 1e-4 and the ring at 1e-3? At lr 1e-3 (descriptive) the torus non-negative
   construction re-solves only 6/8 (acc 0.988). Should the abstract's "re-finds (torus)" be qualified?
9. Was a structured large start tried on the torus (e.g. each pair's sum near 2 pi), rather than a random one?

### Score

**6 / 10 (marginally above the acceptance threshold for a workshop). Confidence 4 / 5.**

The empirical core is careful and the representability/learnability separation is a useful contribution: the
matched-length sign cost, existence at a trained scale, and the frozen-latent test. The score is held back by:
- the generality claimed in the title;
- a mechanism that is ring-only and fits its own data loosely on the |.| side;
- a theoretical "core" that is a one-line identity;
- two sections that do not serve the thesis.

At the main conference this would be a 4-5.

### What would raise the score

1. **To 7, with text changes only.**
   - Scope the title to |.|/softplus-style parameterisations in 1-layer models on navigation maps.
   - Reframe the claim as "barrier at zero" rather than "sign".
   - Demote Theorem 1 to a lemma.
   - Fix the representability paragraph.
   - Report the past-pi solutions' 23-34% cancelling share and the |.| start's sum range.
   - Move 4.6 and 4.7 to the appendix.
2. **To 7-8, with one experiment.** A recipe and depth robustness cell (E1), or a confirmed T1 prediction (E2). A
   theory that predicts where the cost vanishes, confirmed, would make Theorem 1 earn its place.

---

## Part 2. Verification

### 2.1 Numbers (main text and captions)

**Sec. 4.1 and Fig. 1**

| # | claim (section) | draft | source value (file) | status |
|---|---|---|---|---|
| 1 | torus \|.\| - signed (4.1, abstract) | -0.177; 0.821 vs 0.998; perm p 0.0002 | -0.177; 0.821 / 0.998; 0.0002 (`SIGN_MATCHED_RESULTS.md`) | OK |
| 2 | torus SOLVED | 0/8 vs 8/8 | 0/8 vs 8/8 | OK (3/8 \|.\| still DESCENDING: not stated in 4.1) |
| 3 | softplus | -0.218, 1/8 | -0.218, 1/8 (Pos_r4, "GRAPE-AP", init confound) | OK; init-confound caveat missing |
| 4 | ring \|.\| SOLVED, p_plus 0.5 / 0.75 | 2/8, 1/8 vs 8/8; Fisher 0.007, 0.0014 | same (`RING_DIAL_RESULTS.md`) | OK |
| 5 | ring p_plus 0.9 | unmeasured, 4/8 vs 8/8 | same | OK, but the abstract's "0-3 of 8" leaves out this 4/8 |
| 6 | 3x budget | \|.\| at 3/8 | 3/8 at 900 vs 3/8 at 300, same seeds (`RING_BUDGET_RESULTS.md`) | OK |
| 7 | ring accuracy gap | -0.03; \|.\| 0.95-0.97 | -0.028 / -0.032 / -0.046; 0.973 / 0.968 / 0.954 | OK (the -0.046 at 0.9 is not mentioned) |
| 8 | ring floor; 1-layer index | 0.554; 0.72-0.73 | 0.554; 0.717 (CANCEL), 0.730 (GENERALITY), both at p_plus 0.5 (0.676 at 0.75) | OK for p_plus 0.5 only |
| 9 | cancelling share, signed | 64-98% (group means) | 0.79 / 0.78 / 0.64 / 0.98 (`fig_wrap_share_out.txt`) | OK (the torus shares are post hoc, not marked in the text) |
| 10 | cancelling share, \|.\| and softplus | 3-11% | 0.06 / 0.03 / 0.06 / 0.11; softplus 0.03 | OK |
| 11 | AO from scratch, ring | 0/8 vs 8/8, two batches | WRAP_EXISTENCE E3 0/8 vs 8/8; MECHANISM M4 0/8 vs 8/8 | OK |
| 12 | AO from scratch, torus | 0/8 vs 8/8, 1/8 vs 8/8 | GENERALITY E3 0/8 vs 8/8; TORUS_START TS4 1/8 vs 8/8 | OK |
| 13 | ring strata | 0.96-0.99 at gap 2-7; 0.87-0.96 at 32-127 | 0.992 / 0.961 / 0.978; 0.957 / 0.881 / 0.873 (`ring_strata_out.txt`) | OK |
| 14 | strata test | 8/8 runs in each AO batch, sign-flip p 0.0078 | WRAP_EXISTENCE AO 8/8, MECHANISM AO 8/8, p 0.0078 (`FIGS_NOTES.md`) | OK |

**Sec. 4.2 (existence)**

| # | claim (section) | draft | source value (file) | status |
|---|---|---|---|---|
| 15 | construction angle error | <= 1.4e-6 rad | ring 1.2e-6, torus 1.4e-6 | OK |
| 16 | ring weight scale | within ~5x | ~5x (`WRAP_EXISTENCE_RESULTS.md`) | OK |
| 17 | naive construction | cos -0.987 to -0.998; lost in ~10 epochs | same | OK |
| 18 | ring continuation | 8/8, never leaves the solved region | 8/8 SOLVED, 0/8 LOST | OK |
| 19 | torus continuation | peak loss 0.11-0.21 vs <= 0.017; 6/8 leave; 8/8 SOLVED, 1.000 | 0.110-0.208 vs 0.004-0.017; 6/8 LOST; 8/8, 1.000 | OK; the descriptive lr 1e-3 pair (AO_bal_hi 6/8, 0.988) is not reported |
| 20 | usage-weighted cancelling share | 99% -> 98% | 0.99 -> 0.98 (ACCOUNTS T1) | OK |

**Secs. 4.3 and 4.4 (mechanism, torus start)**

| # | claim (section) | draft | source value (file) | status |
|---|---|---|---|---|
| 21 | Table 2 | 8/8, 0/8; 7/8, 0/8; --, 2/8; 8/8, 8/8 | SO, AO; SO_absmag, AO_absmag; AO_mid; SO_high, AO_high (`MECHANISM_ANALYSIS.txt`) | OK |
| 22 | M5 | 0/8 vs 7/8, Fisher 0.0014, -0.092 [-0.135, -0.045] | same (perm p 0.0030, not reported) | OK |
| 23 | M2 | 8/8 vs 0/8, +0.109 [+0.085, +0.135], p 0.0002 | same | OK |
| 24 | M2b | 2/8, Fisher 0.007 | 2/8, Fisher 0.0070 | OK |
| 25 | M6 | 8/8 vs 8/8 | same | OK |
| 26 | how cancelling channels cancel | 98-99% opposite-signed, 0% wrapped; 88% / 100% by wrapping | SO 0.99, SO_same 0.99, SO_absmag 0.98; 0.00 wrapped; SO_high 0.88, AO_high 1.00 (`mech_escape_out.txt`) | OK |
| 27 | T2 frozen latents | 1/8 vs 7/8, Fisher 0.010, -0.075 [-0.130, -0.020]; 15% vs 82% | 1/8 vs 7/8, 0.0101, same CI; 0.15 / 0.82 | OK |
| 28 | F2 | "freezing costs nothing (7/8 vs 8/8)" | NO EFFECT DETECTED, 0.991 vs 1.000 | wording: "no effect detected", not "nothing" |
| 29 | mobility confound | 13-22x; ~5x | same | OK |
| 30 | TS1 | 0/8 vs 1/8, +0.005 [-0.060, +0.070] | same (`TORUS_START_ANALYSIS.txt`) | OK |
| 31 | SO4_high; descending runs | 2/8 vs 8/8; 5/8 per large-start arm descending | 2/8; AO4_high 5 DESC, SO4_high 5 DESC | OK |
| 32 | TS4 | 1/8 vs 8/8, Fisher 0.0014 | same | OK |

**Secs. 4.5 to 4.7 (published methods, depth, clock side)**

| # | claim (section) | draft | source value (file) | status |
|---|---|---|---|---|
| 33 | CARoPE | 0.667 vs 0.722 (p 0.040); ring 0.741 vs 0.730 | same (`GENERALITY_RESULTS.md`) | OK |
| 34 | CoPE ring | 0/8, 0.887 vs 1.000, p 0.0002; +0.157 vs index | same | OK. Other batches: 1/8 (0.840, WRAP_COPE zeros init), 0/16 (0.873, ACCOUNTS); pooled 1/32 would be stronger |
| 35 | CoPE line | 1/16 vs 0/16; signed 8/8; index 0.772 | same (ACCOUNTS T4) | OK |
| 36 | depth factorial | 1.000 / 0.719 / 0.860 / 0.864; p 0.029; floors 0.507 / 0.576 | same (`FACTORIAL_RESULTS.md`) | OK |
| 37 | factorial strata | 0.74-0.79 -> 0.90-0.96; long-gap and wrap-only 0.37-0.53 | same | OK; at or below the 0.507 blank floor, not said |
| 38 | navigation in words | 0.969 vs 0.505, p 0.0002, 7/8 vs 0/8 | registered numbers match. CORRECTED block: seeds 0-1 were the pilot (fresh seeds: 6/6 vs 0/6, +0.474, p 0.0022). AUDIT block: path re-scored 0.969 -> 0.993 | **MISMATCH with Sec. 3's "n = 8 fresh seeds"** |
| 39 | Liu et al. C_4 | median below 90% at 1-15 layers | 56.2-85.8 (`liu2023.txt:1722`) | OK; C_5 / C_7 / C_8 >= 99% from 3 layers left out (selective) |
| 40 | clock side | -0.197 (p 0.053, n 16); -0.106 (p 0.19, n 16) | same | OK |
| 41 | Dyck at matched depth (appendix F) | +0.353 / +0.130 / +0.045 / +0.024 | same (`DYCK_MDEPTH_RESULTS.md:61-64`) | OK |

**Sec. 2, theory, figure captions and protocol**

| # | claim (section) | draft | source value (file) | status |
|---|---|---|---|---|
| 42 | Remark R | ~0.5 rad capture; 12% cross zero | same (`APPENDIX_theory.md` Remark R); also 28% of channels captured from U(1.7, 4.5) | OK; the 28% vs AO_high's 23% is not discussed |
| 43 | P3 | S / sqrt 2 vs (2 pi - S) / sqrt 2; coincide iff S >= pi | appendix P3(c), checked to 2e-4 | OK; the draft sentence conditions on S < pi and then on S >= pi |
| 44 | P6 | (0, 4] < 2 pi | (0, 3] move-query, (0, 4] obs-obs | OK |
| 45 | Theorem 1 checks | bitwise on trained checkpoints, 2-layer model, MapEM | `t1_gauge_out.txt` (a)-(f), all 0.00e+00 | OK |
| 46 | P2 frozen probe (appendix E) | wrapped 0.80, signed 0.998 | 0.802 / 0.998 | OK |
| 47 | Fig. 1 caption | 0.79 / 0.78 / 0.64 / 0.98; 0.06 / 0.03 / 0.06 / 0.11; softplus 0.03 | same | OK |
| 48 | Fig. 2 caption | 0.06 -> 0.89 "all through zero"; 0.04; 0.05; 0.07 -> 0.23 | 0.062 -> 0.895 (0.883 opposite-signed + 0.012 same-sign near the origin); AO_absmag 0.042, of which 0.035 is same-sign near the origin (not cancellation); 0.047; 0.072 -> 0.230 | OK; "all" should be "almost all"; the 0.04 is mostly collapsed steps |
| 49 | SOLVED definition (Sec. 3) | final-5% training loss < 0.05 | `stats_core.classify_run` (tail = final 5%, < 0.05) | OK |
| 50 | protocol (Sec. 3) | every contrast "code-checked independently before launch" | `SIGN_MATCHED_PREREG.md` has no amendment or pre-launch check (its audit, 2026-09-30, came after the batch) | **MISMATCH** |
| 51 | abstract | "learn only from a start ... past pi (8/8 vs 0/8)" | AO_mid 2/8, RING_DIAL Abs 2/8 and 1/8, RING_BUDGET 3/8 all learn from other starts | **OVER-CLAIM**: "reliably only from" |
| 52 | abstract | "the codes it excludes are exactly those in which an inverse pair cancels through zero" | Corollary (iii): excluded = rows with opposite-signed entries; through-zero channels are a subset; "exactly" holds only within the cancelling set | **IMPRECISE** |
| 53 | 4.2 | "A non-negative solution exists and is stable in task terms" | ring HELD; torus NOT HELD by the registered rule ("perturbed and re-found") | **OVER-GENERALISED**: ring only |
| 54 | Sec. 2 | observation-stepping arms not guaranteed representable | ring_dial audit M1 constructs exact \|.\| solutions in the real modules with observation steps <= 1.7e-6; P1(a)'s premise holds for V <= d_model | **INCONSISTENT** with the audit and with CLAUDE.md's sign row |

There are no transcription errors in the numbers. Every value matches its source to the stated rounding. The problems are
omissions (rows 2, 3, 19, 37, 39, 42), wording (rows 28, 48, 51, 52, 53) and two protocol statements (rows 38, 50).

### 2.2 Conflicts with the "Do not claim" list and with CORRECTED blocks

**Main text.** The main text is clean against the claim map's "Do not claim" list:
- representability is attributed to Grazzi et al. and Mamba-3;
- there is no drift, weight-decay, fragility or near-wrap-design claim, and no CoPE wrap claim (it is stated as refuted);
- there is no "start, not constraint", no double dissociation and no "replicates Sarrof / Grazzi".

**Conflicts.**
1. **`TEXTWORLD_RESULTS.md` CORRECTED (2026-09-30): seeds 0-1 are the pilot.**
   - *Conflict.* Sec. 4.6 quotes the n = 8 numbers (7/8 vs 0/8, p 0.0002), while Sec. 3 says every contrast ran on fresh
     seeds.
   - *Fix.* Quote the fresh-seed numbers (+0.474, 6/6 vs 0/6, p 0.0022), or add a footnote. The AUDIT block's re-score
     (0.993) belongs beside it, not only in appendix G.
2. **`SIGN_MATCHED_PREREG.md` lists the softplus arm as "init confound, as in the original".**
   - *Conflict.* The draft reports -0.218 without the caveat.
3. **Appendix E (`docs/paper/APPENDIX_theory.md`) is stale and still contains withdrawn claims.** The draft's appendix
   plan describes it as corrected, but the file is not:
   - P2 "Account" (l. 247-250): "The ring drift 87% -> 18% and the torus knock-out are consistent with P2 and P2.1",
     with weight decay as the push. This contradicts "Do not claim": "held model drifts" and "weight decay drives the
     drift" (ACCOUNTS T1).
   - Remark R (l. 386): the collective route is "untested ... ACCOUNTS T2, pending". T2 has since been run and is
     SUPPORTED.
   - P7 (l. 475): "ACCOUNTS T4 ... pending". T4 has been run: the line also fails.
   - Header: "Written ... for draft 2". P1(a) cites "draft Sec. 4.3" for the construction, which is now Sec. 4.2.
   - Summary table row R: "account, not derived" is still correct, but should now say "one component tested (T2)".
4. **Source files not yet bannered (no draft conflict, but a reader following the citations will meet them):**
   - `WRAP_EXISTENCE_RESULTS.md` still states the 87% -> 18% drift as a secondary, with no CORRECTED banner pointing to
     ACCOUNTS T1.
   - `MECHANISM_RESULTS.md` still carries the "Design implication" that TORUS_START withdrew.
   - `SIGN_MATCHED_RESULTS.md` and CLAUDE.md's sign row still say "replication in a new regime", which the lit note asks
     to retire.

   These files are the authors' to update; the draft itself is consistent.
5. **The claim map (l. 93) is stronger than its evidence.** It says CoPE's obstacle "(precision is)". ACCOUNTS shows only
   that the wrap is not the obstacle. The draft correctly calls precision "a plausible account".

### 2.3 Internal inconsistencies

**Contradictions inside the text**
1. Corollary (i) ("exactly iff sign-coherent") contradicts P1(a) (exact mod 2 pi for opposite-sign rows). Add "without
   wrapping, same latent".
2. The abstract and Sec. 1 say "loses no representational power" with no qualifier. Sec. 2 then says representation is
   "not guaranteed" when observations step. The appendix proves only per-channel (Lemma P1.0) and the m <= r case,
   which, read correctly, already covers the full models (W4).
3. The title says "rarely learn one", but the ring's past-pi start learns 8/8.
4. Table 2 and Sec. 4.3 call the |.| start "small". The angles reach 5.5 rad and the sums 0.01-6.58 (`FIGS_NOTES.md`).

**Notation and definitions**
5. The symbol S is the step table (S_{c,x}, Corollary (i)) and also the sum of two angles (P3) within one page.
6. Used before definition, or never defined:
   - |.| (abstract, Sec. 2; defined in Sec. 3);
   - "observation-step-free" (Sec. 2; defined in Sec. 3);
   - MapWM and MapEM (Theorem 1; MapEM never defined);
   - Theta (Corollary (iii); never defined in the main text);
   - "wrap-only" (4.1, 4.6, Fig. 3; never defined);
   - "small-beta potential" (Remark R; never defined);
   - LOST (appendix plan D);
   - "Case II / III" (Table 1 caption; defined only implicitly by Table 1's first column).

**Figures and tables**
7. Fig. 2 panel titles use arm codes (SO_absmag, AO_absmag, AO_mid, AO_high) that the text never uses.
   `FIGS_NOTES.md` still calls this figure "Figure 3", from the old numbering.
8. Table 2 is never cited in the body text. Fig. C1 is cited in 4.1, before Fig. 2. That is acceptable for an appendix
   figure, but say "appendix Fig. C1".
9. Fig. 3: the index bars for long gaps and wrap-only revisits sit at or below the dashed floor. The caption says
   "floor" without naming which one (blank 0.507; the n-gram floor 0.576 is not drawn).

**Protocol and cross-references**
10. Sec. 3's list of n leaves out the recency batches (n = 16), which Sec. 4.7 uses.
11. Appendix plan I says the toy "fails the signed default". In MECHANISM the toy's signed default solved 6/12. "Fails
    often" is accurate.
12. The section references in the contributions list (4.1-4.6, Sec. 2) are all correct. The appendix letters used in the
    text (C, E, F, H, J) all exist in the appendix plan.

### 2.4 Passages that read as accreted (rule 28)

- **Header (l. 3-6).** Internal provenance (advisor plan, batch names, file paths). Delete for any external reader.
- **4.1, second paragraph.** "The |.| ring models are not maps that just miss a loss threshold" reads as a rebuttal to
  an internal reviewer. Fold it into one sentence after the ring numbers, or move it to appendix C.
- **4.2.**
  - The parenthetical "(Continuation learning rates differ, 1e-3 ring and 1e-4 torus, so the two labels are not
    directly comparable.)" is patched in.
  - The last sentence (usage-weighted 99% -> 98%) answers a withdrawn claim the reader has never seen. Either explain
    why it matters ("the held model keeps its cancelling code in the channels attention reads") or drop it.
- **4.3.**
  - "The escape is collective (registered; fresh seeds)" is a later batch appended to the section, with a different
    seed set from Table 2.
  - The closing "Caveat:" sentence is tacked on.
  - The section has three layers: registered contrasts, then a post hoc readout, then a later registered batch. It
    needs one narrative.
- **4.4.** A negative result sits between the mechanism and the baselines. Its final explanatory sentence comes from the
  results file's "What it means" and is untested.
- **4.6, Liu paragraph.** About 170 words of defensive framing, inherited from a lit-check correction. Two sentences
  would do.
- **Remark R.** Titled "the run-level mechanism is an account": a meta-comment about status rather than content. State
  the collective-route hypothesis and point to 4.3.
- **Related work.** "To our knowledge (one session of search and a 40-paper corpus)" is process language. Use "To our
  knowledge" alone.
- **Limitations.** "(~8 h per torus run)" is an internal cost.
- **Abstract.** Nine sentences and seven parenthetical counts. The mechanism sentence packs four results into one. It
  ends on a caveat rather than a takeaway.
- **Appendix plan E.** "not as an explanation of drift (there is none in used channels)" records a correction history
  the reader does not need.

---

## Part 3. What to change

### 3.1 Fixes needing no new experiments (most important first)

1. **Scope the title and abstract.**
   - *Title.* For example: "Non-negative rotary steps can represent a map but rarely learn one: a learnability cost of
     sign-constrained phase parameterisations".
   - *Abstract wording.* Change "learn only from a start past pi" to "learn reliably only from". Change "0-3 of 8" to
     "0-4 of 8". Qualify "exactly those [codes]" ("among cancelling codes").
   - *Scope sentence.* Add one sentence stating 1 layer, d = 128, one recipe per task.
2. **Reframe "sign" as "a barrier at zero in the parameterisation".**
   - Add the one-line remark that Delta mod (2 pi / omega) is non-negative and identical to the signed step, so what is
     tested is |.| / softplus / sigmoid-type parameterisations that cut the circle at zero (P3).
   - This pre-empts the strongest conceptual objection and costs two sentences.
3. **Rewrite the representability paragraph in Sec. 2.**
   - P1(a) with W_in annihilating the observation embeddings covers every |.| arm. This is the ring_dial audit's M1
     construction in the real modules.
   - Keep "a given signed table with stepping observations need not be reproducible" as a side remark.
   - Add "as real numbers, same latent" to Corollary (i).
4. **Demote Theorem 1 to "Lemma 1 (relative-sign gauge)"** and drop "the exact core". Present it as the tool that
   predicts where the cost should vanish (Corollary (ii) and sign-coherent generators), and say plainly that this is
   untested.
5. **Correct the protocol sentence (Sec. 3).**
   - Name the exceptions: the text world's seeds 0-1 were the pilot (quote the fresh-seed 6/6 vs 0/6, p 0.0022);
     SIGN_MATCHED had no pre-launch independent check.
   - Add the recency n = 16.
6. **Report the mechanism's loose ends in 4.3.**
   - Past-pi solutions cancel in only 23% (AO_high) and 34% (SO_high) of channels.
   - The |.| start's per-channel sums span 0.01-6.58.
   - 44% of AO_absmag channels collapse to a zero step (P4).
   - Change "we identify the reason" (Sec. 1) to "we identify a route that the signed step uses and |.| lacks".
   - Label the 4.4 "far from any map" sentence as an account.
7. **Restrict "stable in task terms" (4.2) to the ring.** Mention the descriptive torus lr 1e-3 pair (6/8 re-solve).
8. **Move 4.6 and 4.7 to the appendix**, and drop contribution 6. This frees ~600 words for 4.3.
   - If 4.6 stays, add Liu's C_5 / C_7 / C_8 >= 99% from 3 layers.
   - Also state that the index model's long-gap accuracy is at the blank floor.
9. **Bring appendix E up to date** (Part 2.2, item 3).
   - P2's Account: delete it, or replace it with "the drift was in unused channels (ACCOUNTS T1)".
   - Remark R: T2 has been run and is supported.
   - P7: T4 has been run; the line also fails.
   - Header, and the reference "draft Sec. 4.3" -> 4.2.
10. **Presentation.**
    - Define |.|, AO, MapWM, Theta, "inverse pair" and "wrap-only" before Sec. 2 uses them (or move Sec. 3's model
      paragraph first).
    - Rename the angle sum in P3 (e.g. sigma).
    - Relabel the Fig. 2 panels in words.
    - In the Fig. 2 caption, note the collapsed near-origin channels in AO_absmag.
    - Cite Table 2 in the text.
    - Name the floor in Fig. 3.
11. **Add the softplus initialisation-confound caveat** in 4.1.
12. **Remove internal process language**: the header, "one session of search", "~8 h per torus run" and the history in
    appendix plan E.
13. **Related work.**
    - Cite TEM-t (Whittington et al., ICLR 2022): an action-driven recurrent position code inside a transformer, the
      closest architectural precedent after MapFormer.
    - Cite the Dale's-law learnability work (Cornford et al. 2021; Li et al. 2023: spectrum at initialisation vs the
      sign constraint), which is the same "start vs constraint" question in another setting.
    - Say explicitly that your CoPE construction refutes MapFormer's statement that CoPE "cannot represent an action
      and its inverse" (`papers/txt/mapformer.txt:585-586, 1701-1703`), at least on the ring.
    - Verify the author lists the lit note marks "from memory", and the Khavari and GridPE claims (abstract-level only).
14. **Re-analyses on CPU from stored artefacts** (no training):
    - SOLVED counts at loss thresholds 0.02 / 0.1 and under an accuracy criterion, from the stored loss curves;
    - per-channel capture vs the end cancelling share for AO_high, to test Remark R's 28% against the observed 23%;
    - the fraction of zero-collapsed channels across all |.| arms (sticky wall vs unreachable wrap).

    Label all of these post hoc.

### 3.2 Experiments that would most change a reviewer's mind

All follow the batch workflow: pre-registration, an independent check, pilots on outside seeds, signed and |.| retrained
in one batch. Costs assume 2 GPUs at 2 jobs per GPU (4 concurrent), ring 300 epochs ~10-25 min per run, torus 900 epochs
~30-45 min per run under load.

**E1. Recipe and depth robustness of the default-start sign cost (addresses W1, the main objection).**
- *Design.* Ring, p_plus 0.5, full models (Signed_r4 vs Abs_r4), n = 8, three cells: lr 3e-3; lr 3e-4 with wd 0; and
  2 layers at the default recipe.
- *Cost.* 3 x 2 x 8 = 48 runs x ~15-20 min ~ 12-16 GPU-h, ~3-4 h wall.
- *Torus option.* Add one 2-layer torus pair: 16 runs x ~40 min ~ 11 GPU-h, ~3 h wall.
- *What it shows.* If the sign cost holds in all cells, "within our budgets" becomes "across the recipes and depths
  tried", and the title can carry more weight.

**E2. Test the gauge lemma's prediction (addresses W3, and makes Theorem 1 predictive).**
- *Design.* Ring, observation-step-free models, signed vs |.|, n = 8, two generator sets: {+1, -1} (inverse pair; a cost
  is predicted) and {+1, +2} (sign-coherent solutions exist; no cost predicted by Corollary (i)). A single stepping
  token (p_plus = 1.0) is a cheaper third cell for Corollary (ii).
- *Cost.* 2 x 2 x 8 = 32 runs ~ 8-13 GPU-h, ~2-3 h wall; +16 runs (~1 h wall) for the p_plus 1.0 cell.
- *What it shows.* Confirming "no cost without inverse pairs" turns the lemma from a re-description into a prediction.
  This is the single most persuasive addition for a geometry or representation workshop.

**E3. Is it |.|, or any barrier-at-zero parameterisation, and is the torus confounded by rank slack (addresses W3,
W7, W8)?**
- *Design.* Ring AO with softplus and exp(.) (LRU-style) steps, beside in-batch signed and |.| controls, n = 8.
- *Cost.* 4 x 8 = 32 runs ~ 8-13 GPU-h, ~2-3 h wall.
- *Torus option.* Add AbsAO_r8 vs AbsAO_r4 vs SignedAO_r4: 24 runs x ~40 min ~ 16 GPU-h, ~4-5 h wall. This tests
  whether extra latent rank closes part of the torus gap; it does not change the ring story.

**If only one experiment can be run, run E2.** It is cheap (ring only, ~2-3 h wall) and directly tests the paper's
theoretical claim. E1 is the one a sceptical reviewer will ask for first.
