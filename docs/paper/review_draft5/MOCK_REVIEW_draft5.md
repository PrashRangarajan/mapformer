# Mock review of Draft 5 (`docs/paper/DRAFT5_workshop.md`), 2026-10-16

Venue assumed: an ICLR 2027 workshop of the "Scientific Methods for Understanding Deep Learning" kind (Sci4DL: 4 pages
plus unlimited references and appendix, double-blind; its call asks for "controlled experimental settings" and for work
that "validate[s] or falsif[ies] hypotheses", and mildly prefers real-world data).

Read: the draft, `APPENDIX_draft5.md` (A-D, F-K), `APPENDIX_theory_v2.md` (E), the four figures (viewed), their
`*_out.txt` files and scripts, `ROBUST_RESULTS.md`, `MECHANISM_RESULTS.md` (spot checks), `DRAFT5_changes.md`,
`VENUES_2026-10-15.md`, and the LRU entry of `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`. I formed the review before
reading `MOCK_REVIEW_draft3.md`; Part 2 compares the two. No existing file was edited, nothing was committed, and no GPU
was used. One CPU check: the Fig. K1 floor line, recomputed from `FACTORIAL_*_STRATA.json` (Part 1, W9).

---

## Part 1. Review

### Summary

The paper studies content-dependent rotary position codes, where each token adds a learned step to a rotary phase,
theta_t = omega * sum_u Delta_u. In the rotary slot a non-negative step *value* loses nothing: Delta mod 2 pi / omega is
the same model. The authors argue that a non-negative *parameterisation* (|s| or softplus(s)) still costs something.
They call it a "barrier at zero": gradient steps cannot carry a step through zero. The models are one-layer, 2-head,
d = 128 MapFormer-WM variants. The step is a rank-4 latent shared across heads. The tasks are a 64x64 torus walk and a
32-cell ring walk, trained and tested at the same length, with pre-registered batches of n = 8 seeds per arm.

The paper claims five things:
1. **The cost.** Barrier steps solve the map on 0-4/8 seeds per batch against 8/8 for signed steps. The main numbers
   are torus |.| 0.821 vs 0.998, 0/8 vs 8/8, and ring 2/8 and 1/8 vs 8/8.
2. **Existence.** An exact |.| solution, built from a solved signed model, is held under further training on the ring.
   On the torus it is knocked out and re-found. From scratch the class solves 0-1/8.
3. **Lemma 1 (a relative-sign gauge).** |.| removes only the relative signs of the tokens that step in a channel. For
   an inverse pair it therefore removes the through-zero route to cancellation, not the function.
4. **A ring mechanism.** The signed model escapes a same-direction start through zero, and does so collectively
   through the move latents (freezing them gives 1/8 vs 7/8). |.| learns 8/8 from a start whose angle sums exceed pi.
   That lever fails on the torus.
5. **Published methods.** Faithful CARoPE (shown unable to represent cancellation) and CoPE (able to represent it)
   both fail these maps.

The authors also report a registered test of Lemma 1's prediction (moves +1/+2: unmeasured, and not confirmed). They
also report a registered recipe/scale check. In that check the cost holds without weight decay, at lr/3, and in a
2-layer d 256 torus model. It becomes unmeasured at lr x3.

### Strengths

- **S1. The study is designed to separate representation from learning, and the design is executed well.** The
  authors construct the |.| solution at a near-trained weight scale. They show it is exact (angle error <= 1.4e-6 rad;
  frozen accuracy equal to the source on 8/8) and then continue training it. They also explain why the naive
  construction is a bad test: it is ill-conditioned (latent cosine -0.99, |W_out| up to 16,320). This is the most
  convincing part of the paper, and it fits a "scientific methods" venue well.
- **S2. The core effect replicates across many independent batches.** The direction of the sign cost is the same in
  about 13 registered cells drawn from eight independent batches:
  - torus, with and without observation steps (three batches);
  - ring at three p_plus values;
  - four observation-step-free ring batches (WRAP_EXISTENCE, MECHANISM, GAUGE, ROBUST default);
  - wd 0, lr/3, and 2-layer d 256.
  Fisher and permutation p-values are given with CIs, and "unmeasured" is used when a contrast falls below the
  registered threshold.
- **S3. The causal interventions have controls.** The start x parameterisation table changes only the initial move
  angles; every other weight is identical at a seed. The frozen-latent test has its own null control: from the default
  start, freezing has no detected effect (7/8 vs 8/8).
- **S4. Failed predictions are reported in the main text, labelled as failed.** These are the +1/+2 test (Sec. 4.5),
  the torus start lever (Sec. 4.4) and the lr x3 cell (Sec. 4.1). The same holds for the one-channel flow's poor
  channel-level fit (kappa +0.09) and the wall check reading MIXED. For a Sci4DL-type venue this is a real asset.
- **S5. The floors are measured, not assumed.** These include the post hoc "retrace" rule, which makes the torus
  result sharper: the |.| mean (0.821) is below a map-free local rule (0.843). The SOLVED criterion is also stress-tested
  at three thresholds and under an accuracy criterion (Table C10).
- **S6. The baselines are faithful, with a sharp control and a correction to the literature.** P6 proves that CARoPE
  cannot cancel. A construction shows that CoPE can represent ring cancellation, contrary to MapFormer's statement. The
  line control (no wrap) shows the wrap is not CoPE's obstacle.

### Weaknesses (ranked, most severe first)

**W1 (major): the learning rate is a confound, and the evidence points towards "slower / needs larger steps" at least
as much as "rarely learns".**
- *What the data show.* In ROBUST (Sec. 4.1, "Recipe and scale"; Table C13) the barrier arm's SOLVED count rises with
  the learning rate on fresh seeds at matched seeds:

  | recipe | \|.\| SOLVED | \|.\| accuracy |
  |---|---|---|
  | lr/3 | 0/8 | 0.841 |
  | lr 1e-3 | 1/8 | 0.911 |
  | lr x3 | 4/8 | 0.987 |

  The signed arm is at 8/8 at the two higher rates.
- *The comparison is not neutral.* lr 1e-3 is the signed model's default recipe. Comparing two parameterisations at
  one arm's learning rate is not a neutral comparison: a parameterisation change is an optimiser change, and the
  project's own rule 15 says so.
- *At scale the cost shrinks.* The 2-layer d 256 torus gap is a quarter of the 1-layer gap. 6/8 |.| runs were still
  descending at 900 epochs, and SOLVED does not survive a 0.1 threshold (5/8 vs 8/8).
- *In the headline torus batch, 3/8 |.| runs were still descending.*
- *The evidence for a block rather than a delay is one cell.* That cell is RING_BUDGET: 3/8 at 900 epochs vs 3/8 at
  300, with no in-batch signed arm.
- *Consequence.* The title's "rarely learn a map" and Sec. 4.1's heading "Barrier parameterisations do not learn the
  map" are claims about one learning rate per task. The abstract mentions the shrinkage at lr x3 but presents the
  d 256 result as "the cost holds", without the quarter-size gap or the 6/8 still descending. A reviewer who reads
  Table C13 will ask the obvious question: what happens at lr 1e-2? If the gap closes there, the paper's message
  becomes "barrier parameterisations need larger steps". That is still interesting, but it is a different paper.
- *Requested.* A per-arm learning-rate sweep, reporting best-of-lr per arm. Without it the robustness claim is
  one-sided.

**W2 (major): the identified route does not explain the failure, by the paper's own registered test, but the
framing still leans on it.**
- *What the abstract and Sec. 1 say.* The abstract: "A per-channel identity shows the barrier removes, for an inverse
  pair, the through-zero route to cancellation, not the function". Sec. 1: "For learning maps, where opposite moves
  must cancel, this matters within our budgets. We identify a route to the map that the barrier removes".
- *What Sec. 4.5 found.* With moves +1/+2 no channel needs to cross zero, and the |.| start (all positive) is already
  sign-coherent. Yet |.| solves only 2/8, no better than on +1/-1 (1/8; Fisher p 1.00). The point estimates of the gap
  are the same (-0.069 vs -0.064). The +1/+2 laps stratum shows the same deficit (0.893 vs 0.884).
- *Two further findings against the route as the cause.*
  - The |.| runs that do solve do so "without the wrapped code (cancelling share 0.04-0.13)" (Sec. 4.1, last
    sentence). So within the |.| class, the cancelling code does not separate success from failure.
  - The ring rescue from past pi uses only 23% cancelling channels, and the per-channel flow does not predict which
    (kappa +0.09).
- *What remains supported.* The barrier removes the through-zero route; that is mathematics. Signed models use that
  route collectively; that is ACCOUNTS T2, a single batch. The claim that losing the route is why |.| fails is not
  supported.
- *What the text does.* Sec. 4.5 admits this ("the barrier account does not explain |.|'s difficulty on +1/+2") and
  Sec. 4.3 says "why |.| fails from a small start ... is not separated". But the title, abstract and contributions 3-4
  still read as if the barrier/route account were the explanation.
- *An alternative is not discussed.* |.| may simply be a worse-conditioned parameterisation for this optimiser. It
  folds the gradient (P4), it collapses 24-70% of move steps to the wall (Table C12), and its learning-rate
  sensitivity (W1) differs from the signed arm's. Such an account would predict a cost on +1/+2 too, which is what
  was observed.

**W3 (major for this venue): length.**
- *The numbers.* The main text is ~3,500 words (`DRAFT5_changes.md`), plus 2 tables, 2 figures and 21 references.
  At Sci4DL's 4-page limit this is roughly 6-7 pages. `VENUES_2026-10-15.md` itself says a 4-page track "needs a cut
  to Secs. 4.1-4.3 plus Lemma 1".
- *Why the cut is not cosmetic.* As written the paper does not fit the venue it targets. The cut decides which of the
  five contributions survive. Sec. 4.6 (CoPE/CARoPE) and the Howard paragraph of Sec. 3 do not carry the thesis
  (W5, W8).

**W4 (moderate-major): Fig. 1, the main-text evidence that "barrier codes do not cancel", uses an unweighted share
that the authors themselves found misleading.**
- *Where it appears.* Sec. 4.1, "The codes show what is missing (Fig. 1, post hoc): signed models cancel ... in 64-98%
  of channels ..., |.| and softplus models in 3-11%". The figure plots the unweighted share among channels with
  |wrap(a(+1))| > 0.05 (appendix A4).
- *Why it matters.* In ACCOUNTS T1 (appendix D6) the same unweighted share fell 0.87 -> 0.18 while the
  usage-weighted share stayed 0.99 -> 0.98. The authors withdrew the unweighted reading there, because unused
  channels dominate it.
- *The figure cannot show what the text says.* Fig. 1 therefore does not show that |.| models lack a cancelling code
  in the channels attention reads. The fact that solved |.| runs also sit at 0.03-0.08 (filled red markers) suggests
  the unweighted share is not the right readout.
- *A smaller inaccuracy.* The caption says "share of rotary channels", but the denominator excludes collapsed
  channels.

**W5 (moderate): the practical relevance is thin, and the "published methods" section does not test the thesis.**
- *The tested parameterisation is not used in practice.* No published content-dependent rotary method uses |.| or
  softplus on the step. Selective RoPE and Mamba-3 are already signed: Mamba-3's angle is softplus(Delta) x a signed
  theta, as appendix E A0.3 notes. So the advice "allow a sign" (Sec. 5) is already standard practice.
- *The two monotone methods tested fail for reasons the paper itself separates from the barrier.*
  - CARoPE cannot represent cancellation (P6), so its failure is representational and says nothing about the
    learnability claim.
  - CoPE is not rotary, and its obstacle is "unidentified" (Sec. 4.6).
- *Contribution 5 is therefore orthogonal to contributions 1-4.* It is a useful negative about two baselines, but it
  occupies main-text space without supporting the title.
- *The method nearest to the thesis is untested.* The LRU's exp(.)-parameterised positive phase is a barrier
  parameterisation in the paper's own sense. The authors cite LRU but do not test exp(.) (Limitations).

**W6 (moderate): the abstract over-reaches in three places.**
- *"barrier steps solve the map on 0-4 of 8 seeds per batch against 8 of 8 for signed steps".* This is false for
  ROBUST lr/3, a one-layer ring batch: the signed arm is at 6/8 there (Table C13).
- *"The cost holds ... at a third of the learning rate".* It holds on SOLVED (0/8 vs 6/8, Fisher p 0.007). On
  accuracy the permutation p is 0.089 and the CI is [-0.215, +0.010], so the accuracy contrast is not detected.
- *"The cost holds ... in a 2-layer d 256 torus model".* This omits the quarter-size gap, the 6/8 still descending,
  and the SOLVED contrast failing at loss 0.1. The body states all three; the abstract should too.
- *"Exact |.| solutions exist at a trained scale".* The built weights are up to ~5x the signed source's (ring max
  |W_out| 0.31-0.38 -> 1.07-1.70). "Near-trained scale (within 5x)" is accurate; "a trained scale" is generous.

**W7 (moderate): on the ring, the "does not learn the map" evidence is a training-loss threshold on models that are
97% accurate.**
- *The numbers.* Ring |.| models reach 0.95-0.97, above the map-free retrace floor of 0.860. The accuracy gap is
  -0.03 (Sec. 4.1). By the post hoc threshold check, the p_plus 0.5 contrast does not survive a 0.1 loss threshold.
- *What the strata show.* They (C11, Fig. C1) show a graded deficit with gap, i.e. an imperfect map, not an absent
  one.
- *The claim needs narrowing.* "Rarely learn a map" over-reads this: the |.| ring models learned most of the map and
  did not reach near-zero loss. The torus carries the accuracy evidence, as Sec. 4.1 concedes. In ROBUST's d 256 torus
  cell, every |.| run is above retrace too (lowest 0.910).
- *Suggested scope.* "Rarely reach an exact map within budget" is the claim the data support.

**W8 (moderate): the theory is mostly bookkeeping, and its one prediction failed.**
- *Lemma 1 is the identity J R(phi) J = R(-phi)* (the paper says "elementary"), and P3 is plane geometry.
- *P4 is known.* It is the Woodworth / Amid-Warmuth sign barrier, as cited.
- *The prediction failed to confirm.* The one falsifiable prediction (corollary ii, no cost without an inverse pair)
  was tested and did not confirm. That leaves Lemma 1 as "a statement about representation only" (Sec. 4.5).
- *The novelty claim rests on a conjunction.* Sec. 5: "neither the relative-sign gauge nor the loss of the
  through-zero route for inverse pairs has been stated; we claim their conjunction". That is a thin claim for one
  elementary identity and a corollary about distances.
- *The Howard paragraph is not used.* Sec. 3, "Three integrators", is not used by any result (the clock side is
  unmeasured, appendix J), and in a 4-page version it should go.

**W9 (minor-moderate): figure and caption accuracy.**
- *Fig. 2 shading.* The shading is labelled "outside the non-negative quadrant (the barrier side, closed to |.|)".
  omega is free in sign, so the |.| class also contains the all-negative quadrant (appendix E P3 proof: "With
  omega < 0 the class also contains the negative quadrant"). Trained omega does cross zero (appendix D1). Only the
  two mixed-sign quadrants are closed. No |.| point in the figure lies in quadrant III (raw ranges >= 0), so the
  figure's data are fine, but the shading misstates the class.
- *Fig. K1 floor.* The caption says the dashed line is the best-constant floor "(overall 0.507)". The script plots
  the mean of the per-stratum floors. Recomputed from `FACTORIAL_*_STRATA.json`, that is 0.513; the batch's own 'all'
  floor is 0.509; 0.507 is appendix B's value from another stream.
- *The headline result is not plotted.* There is no per-seed accuracy or training-loss-curve figure for signed vs
  |.|. Loss curves are the natural evidence for "stalled vs slow" (W1), and the data exist.
- *What is correct.* Every figure the text cites exists. Fig. 1 and Fig. 2 numbers match `fig_wrap_share_out.txt` and
  `fig_mechanism_out.txt` exactly (0.79 / 0.78 / 0.64 / 0.98; 0.06 / 0.03 / 0.06 / 0.11; softplus 0.03; 0.06 -> 0.89;
  0.04 of which 0.035 is collapsed; 0.05; 0.07 -> 0.23).

**W10 (minor-moderate): uncontrolled architectural factors on the torus.**
- *Rank slack.* The |.| wrapped code needs extra latent slack on the torus (m = 4 = r, exactly at P1(a)'s limit), and
  per-head rank is known to decide torus learnability for signed steps (Limitations). An r = 8 |.| arm would control
  this.
- *Shared latent.* The shared-across-heads latent is a deviation from MapFormer. Whether the sign cost survives
  per-head latents is untested.

**W11 (minor): the protocol statements are broader than the protocol.**
- *"Every contrast was pre-registered with its decision rule" (Sec. 2).* Several main-text contrasts are post hoc
  (floors, threshold sensitivity, Fig. 1, the laps comparison). The sentence should say "every verdict".
- *The independent code-verification agent.* It is described as "an automated reviewer". Reviewers will ask what it
  is (an LLM agent from the same pipeline?) and in what sense it is independent.
- *The pre-registrations are not verifiable by an outside reader.* An anonymised repository with prereg commit
  timestamps before results is needed for "pre-registered" to carry weight at review.
- *Appendix A5.* Its batch table omits ROBUST, which Sec. 4.1 and Table C13 use (it had an independent check:
  `docs/audits/2026-10-16/robust_audit.md`).
- *Code release and compute* are not stated.

**W12 (minor): softplus is in the headline but carries two caveats that a short CPU construction could remove or
confirm.**
- *Why the caveat looks avoidable.* Softplus "cannot set a step to zero" (Sec. 3), but representability by wrapping
  looks easy:
  - With W_in annihilating the observation embeddings, every observation's step is softplus(0) = ln 2 in every
    channel.
  - Choosing omega_c = 2 pi k / ln 2 makes every observation's angle 0 mod 2 pi.
  - The m <= r move columns can then be set through softplus^{-1} of the wrapped target steps.
- *Suggested change.* If this works in the real modules, softplus joins P1(a) and the "untested representability"
  caveat disappears. The initialisation confound remains.

### Clarity

- *Density.* The writing is precise but very dense. Sec. 4.1 is a wall of parenthesised counts, p-values and floors
  in two paragraphs. Contributions 3-4 are hard to parse on first read.
- *Undefined internal vocabulary.* The main text uses "AO" without saying what the letters stand for. "Retrace",
  "SOLVED", "observation-step-free" and the "balanced construction" are all defined, but in a 4-page paper they cost a
  reader a lot.
- *Hedge-heavy sentences.* Several sentences carry three hedges each (e.g. Sec. 4.1, "but the accuracy gap is a
  quarter ... (an out-of-batch comparison), 6/8 |.| runs were still descending ..., and the SOLVED contrast does not
  survive ... (post hoc): at scale part of the cost may be speed"). The honesty is welcome; the reader needs one
  sentence of claim and one sentence of caveat.
- *What is clean.* Lemma 1 and P3 are stated cleanly. Table 1 is the clearest object in the paper.
- *Process lines remain.* The header line ("Delete this line before submission") and a few paths in the reference
  preamble ("`papers/new_2026-10-15/`") are internal and must go.
- *Notation.* Plain-text math (theta_t, omega * sum_{u <= t}) is fine for a draft, but should be typeset.

### Registered vs post hoc, budget scoping, title and abstract (the checks the caller asked for)

**Registered vs post hoc.** This is mostly clear.
- *Labelled correctly:*
  - Fig. 1, the floors, the threshold re-analysis and the Table 1 caption;
  - "What fits loosely", the laps comparison and the CoPE pooling (all post hoc);
  - the freezing test and the ROBUST cells (registered);
  - Sec. 4.4's "(secondary)".
- *Unlabelled:*
  - Sec. 4.1, "In every cell the |.| runs that solve do so without the wrapped code": a declared secondary, not
    marked;
  - Sec. 4.2, the usage-weighted 99% -> 98%: registered in ACCOUNTS T1, not marked;
  - Sec. 4.5, the signed models' "0.92 of channels end with both move angles of one sign": a declared secondary, not
    marked.
- *Too broad.* Sec. 2's "Every contrast was pre-registered" (W11).

**Budget scoping of "does not learn".** Sec. 2 defines "does not learn" as within the stated budget, and the abstract
ends "Negatives hold within our budgets (300-900 epochs)". Unscoped instances remain:
- *The title* ("rarely learn"): lr- and budget-dependent (W1).
- *The Sec. 4.1 heading* "Barrier parameterisations do not learn the map".
- *The Sec. 4.2 heading* "The |.| solution exists and survives training". On the torus 6/8 runs left the solved
  region, so it did not "survive" in the plain sense.
- *Sec. 4.4,* "The start lever does not transfer to the torus". 5/8 runs per large-start arm were still descending
  at 900 epochs: this is exactly a budget-scoped negative and should say "within 900 epochs".
- *Sec. 4.6 / contribution 5,* "fail these maps" and "removing the wrap does not make CoPE learnable": no budget
  stated.
- *Learning rate.* Nowhere is "within our budgets" extended to "at our learning rates", which W1 shows is the binding
  scope.

**Title.** "Rotary phase steps parameterised to be non-negative rarely learn a map, even when they can represent it
(one-layer models, synthetic maps)".
- "One-layer models, synthetic maps" is honest scoping.
- "Even when they can represent it" is shown for |.| only.
- "Rarely learn a map" is supported on the torus at d 128 and lr 1e-3. On the ring it means "rarely reach loss < 0.05"
  (W7). At lr x3 it is 4/8, and at d 256 it is 2/8 with 6/8 still descending.
- I would accept "rarely learn a map within budget" only with the lr sweep (W1). Without it, prefer a title that
  names the measured quantity, e.g. "... are slower or fail to learn exact maps at standard learning rates".

**Abstract.** Supported apart from W6's three items and the framing issue in W2.

### Claims vs evidence (summary)

| claim | evidence | verdict |
|---|---|---|
| Barrier steps pay a learnability cost on these maps (torus, d 128, lr 1e-3) | -0.177, 0/8 vs 8/8, below retrace, replicated in 3 torus batches | **supported** |
| Same on the ring | SOLVED 2/8, 1/8 vs 8/8; accuracy -0.03; fragile at loss 0.1 for p_plus 0.5 | **supported as "rarely reach exact"**, not "do not learn the map" |
| The cost is robust to recipe and scale | wd 0 and lr/3 yes; lr x3 unmeasured, with \|.\| rising monotonically in lr; d 256 gap a quarter, 6/8 descending | **partly**; the lr direction is unresolved |
| \|.\| solutions exist and are held (ring) / re-found (torus) | exact constructions, continuation 8/8 | **supported**, at ~2.5-5x weight scale |
| The barrier removes the through-zero route, not the function | Lemma 1 (iii), P1, P3 | **proved** (elementary) |
| That removal is why \|.\| fails | +1/+2 test does not confirm; solved \|.\| runs do not use the wrapped code; the flow fit is kappa 0.09 | **not supported**; the paper says so in Sec. 4.5, but the framing leans on it |
| Signed steps escape collectively through the latents | ACCOUNTS T2: 1/8 vs 7/8, Fisher p 0.010, single batch | **supported**, one batch, ring only |
| A past-pi start rescues \|.\| (ring) | 8/8 vs 0/8; sum vs size only partly separated (13-22x vs ~5x mobility, C12) | **supported** on the ring; the "sum, not only the size" sentence (Sec. 4.3 (iii)) is stronger than C12 allows |
| Published monotone methods fail | CARoPE: representational (P6); CoPE: obstacle unidentified | **supported**, but **not evidence for the thesis** |

### Novelty relative to cited prior art

- **Grazzi et al. 2025 and Sarrof et al. 2024.** These prove a representational cost for non-negative real
  eigenvalues. The paper correctly notes that the theorems do not reach rotations by non-negative angles, and that
  Grazzi et al. / Mamba-3 build Z_m from such rotations. The paper does not overclaim against them.
- **Mamba-3 and Selective RoPE.** Both are signed in the angle (Mamba-3: softplus(Delta) x signed theta). The paper's
  practical advice is what they already do; the novelty is the empirical reason, not the design.
- **CoPE and CARoPE.** The faithful re-implementations, the P6 impossibility, and the CoPE representability
  construction contradicting MapFormer's statement are a small but genuine contribution. They are orthogonal to the
  barrier thesis (W5).
- **LRU.** Its phase is exp(.)-parameterised, i.e. a barrier parameterisation in the paper's sense, and LRU reports a
  phase-initialisation effect on PathX (small phases help). That is a published "start decides learnability" result
  for positive phases. The paper cites it in one sentence and should contrast it: LRU's phases are input-independent,
  so its sign is a conjugate-pair convention (Lemma 1 (ii) in the time-invariant case).
- **Sign-constrained reparameterisation.** Woodworth et al. 2020 and Amid & Warmuth 2020 show that w = |u| cannot
  change sign under gradient flow; Dale's-law work (Cornford 2021; Li et al. 2023) asks the same "constraint or
  starting point" question. Against these, P4 is known, and "a sign-constrained parameterisation is harder to optimise
  when the solution wants opposite signs" is unsurprising.
- **What the paper adds.** In the rotary slot the solution does NOT need opposite signs: wrapping provides one. The
  cost persists anyway, and it does so even where no zero crossing is needed (+1/+2), which is the most surprising
  result in the paper and is under-sold.
- **Missing related work:**
  - trivialisations and the optimisation consequences of parameterising angles or orthogonal maps (Lezcano-Casado and
    Martinez-Rubio 2019);
  - circular / phase-wrapping regression;
  - RNN path-integration models trained with signed velocities (Cueva and Wei 2018; Banino et al. 2018), for the map
    side.

**Net novelty.** Modest. A careful empirical characterisation of a learnability gap whose mechanism is still open,
plus an elementary gauge lemma.

### Questions to the authors

1. **Learning rate.** What happens at lr 1e-2 (and with a per-arm-tuned lr, reporting best-of-lr for each arm)?
   ROBUST shows |.| at 0/8 -> 1/8 -> 4/8 from lr/3 to lr x3. Is there a learning rate at which the cost vanishes?
2. **How the solved |.| runs solve.** How do the solved |.| runs (2/8 on the ring, 4/8 at lr x3, 2/8 at d 256) solve
   without a cancelling code (share 0.04-0.13)? What does the usage-weighted cancelling share look like in Fig. 1?
3. **The +1/+2 failure.** On +1/+2 the |.| start is already sign-coherent and no step must cross zero, yet |.| solves
   2/8. What is your best account? Is it the wall (P4: collapsed steps), the 2:1 ratio constraint, or something
   else? Can the GAUGE design be redone with revisit statistics matched between the two move sets? +1/+2 has only
   long-gap laps, so the task changed, not only the sign structure.
4. **d 256 torus.** Do the 6/8 descending |.| runs solve with 2x the budget?
5. **exp(.) parameterisation.** Does an exp(.) (LRU-style) step show the same cost on the ring? It is the barrier
   parameterisation actually used in the literature.
6. **Rank and latents on the torus.** Does an r = 8 |.| arm, or per-head latents, change the torus result?
7. **Softplus.** Can softplus represent the map by the omega_c = 2 pi k / ln 2 construction (W12)?
8. **The verification agent.** What exactly is the "independent code-verification agent", and can the
   pre-registrations be time-stamped publicly (anonymised repo, commit hashes)?
9. **Sum vs size.** Table 1 (iii) says "the sum, not only the size". Appendix C12 says the sub-pi control matches
   mobility only ~5x against 13-22x. Should (iii) be weakened to "partly the sum"?

### Score

**6 / 10 (weak accept for a workshop), confidence 4 / 5.**

- *For acceptance.* For a methods-of-science workshop this is close to a model of how to report a learnability claim:
  - pre-registered batches;
  - existence at a trained scale;
  - causal interventions with controls;
  - failed predictions reported as failed.
- *Against.*
  - The central claim narrowed under the authors' own robustness checks: the lr direction (W1) and the +1/+2 result
    (W2).
  - The mechanism the title evokes is not what the evidence supports.
  - The practical reach is small (W5).
  - As submitted, it is ~1.5-2x the page limit (W3), which is a desk-reject risk at a 4-page venue.
- *Range.* If cut to 4 pages around Secs. 4.1-4.3 + 4.5 with the framing fixed, 6-7. With the lr sweep holding the
  cost, 7. With the lr sweep closing the gap and the paper rewritten around "barrier parameterisations need larger
  steps", still 6-7: it would be a different but honest paper.

### The single change that would most raise the score

**Run a per-arm learning-rate sweep on the observation-step-free ring (signed vs |.|; lr 1e-3, 3e-3, 1e-2, n = 8,
fresh seeds), report best-of-lr per arm, and retitle according to the result.**
- *If |.| stays below signed at every lr.* "Rarely learn" becomes robust, and W1 (the most damaging objection)
  disappears.
- *If |.| catches up at high lr.* The paper has a cleaner and more useful finding ("barrier parameterisations need
  ~3-10x larger steps"). That finding connects directly to P4 (chatter amplitude ~ lr) and to the wall data.
- *Either way.* The result removes the reviewer's main reason to doubt the title.
- *Second, for this venue specifically.* Cut to 4 pages; this is mandatory rather than score-raising.

---

## Part 2. Draft 3's mock review: what draft 5 fixed and what remains

| draft 3 point | status in draft 5 | evidence |
|---|---|---|
| W1 title/abstract too general; no recipe, depth or lr variation | **Partly fixed.** Title scoped ("one-layer models, synthetic maps"); ROBUST added wd 0, lr x3, lr/3, 2-layer d 256 torus. **New problem**: ROBUST shows \|.\| improving with lr and the gap shrinking at scale, so the lr scope is now the binding one (this review's W1) | Sec. 4.1 "Recipe and scale"; C13 |
| W2 mechanism ring-only and loose on the \|.\| side (23% cancel, 0.01-6.58 sums, 44% collapse, start-size confound, T2 signed-only) | **Fixed as reporting; remains as science.** All loose ends now stated ("What fits loosely", C12); "we identify the reason" replaced by "a route ... we do not explain why barrier steps fail". The \|.\| side is still open, and the start-size confound is still only partly controlled, while 4.3 (iii) still says "the sum, not only the size" | Sec. 1, 4.3; C12 |
| W3 theory thin; "sign" is the wrong variable | **Fixed in framing**: Theorem 1 is now "Lemma 1"; the "barrier parameterisation" vs "non-negative value" distinction leads the abstract. **Remains**: the theory is still thin, and its one prediction (E2) was run and NOT confirmed, so the lemma did not earn its place by prediction | Abstract; Sec. 3; Sec. 4.5 |
| W4 representability paragraph muddled; Corollary (i) contradicts P1(a) | **Fixed.** P1(a) with W_in annihilating observations covers every \|.\| arm (built in the real modules, D4); Lemma 1 (i) says "with the same latent"; softplus representability flagged as untested | Sec. 3; D4; E T1(i) |
| W5 ring evidence is a loss threshold; sensitivity not shown | **Fixed as analysis** (C10: four criteria, p_plus 0.5 fails at 0.1, disclosed in 4.1 and Limitations); ring retrace floor added. **Remains as a substantive limit** (this review's W7) | Sec. 4.1; C10, B |
| W6 old 4.6 (depth) and 4.7 (clock) dilute | **Fixed.** Moved to appendices K and J; Liu's C_5 / C_7 / C_8 now quoted; index long-gap at the floor now stated | App. J, K |
| W7 torus rank-slack confound | **Remains**; now stated in Limitations, untested | Sec. 6 |
| W8 narrow baselines: no SSM, no exp(.), softplus init confound | **Partly fixed**: softplus confound stated in Sec. 2 and 4.1. **Remains**: no exp(.), no SSM, CARoPE/CoPE at 1 layer | Sec. 2, 6 |
| W9 presentation: undefined terms, S overloaded, arm codes in Fig. 2 | **Mostly fixed**: terms defined in Sec. 2; sigma for the angle sum; Fig. 2 panels relabelled in words; collapsed channels noted in the caption. **Remains**: "AO" letters unexplained; dense prose; Fig. 2 shading misstates the class (this review's W9) | Sec. 2; Fig. 2 |
| W10 protocol over-claims ("fresh seeds", "code-checked before launch") | **Fixed** for the named cases: SIGN_MATCHED's post-run audit disclosed; text-world pilot seeds disclosed in G/K. **Remains**: "every contrast was pre-registered" is broader than the post hoc analyses in the main text; ROBUST missing from A5 | Sec. 2; A5 |
| Q1 recipe/depth robustness | **Answered** (ROBUST): partly unmeasured | C13 |
| Q2 exp(.) / sigmoid parameterisations | **Unanswered** | Sec. 6 |
| Q3 per-channel capture vs the 23% | **Answered** post hoc: kappa +0.09, the flow does not predict | C12 |
| Q4 wall vs unreachable wrap | **Answered** post hoc: MIXED (51-57% of collapsed steps have l'(0) > 0); not separated | C12, E P4 |
| Q5 test of the gauge prediction (+1/+2) | **Answered** (GAUGE): unmeasured, prediction not confirmed | Sec. 4.5 |
| Q6 r = 8 on the torus | **Unanswered** | -- |
| Q7 SOLVED threshold sensitivity | **Answered** | C10 |
| Q8 torus continuation lr | **Answered**: reason given (the pilot knocked out the signed control at 1e-3); 6/8 at 1e-3 reported | Sec. 4.2; D5 |
| Q9 structured large start on the torus | **Unanswered**; the torus account is now labelled "untested" | Sec. 4.4 |
| Part 2.2: appendix E stale | **Fixed** (v2: drift account withdrawn, T2/T4 results in) | E header, P2, R, P7 |
| Part 2.4: process language | **Mostly fixed**; header line and reference-preamble paths remain | header; References |
| Part 3 E1 / E2 experiments | **E1 run (ROBUST), E2 run (GAUGE)**; neither came out cleanly in the paper's favour. E3 (exp(.), r = 8) not run | -- |

**Net.** Draft 5 fixed nearly every presentation, scoping-in-words and honesty issue raised against draft 3, and it
answered most of the questions with registered batches or post hoc analyses. The two experiments draft 3 asked for
were run, and both weakened the original story:
- E1 surfaced the lr dependence;
- E2 failed to confirm the gauge prediction.

The paper is more honest and better evidenced than draft 3, but its thesis is narrower, so the score does not rise. I
keep 6, with a narrower spread of reasons.

---

## Part 3. Author-side triage

Costs assume the project's setup:
- 2 GPUs at 2 jobs per GPU (4 concurrent);
- a ring run (300 epochs) ~10-25 min;
- a 1-layer torus run (900 epochs) ~30-45 min under load;
- the ROBUST batch (64 ring + 16 d 256 torus runs) took ~10.3 h wall.

| # | weakness | fix type | what to do | estimated cost |
|---|---|---|---|---|
| W1 | lr confound; slow vs stuck | **GPU** (+ text now) | Text now: add "at our learning rates" to the scope in the title, abstract and Sec. 4.1 heading; state the monotone \|.\| trend (0/8, 1/8, 4/8) in the abstract. GPU: pre-registered per-arm lr sweep, ring AO, signed vs \|.\|, lr 1e-2 (and optionally 3e-2), n = 8 fresh seeds, best-of-lr per arm (lr 1e-3 and 3e-3 exist in ROBUST but are on other seeds; rerun them in-batch per rule 12) | Text: 30 min. GPU: 3 lr x 2 arms x 8 = 48 ring runs, ~12-16 GPU-h, ~3-4 h wall; +1 lr adds ~1 h wall. Optional: continue the 6 descending d 256 torus \|.\| runs 900 more epochs (~6-10 GPU-h, ~2-3 h wall), or retrain both arms at 1800 epochs (16 runs, ~15-25 GPU-h) |
| W2 | route framed as the cause | **Text** | Abstract/Sec. 1: say plainly that the barrier removes the through-zero route (proved) and that signed models use it (registered), but that removing it does not explain \|.\|'s failure: \|.\| fails equally where no crossing is needed (+1/+2), and solved \|.\| runs do not use the wrapped code. Promote the +1/+2 result from caveat to finding. Contributions 3-4: one sentence each | 1-2 h |
| W2 (deeper) | why \|.\| fails, incl. +1/+2 | **CPU** first, then GPU | CPU: wall / collapse and gradient-into-wall analysis on the GAUGE +1/+2 checkpoints (scripts exist: `collapse_wall`, the C12 rules). GPU (optional): GAUGE redesign with matched revisit statistics (e.g. +1/-1 at p_plus 0.9 vs +1/+2), n = 8 | CPU: 2-4 h. GPU: 32 ring runs, ~8-12 GPU-h, ~2-3 h wall |
| W3 | length vs 4 pages | **Text** | Cut to Sec. 2 (short), Lemma 1 + P3 in one paragraph, 4.1 (torus + ring in one table), 4.2, 4.3 (Table 1 + freezing), 4.5; one-line pointers to 4.4 and 4.6; drop Howard; everything else to the appendix | 1 day |
| W4 | Fig. 1 unweighted share | **CPU** | Recompute the cancelling share usage-weighted (ACCOUNTS T1's q x k norm weighting) for every Fig. 1 run plus the solved \|.\| runs; redraw; fix the caption denominator. Label post hoc | 2-3 h |
| W5 | practical relevance; 4.6 off-thesis | **Text** (+ optional GPU) | Text: say in Sec. 1 that no published content-dependent rotary method uses \|.\|/softplus on the step, that CARoPE's failure is representational (P6), and that 4.6 is a side result; contrast LRU's input-independent exp(.) phase. GPU (optional): exp(.)-parameterised step on the ring AO beside in-batch signed and \|.\|, n = 8 | Text: 1 h. GPU: 24 runs, ~6-10 GPU-h, ~2 h wall |
| W6 | abstract over-reach | **Text** | "0-4 of 8 ... against 6-8 of 8"; "at a third of the learning rate on SOLVED (accuracy not detected)"; "in a 2-layer d 256 torus model, with a quarter of the accuracy gap and most \|.\| runs still descending"; "near-trained scale (within ~5x)" | 30 min |
| W7 | ring = loss threshold on 97%-accurate models | **Text** | Retitle or rescope to "exact map"; state that ring \|.\| models are above every map-free floor and that the deficit grows with gap (C11) | 30 min |
| W8 | thin theory; Howard unused | **Text** | Present Lemma 1 as a tool with a failed prediction (already in 4.5); drop the novelty-of-conjunction sentence or reduce it to the measurement claim; cut Howard to one sentence | 30 min |
| W9 | figure accuracy | **CPU** | Fig. 2: shade only quadrants II and IV, or caption "closed to \|.\| for omega > 0". Fig. K1: caption 0.513, or plot the 'all' floor. New figure: training-loss curves (median and range) signed vs \|.\| for SIGN_MATCHED torus and one ring batch, from stored per-epoch losses | 1-2 h |
| W10 | rank slack and shared latent on the torus | **GPU** | AbsAO_r8 vs AbsAO_r4 vs SignedAO_r4 on the torus, n = 8; optionally per-head-latent \|.\| | 24 torus runs, ~12-18 GPU-h, ~4-5 h wall; per-head adds ~8-12 GPU-h |
| W11 | protocol statements; verifiability | **Text** | "every verdict was pre-registered"; describe the verification agent in one sentence (what it is, what it checks, blind to what); add ROBUST to A5; anonymised repo with prereg commit hashes and timestamps; state code release and GPU-hours | 1-2 h (repo anonymisation +2-4 h) |
| W12 | softplus representability | **CPU** | Build the omega_c = 2 pi k / ln 2 construction in the real `model_sign` "pos" modules on a ring checkpoint; check angle error and frozen accuracy (as D4). If exact, add it to P1(a) | 1-2 h |
| Unlabelled secondaries | -- | **Text** | Mark the three unlabelled declared secondaries (4.1 last sentence, 4.2 usage-weighted, 4.5 sign-coherent 0.92) | 10 min |
| Scoping of headings | -- | **Text** | "within 900 epochs" on 4.4; "on the ring; re-found on the torus" on the 4.2 heading; "(within budget)" on the 4.1 heading and contribution 5 | 15 min |
| 4.3 (iii) "the sum, not only the size" | -- | **Text** | "partly the sum (the sub-pi control matches mobility only ~5x of 13-22x)" | 5 min |

**Priority.**
1. Text-only (about one day including the 4-page cut): W3, W2, W6, W7, W11, scoping, then W5 and W8.
2. CPU (about one day): W4, W9, W12, the W2 wall analysis on GAUGE.
3. GPU, if one batch: the W1 lr sweep (~3-4 h wall). Then optionally exp(.) (W5) and r = 8 (W10).
