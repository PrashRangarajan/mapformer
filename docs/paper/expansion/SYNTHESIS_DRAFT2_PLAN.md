# Draft 2 of the main paper: integrated plan from the seven expansion reports (2026-10-20)

*CORRECTED 2026-10-21. The independent analysis-script audit (`docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md`) changes
several numbers and claims this plan quotes (text-world NEEDS-MAP gloss and 100-walk subset, the clock / map table, the
T = 1024 coherence strata, "never rank 4's 2D code", parking, the ring "global count"); they are applied in
`docs/paper/MAIN_DRAFT2.md` / `MAIN_APPENDIX2.md` and listed in `MAIN_DRAFT2_notes.md`. The lines updated here are the
what/where run (complete) and X2. Where this plan and the draft differ, the draft is current.*

Inputs, read in full: `PAPER_LEVEL.md`, `MAPPOPE.md`, `RECURRENCE.md`, `RANK_THEORY.md`, `SEARCH_THEORY.md`,
`DEPTH_THEORY.md`, `LANGUAGE.md` (this directory), against `docs/paper/MAIN_DRAFT1.md`, `MAIN_APPENDIX1.md`,
`APPENDIX_theory_v3.md`, `OUTLINE_main_v2.md` and CLAUDE.md. Spot checks made for this plan (CPU, read-only): the
SEARCH / RANK clock outputs (`scripts/sign/rank_clock_out.txt`, `scripts/rank/rank_geom.py`), the what/where
re-analysis at T = 1024 (`docs/audits/2026-10-20/causal_whatwhere_T1024.{log,json}`; then running, complete since with its
`.txt`), the T = 1024 floors
(`docs/audits/2026-10-16/sign_floors_out.txt`), `figs/fig_exchange_rate_out.txt`. Nothing trained, nothing committed, no
existing file edited.

Labels used throughout: **registered** (a registered verdict or declared secondary; the results file is authoritative),
**analysis** (post hoc, eval-only on stored checkpoints or CPU pilots; in the paper every such number carries
"(analysis)"), **theory-proved** (a proof is given in a report or the theory appendix), **theory-conjecture** (a derivation
whose dynamics or assumptions are not checked), **speculation** (untested). Nothing withdrawn in CLAUDE.md is used; the
guards are listed in Sec. 1.3.

---

## 1. The thesis after the expansion

### 1.1 One paragraph

A map in the phase is a code without a clock. Path integration writes the prefix sum of the moves into the rotation, so
one attention layer compares two positions at any gap and across the wrap (theory-proved; registered: 0.998, 8/8 at
T = 1024 against retrace 0.843). Index attention sees only token offsets and must build that sum in its residual stream.
Two layers suffice in principle given a normaliser and enough width (Liu et al. 2023, Lemma 6), so the exchange rates
(about 3 on Dyck-2 and the biased ring, more than 4 on the torus) are what training finds, not representational
bounds. On the torus the trained index models learn local undo rules whose reach grows segment by segment with depth
and never find the global count (analysis); on the small ring three layers do find it (analysis). Whether a path model
finds its map depends on whether training removes the clock: the phase that an out-and-back adds at zero displacement.
Any clock serves short-gap revisits and loses long-gap and wrap-only ones (theory-proved). That single signature is
shared by every failing model, index or path, at the level of what one attention pass can read. A signed step at
per-head rank above the map's dimension D has two map-free ways to remove the clock. One is collective: driving the
latents to z_x + z_y = 0 zeroes every channel at once (theory-proved; registered causal test on the ring: frozen move
latents 1/8 vs 7/8). The other is per channel: each channel can stop reading the part of the clock that lies off the map
plane ("parking"; theory-proved; solved heads show it, analysis). A non-negative step has neither route from a small
start. Under |.| the collective move makes a clock, at a generic readout no latent choice cancels more than 2r channels,
and each channel's clock-free points lie past pi (theory-proved). At rank D the collective route remains, but no channel
can be both clock-free and a full map (theory-proved). The free alternative, "hiding" the drift by no longer resolving
its direction, costs nothing to first order in D >= 2 and is conjectured to starve the collective route. In 1D there is
nothing to hide along, which fits rank 1 learning the ring (theory-conjecture; decisive test G-1). Downstream computation
reads a code but does not repair it. A looped block or extra layers read the same rank-2 code with context (code quality
unchanged, analysis). PoPE's polar score fixes every kernel's peak at zero phase difference, which aims attention at the
target when the phase is a place and away from it when the phase is an index (theory-proved given delta = 0; analysis).
In language the same rules hold word by word. A word gets a step if and only if its count per move is fixed. Asides keep
a small offset so that what is mentioned is not filed at the place. A per-move tick survives only in channels that
retrieval does not use (tick rule, theory-proved; analysis). The used-channel clock ratio tells map data from clock data
where the pooled growth exponent cannot (analysis).

### 1.2 How the pieces fit (one line each, with the report that carries it)

| piece | statement | label | report |
|---|---|---|---|
| why one path layer | the integrator supplies P_t exactly; the score is a function of omega (P_t - P_s) | theory-proved | DEPTH 3.2 |
| why index depth is not enough on the torus | route A (global count) is depth 2 in principle but needs a normaliser, 1/n precision and width ~ N x T; route B (local return sets) is what is learned, reach measured in direction runs | theory (cited construction) + analysis | DEPTH 3.3-3.4, 4.1-4.5 |
| the failure signature | a channel with clock k serves revisits with gap < pi / k; a channel serves place-keyed retrieval only if its tick is 0 mod 2 pi / g | theory-proved | SEARCH S5; LANGUAGE T1 (the same statement, generalised) |
| what removes a clock | collective route R1 (signed, any rank); per-channel route R2 (signed, rank > D: parking) | theory-proved | SEARCH S2, S6(b); RANK Prop. 1(a) |
| why sign matters | |.| has no R1 for an inverse pair and its R2 targets lie past pi; clock >= map below pi | theory-proved | SEARCH S1, S3, S4; v3 P3 |
| why rank D matters (D >= 2) | no clock-free full-map channel; hiding is free to first order and lowers the resolution that drives R1 | proved (static, first-order); dynamics conjecture | SEARCH S6(a); RANK Props. 1(b), 2, 3 |
| why 1D is exempt | no hiding direction; the readout cannot change the drift-to-map ratio | theory-proved | RANK Prop. 1(c) |
| "exists, kept, not found" | the start distribution lies outside the basin of a non-empty, stable solution set | theory (definitional) + analysis | SEARCH S8, 3.3-3.4 |
| readers vs code | one-layer models are position-only readers of their own code; extra passes widen the reader, never the code; a score rule acts on the readout term only | theory-proved (architecture) + analysis | RECURRENCE T1-T2; MAPPOPE Theory 2 |
| the polar score | peak pinned at dphi = 0: helps a path phase, hurts an index phase | theory-proved (delta = 0, measured) + analysis | MAPPOPE Prop. 1, Cor. 1a-1c |
| words | role, not identity, sets a token's offset; tick rule; used-channel rho | theory (light) + analysis | LANGUAGE 2-3, 5 |

### 1.3 Contradictions between the reports, and their resolution

| # | contradiction | resolution | status |
|---|---|---|---|
| X1 | SEARCH: the strong unification ("the failing class lacks a linear route to cancellation") is **false for rank D**, which has R1. | RANK Props. 2-3 supply the missing piece (hiding starves R1 at D >= 2; none in 1D). The first-order statement is proved. The feedback is **not shown**: the one failing pilot seed inherited its weak axis (RANK C6), and the readout gradient at initialisation is uninformative (C8, drift too large for assumption A1). The paper states the route inventory as proved and the hiding dynamics as a conjecture. G-1 decides it. | flagged: conjecture |
| X2 | RANK: solved rank > D heads **park** the clock (off-plane 0.96-1.00, leftover lam 0.003-0.004). SEARCH: at rank 3/4 it is **removed in the latent (R1), not parked** (latent clock 0.045-0.065, in-span 0.999-1.000). | The two use different definitions. RANK's c is the clock common to all four moves plus the mean observation step, measured in the native latent against U. SEARCH's are per inverse pair (c_x = z_-x + z_+x, c_y), and its "in-span" is the share of the *read* channel clock inside the map span; a parked clock is by definition the unread part, so the two agree that what is still read lies in-plane and is small. The magnitudes (0.004 vs 0.065) are not comparable. Unresolved: whether the final shrinkage at rank > D is gradient-driven (R1) or weight decay on a parked, unread direction (RANK C2: "consistent, not a test"). Before drafting: **one definition** (per inverse pair, SEARCH's) and a recomputation of RANK's off-plane share and readgain with it (CPU-3). Add a rank-4 wd 0 arm to G-1 (parking predicts a solved model with a large parked clock; R1 predicts the clock is removed anyway). | resolved (audit, coordinator view): RANK's c = a + o is the drift a walk accumulates per move and observation; in solved rank > D heads the mean move step a and the mean observation step o (each about 0.02 of a map step) lie in the map plane and cancel (cosine -0.79 to -0.999), and only a tiny residual (0.001-0.025 of a map step) is off-plane. SEARCH's per-pair clock leaves out o and so overstates the drift; the two agree on what is read. The end states do not separate parking from removal (the escape arms' latent clock shrinks 14-85x); G-1's r4 wd 0 arm and the seeded escape (G-2) remain the tests (`scripts/rank/clock_decomp_out.txt`) |
| X3 | SEARCH: solved rank-2 heads "keep a moderate clock, 0.26-0.40". RANK: in solved rank-D heads the clock "is simply gone (lam <= 0.001)". | SEARCH's numbers are means over **both heads** of solved runs, and solved rank-2 runs have one clean head and one drifting head (per-head r2 s1: h0 lat-clock 2.10, h1 0.028; block-diagonal s6: h0 0.082, h1 3.50; `rank_clock_out.txt`). This matches "SOLVED iff a clean head" (39/40, close to circular). Wording fix: report the clean head and the other head separately. | resolved |
| X4 | RECURRENCE calls the loop's 1D cancelling code "a third end state". RANK classes are CLOCK (A) and COLLAPSE (B), "points on one curve" (R2). | They are the same object: the 1D code is RANK's collapse corner with the common mode removed (hiding complete). One taxonomy for the paper: **map** (2D, clock-free), **collapsed** (clock-free, one map dimension lost), **clock** (drifting). RECURRENCE's "cancel" is not wrap-aware (its reader columns are), so use the reader-based classification. | resolved (naming) |
| X5 | PAPER_LEVEL 3.1: LOOP_RANK shows "more search (depth) partly recovers rank 2, as a search account predicts". RECURRENCE: the gain is a wider **reader** on an unchanged code, not search (code quality L - A +0.027, p 0.28; L4 - A +0.031, p 0.27; 2D codes 1/16 vs rank 4 8/8). | Follow RECURRENCE, as CLAUDE.md already does ("NOT search at constant capacity"). The registered UNMEASURED verdict stands. The paper's sentence: "more computation reads a rank-2 code; it does not repair it" (analysis). | resolved |
| X6 | PAPER_LEVEL abstract and contribution: "every failing model, index or path, ends with the same code: short-gap revisits served, long-gap and wrap-only at the floor". RECURRENCE: unsolved L / L4 rank-2 runs reach 0.93-0.99 on long gaps. | The signature is a property of the **code read by one pass** (SEARCH S5, RECURRENCE T1). Deep readers recover accuracy, but not the code. State it at the code level. At the accuracy level it holds for one-layer models only. | resolved (wording) |
| X7 | MAPPOPE's proposed main text: "the rank-2 phase does not place a wrapped cell at zero". This is close to the **withdrawn** "the rank defect is the periodic code" (CLAUDE.md, 2026-10-11). | MAPPOPE's own A2 shows that the code term C0 is also low at gap >= 128 (0.81-0.84 vs 0.99 at rank 4), and RANK R1 explains wrap-only revisits as long-gap revisits read through a drift. Rephrase as "does not place long-gap or wrapped revisits at zero phase (C0 0.81-0.84 at gap >= 128, 0.41-0.46 wrap-only; analysis)". Wrap and gap are still not separated (RANK_NOWRAP would separate them). | resolved (guard) |
| X8 | RECURRENCE: readers never repair the code. MAPPOPE: at T = 128, MapPoPE's code term is also higher (C0 0.996 vs 0.905, p 0.0002). | At T = 1024 the score leaves C0 unchanged (short gaps 0.900 vs 0.960, p 0.16; wrap 0.457 vs 0.411). At T = 128 a reader change co-occurs with a better code, plausibly through easier finding (MAPPOPE: speculation; fits GAIN_GRAIN's faster convergence, which did not replicate in GAIN_PHASE). Scope the claim: "at T = 1024, neither the loop, extra depth nor the polar score changes the code". | scoped |
| X9 | LANGUAGE: on the text world's own eval stream, retrace scores **0.934** (0.9337; TW_NORMSTEP stream 0.9365). The draft and PAPER_LEVEL's abstract cite only the 0.505 / 0.597 floors. | Correction C-1 (Sec. 4). The text-world claim must rest on (i) the 179 / 3,055 NEEDS-MAP revisits: path 0.961 on fresh seeds 2-7 and 0.920 on all 8 re-scored, against RoPE 2L 0.138 and best constant 0.112 (analysis); (ii) the mechanism (cancelling steps; direction steps zeroed -> floor, registered). PAPER_LEVEL's CCN abstract must carry the same caveat. | resolved: correction |
| X10 | LANGUAGE counts clock channels as 26-38 (drift between **consecutive** visits). The registered CORRECTED block counts 31-38 (drift from the **first** visit). | Keep the registered 31-38 with its definition. Add the consecutive-visit count as analysis only where the tick-predicts-drift result is quoted. | resolved |
| X11 | PAPER_LEVEL: EM's recency deficit is a **contrast**, not a third axis ("kept" only after restricting the model; 4x budget closes most of it). No other report disagrees. SEARCH S8's "held" leg fits rank and sign, not EM. | Adopt: two sentences in the Discussion plus one appendix table. Keep the 1800-epoch caveat (5/8 rank-2 runs still descending) on "stays". | adopted |
| X12 | RANK, PAPER_LEVEL and RECURRENCE each propose a new "Appendix M" (or "X"). | Re-letter as in Sec. 3.3. | resolved |
| X13 | RECURRENCE E4 proposes a registered loop vs Level 1.5 replication under action noise. CLAUDE.md lists Level 1.5 / InEKF and "filter x loop" as **live negatives: do not re-run**. | Drop E4. The n = 3 loop-under-noise observation stays out of the paper. | dropped |
| X14 | SEARCH S6(b): at rank 3 only **one** clock direction can be parked (two independent pair clocks need rank 4), which would explain rank 3 6/8 vs rank 4 8/8. RANK P9: the expected off-plane share of a random clock is 0.58 vs 0.71. | Two competing accounts of rank 3's shortfall. CPU-4 checks whether the unsolved rank-3 heads carry two independent pair clocks. Neither account goes in the main text until checked. | flagged |
| X15 | RANK C2 explains the parked remainder by AdamW decay. The theory appendix v2 **withdrew** "the drift / weight-decay account" (ACCOUNTS T1) for the **sign** axis. | Different claims: the residual size of a solved rank > D head versus the cause of the |.| failure. Phrase RANK's version as "of the size decoupled weight decay alone gives an unread direction (consistent, not a test)", and never as an account of the sign failure. | guarded |
| X16 | DEPTH: the rates are not representational bounds. The draft reads them as "what a layer of attention computes". | This strengthens the paper's thesis: it is a third learnability statement, for index attention. The construction's MLP width (Lemma 6: 4nT, here 64 x 2048 per axis) exceeds our 512, so state it as "in principle, with a normaliser and enough width"; we do not claim our index models could represent it. | adopted with caveat |
| X17 | LANGUAGE 4.2 lists the Case I/II/III identification for rotary attention as "ours". CLAUDE.md: the taxonomy is published (GRAPE, Mamba-3, Selective RoPE). | LANGUAGE itself says "present it as an organising remark, not a result". The draft already uses it as background. Claim only the measurements (which case a used channel realises is set per channel by the readout; Cases II and III in one model). | guarded |

Not contradictions, but joint support worth one sentence each: MAPPOPE Cor. 1c (RoPE's one-layer kernel is a comb at the
retrace offsets 4j - 1) and DEPTH 4.5 (one index layer learns exactly retrace and the clock) describe the same
one-layer solution from the score side and the accuracy side. RECURRENCE 5.7 and DEPTH 4.2 agree that the index loop and
index depth add only short-gap retrieval, pass by pass. LANGUAGE's pi-tick parity channels are the even-gap lemma of
v3 P5.0 at work.

---

## 2. Title, abstract, contributions

### 2.1 Title

Recommended: **A map is a phase without a clock: what path-integrated attention learns, and what decides whether
training finds it**

Alternatives:
- *Representable, kept, not found: what decides whether path-integrated attention learns a map* (PAPER_LEVEL 2.1).
  Best if the theory section is cut for an 8-page venue.
- *Clocks and maps in rotary attention: depth, rank and sign as learnability variables*. For a main track, once the SSM
  arm (G-3) lands; then generalise to "content-dependent phases".

### 2.2 Abstract (about 240 words; registered numbers only, theory statements only where proved)

> Path-integrated rotary attention replaces RoPE's token index with a learned phase that sums a content-dependent step
> per token. We give an account of what such models learn and of what decides whether training finds a map. On
> 1024-move torus walks, where a map-free rule that retraces the last run of moves scores 0.843, one path-integrated
> layer scores 0.998 (8 of 8 seeds solved) against 0.731 for index RoPE. The rotation carries the prefix sum of the
> moves, so one layer compares positions at any gap. Index attention must build that sum in its residual stream: at
> matched depth one path layer is worth about three index layers on Dyck-2 and on a biased 1D walk, and more than four
> on the torus, where depth helps only short-gap revisits. We show that a map code is one whose phase carries no clock,
> no phase gained over an out-and-back, and that any clock serves only short-gap revisits. Two choices decide whether
> training removes it. A signed step at per-head rank above the map's dimension has two clock-removing routes that
> leave the map intact. A non-negative step has neither from a small start, and at rank equal to the dimension no
> channel can be clock-free and a full map at once. In line with this, rank 2 solves the torus on at most 2 of 8 seeds
> against 8 of 8 at rank 4, and non-negative steps on 0 of 8 against 8 of 8 for signed steps. Yet both constrained
> classes represent the map exactly and keep or re-find an installed solution, and the sign gap holds on a ring at each
> step's best learning rate. Told in English, the walk is learned by one path layer (0.979 vs 0.505), which learns which
> words move it and cancels opposite directions.

Notes:
- "on a ring" stays: the best-rate test is ring-only until G-4.
- Do not add "every failing model ends with the same code" (X6).
- Swap the last clause for the SSM result if G-3 is positive.

### 2.3 Contributions

1. *Measuring map use.* Map-free floors (blank, n-gram, retrace) and revisit strata (short-gap, long-gap, wrap-only)
   defined once and used throughout. At T = 128 retrace scores 0.934, so we measure at T = 1024 (retrace 0.843). There
   one path layer scores 0.998 against index RoPE's 0.731 (8/8 vs 0/8, p 0.0002), and only the path model holds
   long-gap and wrap-only revisits.
2. *Why one path layer, and what depth buys index attention.*
   - The prefix sum is in the rotation (proved).
   - Index attention reaches the map only through a global count. That count is depth 2 in principle but needs a
     normaliser, 1/n precision and large width.
   - Trained index models instead learn local return rules, whose reach grows with depth in direction runs (analysis).
   - Registered exchange rates: about 3 on Dyck-2 and the ring, more than 4 on the torus.
3. *A theory of when the map is found.*
   - The clock / map decomposition of a learned phase, and the failure signature it implies (proved).
   - A route inventory of each step class (proved): signed above rank D, two routes; signed at rank D, one; |.|, none
     from a small start.
   - Registered support on both axes: the solution exists, is kept or re-found, and is not found from scratch. Rank 2:
     at most 2/8; |.|: 0/8 against 8/8; 2x budget; best of four rates on the ring.
   - The causal test of the collective route: freezing the move latents gives 1/8 vs 7/8.
   - End-state checks on every stored model (analysis): failures end with a clock about as large as the map, successes
     clock-free.
4. *Readers and codes.* A score rule, a loop or extra layers change how a code is read, never the code (analysis at
   T = 1024).
   - Loops and depth raise rank-2 accuracy (+0.079 / +0.096, registered) without improving the code (analysis).
   - PoPE's polar score pins the kernel peak at zero phase difference (proved given delta = 0). This explains the
     registered +0.154 interaction: about 80% of it is the index side's loss on revisits at fixed far offsets
     (analysis).
5. *Maps from words.*
   - One path layer learns navigation told in English (registered).
   - Above retrace, the evidence is the 5.9% of revisits that retrace gets wrong: path 0.961 vs two index layers 0.138
     (analysis).
   - The step table follows one rule: a token's offset from its place depends on its role, not its identity (analysis;
     the leak, registered, is the same rule).
   - Clock channels are tolerated only where retrieval does not look (tick rule, proved; analysis).

---

## 3. Section-by-section plan for draft 2

### 3.1 Structure and page budget (TMLR, main text about 13-14 pages; 8-page cut in 3.5)

| # | section | pages | built from |
|---|---|---|---|
| -- | title, abstract | 0.3 | Sec. 2 here |
| 1 | Introduction | 1.0 | rewrite around the thesis; PAPER_LEVEL's bridge sentence ("depth does not buy the map, so the step map must find it") |
| 2 | Background | 0.6 | draft 2.1 replaced by LANGUAGE 6.2; 2.2-2.3 kept; 2.4 moved to Related Work in Sec. 9; 2.5 (Prop. 1, Lemma 1) moved into Sec. 4 |
| 3 | Setup and how map use is measured | 0.8 | draft Sec. 3, plus the strata defined here (PAPER_LEVEL 1), the scoped protocol sentence (C-3), and the statement that the theory and its checks are post hoc |
| 4 | Theory: maps and clocks in the phase | 1.6 | Sec. 3.2 here |
| 5 | One path layer solves the long-walk map | 0.8 | draft Sec. 4 (Table 1), shortened |
| 6 | What index depth buys | 1.3 | draft Sec. 5 + DEPTH 5 text + RECURRENCE 6.2 sentence + segments figure |
| 7 | What decides whether training finds the map | 3.6 | draft Sec. 6 + SEARCH 5.1 + RANK 5.1 + RECURRENCE 6.1 + MAPPOPE main paragraph |
| 8 | Maps from words | 1.2 | LANGUAGE 6.1 (replaces draft Sec. 7) |
| 9 | Discussion and related work | 1.0 | draft Sec. 8 + LANGUAGE 6.3 + PAPER_LEVEL 3.2 (EM) + what/where at T = 1024 + draft 2.4 |
| 10 | Limitations (the one place scope is stated) | 0.4 | Sec. 3.4 here |
| | total | about 12.6 + figures | |

### 3.2 Section 4, the theory section (one place, every statement with its status)

Put it before the results. Its opening sentence says: "This account was developed after the registered batches; every
check in Secs. 5-8 marked (analysis) was computed afterwards on stored models." Each proposition is stated in two to
four lines in the main text, with the proof in App. E. The status column goes into a table at the end of the section
(Table 4).

| # | statement (short) | status | proof / source |
|---|---|---|---|
| 4.1 | **Score identity.** The rotary position factor is the Howard propagator with s = i omega. Position enters a channel only through omega_c (Phi_t - Phi_s). | theory-proved (identity is Mamba-3 Prop. 3) | v3 A0 |
| 4.2 | **The prefix sum is in the phase.** With move steps fixed per move and observation steps zero, theta_t = omega P_t: one layer, any gap, any T, wrap free when omega x step is a multiple of 2 pi / N. | theory-proved | DEPTH 3.2 |
| 4.3 | **Index attention needs a count.** Route A (global count) is depth 2 in principle (Liu et al. Lemma 6; Yao et al. 2021). Its costs are a normaliser, precision 1/n_t, MLP width ~4nT and sensitivity growing with the gap (Hahn and Rofin, used for direction only). Route B: bounded return rules (retrace, constant offset) extended by depth. | theory (cited upper bound); route B's reach is measured, not derived | DEPTH 3.3-3.4 |
| 4.4 | **Clock and map.** For an inverse pair, clock k_c = a_c(x) + a_c(y) and map m_c = a_c(y). Phase over an interval = m_c d + k_c n_x. **Signature:** a clock serves revisits with gap < pi / k and loses the rest. **Tick rule:** a channel serves place-keyed retrieval only if its tick is 0 mod 2 pi / g (g = 2 on the even torus). | theory-proved | SEARCH S5; LANGUAGE T1 (merge into one proposition) |
| 4.5 | **Representability** (Prop. 1) and **relative-sign gauge** (Lemma 1), unchanged from draft 1. | theory-proved | v3 P1(a), P1(f), T1 |
| 4.6 | **Route inventory.** (a) Signed: the clock is diag(omega) W u with u = z_x + z_y. The set {u = 0} cancels every channel and leaves the map free. Gradient flow on the clock surrogate converges to it from every start, and it is generically the only such set. (b) |.|: the same move ends at a(y) = a(x), a clock. At a generic readout at most 2r channels cancel by any latent choice. Each channel has clock >= map below pi, and its non-blind zero-clock points are at |k| = 2 pi n. (c) Rank D: c = U beta, so a channel is clock-free only by losing a map direction; at rank > D the off-plane part of c can be cancelled per channel with the map untouched (parking). (d) D = 1: the readout cannot change the drift-to-map ratio; only the global route acts. | theory-proved | SEARCH S1-S3, S6; RANK Prop. 1 |
| 4.7 | **Hiding.** At rank D >= 2 with isotropic resolution, rotating a channel's resolution off the drift direction lowers the drift loss at zero first-order map cost, and that lowers the rate at which the global route removes the drift. | first-order gradient statement theory-proved under A1-A3; feedback and plateau theory-conjecture (P2 NOT SHOWN) | RANK Props. 2-3 |
| 4.8 | **Exists, kept, not found** = the initial distribution lies outside the basin of a non-empty, stable solution set. | theory (definitional) | SEARCH S8 |
| 4.9 | **Readers and codes.** In one layer, Q and K at a scored query depend on token identity only, so the model is a position-only reader of its own code. Theta is computed once (`model.py`, `model_looped.py`), so loops and layers widen the reader and never change the code class. Any score of the form sum A_c cos(dphi_c + beta_c) splits target coherence into a code term C0 and a readout term C - C0; a score rule acts only on the second. | theory-proved (architecture + algebra) | RECURRENCE T1-T2; MAPPOPE Theory 2 |
| 4.10 | **Peak pinning.** With beta = 0 (PoPE, delta in its dead zone), every head's kernel peaks at dphi = 0 for every content pair (Bochner). Path phase: dphi = 0 is the same place, which is the target. Index phase: it is the current token, so a fixed far offset is unreachable in one layer. | theory-proved given delta = 0 (measured: 98-100% of entries in the dead zone) | MAPPOPE Prop. 1, Cor. 1a-1c |
| 4.11 | **Used-channel ratio.** Steps split into the frequency-weighted mean (clock) and a content part. Place-keyed readouts push used channels to rho << 1, count-keyed readouts to rho >> 1. | theory (light) | LANGUAGE 3.1-3.3 |

Speculations, which stay out of Sec. 4 or are flagged in App. E only: RECURRENCE T4 (deep readers make the collapsed
code a usable "compensation basin"); RANK's reduced plateau law; DEPTH's per-layer hop model (linear vs logarithmic
reach); LANGUAGE's "the clock is used early and abandoned later"; MAPPOPE's "a pinned peak makes a good code easier to
find on short walks".

Figure T1 (new, schematic, no data): left, one channel's phase over an out-and-back for a map (returns to 0), a clock
(advances by k per pair) and a |.| channel (cone: clock >= map). Right, the route diagram: collective (latent u -> 0),
parking (off-plane), hiding (rotate off the drift direction). Five minutes of readers' time saved; costs about 0.3 pages.

### 3.3 Section by section: add, move, cut

**1 Introduction.**
- Add: the thesis paragraph (1.1 here, compressed to about 150 words) and the five contributions (2.3).
- Add PAPER_LEVEL's positioning sentence, early: MapFormer already shows path integration helps; ours are the floors,
  the matched-depth rate and its mechanism, the theory of the two axes, and the words.
- Cut: the three-questions paragraph (it is replaced by the thesis).

**2 Background.**
- Replace 2.1 by LANGUAGE 6.2. Howard's three cases become an organising remark (X17), and the new citations go in:
  Alipour et al. 2025, GridPE (arXiv:2406.07049), Mochizuki-Freeman et al. 2023, Kabir et al. 2025; optionally Sarkar
  and Howard 2026 and RoMem.
- Keep 2.2 (MapFormer) and 2.3 (representability).
- Move 2.4 (orientation-fixing parameterisations, Shakerinava, Liu, Hahn and Rofin) to Related Work at the end of
  Sec. 9.
- Move 2.5 (Prop. 1, Lemma 1) into Sec. 4.

**3 Setup and measurement.**
- Define the strata here and make Fig. 2 the figure the paper returns to (PAPER_LEVEL 1).
- Fix the protocol sentence (C-3).
- Add one sentence on the analyses: post hoc, eval-only, scripts in `docs/paper/expansion/scripts/`, with
  independent-audit status per script in App. O (CPU-9).

**4 Theory.** As in 3.2.

**5 One path layer solves the long-walk map** (draft Sec. 4).
- Keep Table 1 and the headline paragraphs.
- Add one sentence tying it to 4.2: "the score depends on P_t - P_s directly".
- Shorten "Retrace is strong on short walks" to three sentences.

**6 What index depth buys** (draft Sec. 5).
- Keep the three registered rate paragraphs and Fig. 1.
- Replace "What depth buys index attention" and the positioning paragraph by DEPTH 5 (the "Why the rate differs by
  task" text), each analysis number labelled.
- Add RECURRENCE 6.2's sentence (the same four passes lift long-gap revisits on a rank-2 path code, where the code has
  something to read).
- Add the exact accounting of "just above retrace" (DEPTH 4.1: (0.680 x 0.996 + 0.148 x 0.699) / 0.828 = 0.943,
  analysis).
- Add the one-sentence statement that the rates are not representational bounds (X16).
- New Fig. 8 (segments; Sec. 3.4).
- The ring's 0.75 dip decomposition goes to App. F only, labelled descriptive (DEPTH 4.5; the registered primary
  attached no mechanism sentence).

**7 What decides whether training finds the map** (draft Sec. 6, reorganised).
- *7.1 The clock and the signature.*
  - Move draft 6.3 here, now derived from 4.4.
  - Add SEARCH's end-state table (S5) as Table 5: usage-weighted clock / map per arm, analysis.
    - Signed solutions: 0.001-0.018.
    - |.| from small starts: 1.10-1.68.
    - Frozen-latent signed: 1.39.
    - Unsolved rank-2 heads: 0.5-1.9.
    - Solved rank-3/4 heads: 0.043-0.051.
  - Use the clean-head / other-head split (X3) and SEARCH's per-pair definition (X2).
- *7.2 Rank.*
  - Keep the draft's registered paragraphs and Table 2; add Holm (C-11).
  - Add RANK 5.1 ("Why the extra rank helps"), trimmed to about 180 words. Keep its wording constraints: "parks the
    clock outside the map plane" only with the off-plane share and its random baseline beside it, and nothing causal
    until G-1.
  - Add the SCORE_RANK sentence (PAPER_LEVEL 3.1): the polar score does not rescue rank 2 (3/12 vs 3/12). The mechanism
    is in 7.4.
- *7.3 Sign.*
  - Keep the draft's registered paragraphs, Table 3, Figs. 4-5.
  - Add SEARCH 5.1, cut to about 220 words: the two routes and their absence under |.|, the path-profile result and the
    frozen-content qualification.
  - Keep "How signed steps reach the map" (the registered causal test) and the CoPE / CARoPE paragraph.
- *7.4 Readers cannot repair a code.*
  - RECURRENCE 6.1 (about 190 words) on loops and depth.
  - MAPPOPE's main-text paragraph (about 200 words), with the X7 rephrasing.
  - Draft 2 should leave out "loop on Match-Query +0.346" (PAPER_LEVEL keep-out list).

**8 Maps from words** (draft Sec. 7).
- Replace with LANGUAGE 6.1. It reports retrace 0.934, the NEEDS-MAP stratum, the step table by word class, the aside
  collapse / lift test, the idle clock, and "what must and must not move" (the leak with its failed accuracy
  replication).
- Fig. 7 gains panels (d), the step table by word class, and (e), aside collapse / lift.
- Clock-seed per-seed detail goes to App. J.

**9 Discussion and related work.**
- Rewrite "What the two failure axes share" in route terms (one paragraph, from 1.1), with RECURRENCE 6.3's sentence.
- Keep design guidance, and add: "measure the used-channel clock ratio, not the pooled growth exponent".
- *What and where:* the T = 1024 run (CPU-1) is complete (`docs/audits/2026-10-20/WHAT_WHERE_T1024.md`, `.json`, `.txt`;
  analysis; all 8 seeds per arm):
  - Signed rank-4 models keep 0.925-0.999 (mean 0.966) with token identity within type removed (8/8 lower, sign-flip
    p 0.008, uncorrected; Holm over 24 contrasts 0.08). That is above retrace (0.843) on every seed, but the cost (mean
    0.032, up to 0.064) is larger than at T = 128 (0.011-0.025).
  - The shared position kernel times a per-pair content gain plus offset keeps 0.990-1.000 (cost <= 0.001).
  - |.| models: intact 0.821, identity removed 0.544 (-0.277; 7/8 runs lose >= 0.15), gain 0.584 (-0.237).
  - Softplus: intact 0.780, identity removed 0.476 (-0.305), gain 0.675 (-0.106). RoPE: intact 0.731, identity removed
    0.479 (-0.252), gain 0.439 (-0.293).
  - Reading: in the long-walk regime the solving models separate where from what, and the failing ones lean on
    content-position interactions; separation tracks success within the non-negative arms (Spearman 0.84, p 5e-5,
    analysis; `scripts/whatwhere/whatwhere_stats.py`).
- *Integrator view* and *Text:* LANGUAGE 6.3. The pooled growth exponent is replaced by the used-channel ratio (C-2).
- *EM contrast:* PAPER_LEVEL's two-sentence version with the 1800-epoch caveat (X11).
- *Egocentric actions:* two sentences as a boundary only. Rotation actions break a cumsum phase and allocentric
  recoding restores it (+0.050 -> +0.488, 8/8; 16-epoch recipe, index arm on the floor, not registered).
- *Related work:* draft 2.4 plus PAPER_LEVEL's positioning (Shakerinava owns "optimisation not expressivity" for
  non-Abelian; Grazzi / Mamba-3 own representability; ours is the learnability cost in the rotary slot and its route
  account).

**10 Limitations.** Sec. 3.4.

**Appendix, re-lettered.**

| letter | content | source |
|---|---|---|
| A | models, recipes, batches, reproduction of MapFormer v4 Table 2 | draft A |
| B | floors (add the text-world retrace 0.934 / 0.937 and the NEEDS-MAP share; T = 1024 floors per seed) | draft B + LANGUAGE 2.6 |
| C | per-seed tables | draft C |
| D | constructions (add the exp-step construction, CPU-7) | draft D |
| E | theory: v3 (A0, B0, T1, P1-P7, Remark R updated) + SEARCH S1-S8 + RANK Props. 1-3 with A1-A3 + DEPTH 3.3 + RECURRENCE T1-T2 + MAPPOPE Prop. 1 / Theory 2 + LANGUAGE T1-T2; one status table | all reports |
| F | exchange rate: draft F + DEPTH 4.1-4.6 (class splits, attention, probes, run-length eval, ring rules and count, Dyck blocks, the 0.75 decomposition labelled descriptive) | DEPTH |
| G | rank: draft G + RANK C1-C9 (C6 and C9 as CPU pilots; the 5D non-reproduction stated as a recipe difference, rule 16) + Fig. G2 (pilot dynamics) | RANK |
| H | sign: draft H + SEARCH 3.1-3.4 (latent geometry, reachable share, path profiles, basin); H4 rewritten (C-7) | SEARCH |
| I | CARoPE and CoPE | draft I |
| J | text world: draft J + LANGUAGE 2.1-2.6 | LANGUAGE |
| K | what / where at T = 128 and T = 1024; the leak with its fresh-seed accuracy non-replication (+0.0045, p 0.46) | draft K + CPU-1 |
| L | PoPE's score: MAPPOPE's one-page appendix (Tables L1, L2; CORRECTED qualifiers; "what NOT to claim") | MAPPOPE |
| M | loops and depth on a rank-2 code | RECURRENCE 6.4 |
| N | EM recency as a contrast case (WARM, NOLEAK, SEARCH, SPREAD rows) | PAPER_LEVEL 3.2 |
| O | analysis scripts: path, what each computes, verification check (logit-exact / reproduces stored accuracy), audit status | all reports' reproduce sections |

### 3.4 Limitations: the single scope statement (replaces draft Sec. 9)

> Scope.
> - The models are small (1-4 layers, 2 heads, d_model 128-256) and the tasks synthetic; natural text appears only as
>   a three-seed pilot.
> - "Not found" is relative to the stated budgets (300-1800 epochs). Learning rates were swept on the ring only; on the
>   torus, in the larger model, the sign gap is a quarter of the one-layer gap with most |.| runs still descending.
> - Rank results use two heads; our latent is shared where MapFormer's is per head (per-head arms are reported).
> - The torus exchange rate is a lower bound measured at n = 4.
> - The theory was developed after the registered batches. Its proved statements concern the step parameterisation and
>   quadratic surrogates. The hiding feedback at rank D is a conjecture. Every check marked (analysis) is post hoc, on
>   one batch per arm, without fresh-seed replication [and without an independent code audit, until CPU-9].
> - The text world is scripted, its direction words are context-free, and its walks are short enough that retrace
>   scores 0.934.
> - No recurrent or SSM baseline [until G-3].

### 3.5 For an 8-page cut

Keep Secs. 1, 3, 5, 6, 7.1-7.3 and 8. Shrink Sec. 4 to Figure T1 plus 4.4, 4.6 and 4.9 (half a page). Move Fig. 6
(mechanism), 7.4 and the language theory to the appendix.

### 3.6 Figure plan

| fig | file | status | section | content / data source |
|---|---|---|---|---|
| 1 | `fig_exchange_rate` | exists | 6 | caption: name the stream for retrace (0.844 is the FACTORIAL stream; Table 1's 0.843 is the sign batch's) |
| 2 | `fig_strata_modes` | exists | 3, 5-7 | recurring strata figure; add "one-pass reader" ticks for rank 2 if 7.4 keeps the reader (optional) |
| 3 | `fig_rank_solved` | exists | 7.2 | unchanged |
| 4 | `fig_existence` | exists | 7.2-7.3 | unchanged |
| 5 | `fig_loss_curves` | exists | 7.3 | unchanged |
| 6 | `fig_mechanism` | exists | 7.3 (App. H in 8 pages) | unchanged |
| 7 | `fig_textworld` | extend | 8 | add (d) step table by word class (`scripts/language/tw_step_account.json`), (e) aside collapse / lift (`tw_attention_asides_out.txt`, `tw_attention_asides_dironly.json`); caption: eval mode for (c), re-scored for (e) |
| T1 | `fig_theory_routes` | new, schematic | 4 | clock / map / cone for one channel; route diagram (collective, parking, hiding) |
| 8 | `fig_segments` | new | 6 | accuracy vs direction runs spanned (2, 3-4, 5-8, 9-16, 17+): torus index 1L / 4L / loop / path (`scripts/depth/out/torus_depth_*.json`) beside ring index 1L / 2L / 3L (`ring_depth.json`); optional inset: fixed-gap 32-127 rows |
| 9 | `fig_clock_endstate` | new | 7.1 | x: usage-weighted clock / map (log), y: arm; dots = runs, colour = SOLVED; both axes (SEARCH `sign_clock_rows.json`, `rank_clock_out.txt` after CPU-3 harmonisation); side panel: off-plane share of solved rank > D heads vs the random-direction baseline (RANK `sweep_final.json`) |
| 10 | `fig_loop_reader` | new | 7.4 | per stratum, bars for A / L / L4 / rank 4 split into "read by one pass" (soft reader) and "added by later passes"; index 1L / loop as a cross-batch group (RECURRENCE 6.4, `out/SUMMARY.txt`) |
| 11 | `fig_kernel_peak` | new | 7.4 (or App. L) | (a) content-pair kernel vs token offset: PoPE peaks at D = 1, RoPE a comb at 3, 7, 10-11; (b) target coherence C vs C0 for MapWM vs MapPoPE at T = 128; (c) C0 by stratum at T = 1024, r2 vs r4 (`scripts/mappope/kernel_probe_*.json`) |
| G2 | `fig_rank_pilot_dynamics` | new | App. G | kappa and off-plane share vs step for r2 / r4 / R2Clock (pilot n = 3; `scripts/rank/pilot_t128/`) |
| H7 | `fig_path_profiles` | new | App. H | loss along straight paths, signed vs |.|, and the latent-only route (`scripts/sign/path_profiles.json`) |
| J2 | `fig_used_channel_rho` | new | App. J (or Sec. 9) | used-channel rho across torus / text world / code / TinyStories / recency (`crossover_usage.json`, `usage_tick.json`) |

---

## 4. Corrections to draft 1 found by the reports

| id | where in draft 1 | problem | fix | source |
|---|---|---|---|---|
| C-1 | Sec. 7, App. B, App. J, abstract | Text-world floors given as 0.505 / 0.597 only; retrace scores 0.934 on that stream | Report retrace beside the others. Rest the map claim on the NEEDS-MAP revisits (path 0.961 fresh seeds 2-7, 0.920 all eight re-scored; RoPE 2L 0.138; best constant 0.112; analysis) and the registered mechanism. Note that DirOnly (0.970-0.974) is barely above retrace. Run CPU-2 first | LANGUAGE 2.6 |
| C-2 | Sec. 8 "Text" | "the phase grows linearly with token index ... consistent with there being no map to build": text-world models with a perfect map show the same alpha (0.98-1.02) | Replace with LANGUAGE 6.3 "Text". Linear growth is not diagnostic; the used-channel ratio (0.06-0.23 vs 9.9-12.3) separates the two | LANGUAGE 3.4 |
| C-3 | Sec. 3 "Protocol" | "Every verdict was pre-registered" sits beside the ring exchange rate (a declared secondary; CANCEL's registered primary landed in no branch) and beside analysis floors | "Every contrast in Tables 1-3 and Figs. 1, 3-5 was pre-registered with its decision rule; declared secondaries and analyses are marked". Check against App. A | PAPER_LEVEL 1 |
| C-4 | Sec. 8 "What and where", App. K | The checks ran on T = 128 checkpoints, where retrace scores 0.934 and a map is not needed | Re-run at T = 1024 (done 2026-10-20). Replace the paragraph with the T = 1024 numbers and keep T = 128 in App. K | PAPER_LEVEL 1, 3.1 |
| C-5 | Sec. 7 clock seeds | "the model counts steps"; "solve as well"; "zeroing the non-direction words' steps costs it more" read as clock use | The clock counts **moves** (an aside adds <= 0.06 of a tick, a filler <= 0.007). It sits in channels retrieval does not use, at a cost <= 0.009 to remove on the 7 clock models at ceiling. Seed 0 (a clock seed, 0.881 eval mode) is below ceiling and still uses it, so drop "solve as well" for all four. The zeroing cost tracks the direction words' common component in the used channels (Spearman +0.905), not clock use. All analysis | LANGUAGE 2.2-2.3, 6.5 |
| C-6 | Sec. 9 Limitations | "why the non-negative parameterisations stall from small starts is not identified" | Replace with the route account (theory-proved inventory; analysis end states). Keep "the per-channel |.| dynamics are underived" | SEARCH 5.1-5.2 |
| C-7 | App. H4 / theory T1(ii) | "+1/+2 is harder for both; why is open" | "On +1/+2 the |.| step code is learned (relation in 8/8 runs in the used channels); the failures shared with signed lie outside the step code (analysis). There is no evidence of a sign cost without an inverse pair" | SEARCH 0.3, S4 |
| C-8 | Sec. 5 | The rates read as what a layer "computes" | Add: two layers suffice in principle (cited construction, normaliser and width needed); the rates are what training finds at these budgets | DEPTH 3.3 |
| C-9 | Sec. 5 "What depth buys" | "just above retrace" stated without an account | The exact class accounting 0.943 (analysis), and the reach in direction runs | DEPTH 4.1 |
| C-10 | any mention of LOOP_RANK | Risk of "the loop is a search aid at constant capacity" | "Same parameters and initialisation, larger reader function class; the code is unchanged (analysis)" | RECURRENCE 4 |
| C-11 | Sec. 6.1 | "rank 3 improves on rank 2 by +0.102 (permutation p 0.027)" without the family correction | Add Holm 0.054 (CLAUDE.md, `RANK3_RESULTS.md`) | CLAUDE.md |
| C-12 | Fig. 1 caption vs Table 1 | Retrace 0.844 vs 0.843 | Say which stream each is (FACTORIAL s500-503 vs SIGN_MATCHED s0-7) | `fig_exchange_rate_out.txt`, `sign_floors_out.txt` |
| C-13 | Sec. 6.3 | Signature stated as an observation over accuracies | Derive it from 4.4. State it at the code level (X6) | SEARCH S5 |
| C-14 | Sec. 8 "Integrator view" | "signed steps make a map and non-negative steps a path-length clock" | Which case a used channel realises is set by what the readout keys on. The same signed class gives rho 0.064-0.079 on the torus and 21.8-22.1 on recency (analysis) | LANGUAGE 3.3, 6.3 |
| C-15 | App. L | Two lines | MAPPOPE's appendix. The path-side gain is small (+0.0243, shrinking to +0.009 on fresh seeds, plausibly convergence speed); the interaction is about 80% index side | MAPPOPE |
| C-16 | App. K leak paragraph | Gives the fresh-seed accuracies 0.9940 vs 0.9895 without saying the accuracy gain did not replicate | Add "+0.0045, p 0.46: the convergence gain replicated, the accuracy gain did not" | LANGUAGE 5; CLAUDE.md |
| C-17 | Clock-channel counts | 31-38 vs 26-38 | Registered 31-38 (first visit); analysis 26-38 (consecutive visits), each with its definition | X10 |
| C-18 | Abstract (PAPER_LEVEL's proposed version) | "at each parameterisation's best learning rate" drops "on a ring" | Keep "on a ring" | PAPER_LEVEL 2.2 vs draft |

---

## 5. GPU experiments, merged and ranked; CPU checks before drafting

Every GPU batch needs a pre-registration, a CPU task gate, pilots on seeds outside the batch, a results-blind
verification agent with its amendment before reading (rules 7, 11, 29), a check of `nvidia-smi` owners (shared server),
and the user's approval of the cost. Costs are the reports' estimates, scaled from measured timings (torus T = 1024,
1 layer: about 2,700 run-epochs per hour on the two 4090s; 4-layer or loop runs about 3.5x).

### 5.1 Ranked list

"Decisive" means the theory predicts opposite outcomes under its rival accounts.

| rank | id | experiment (merged from) | design | cost | claim it unlocks | decisive? |
|---|---|---|---|---|---|---|
| 1 | **G-1 Clock-route batch (rank)** | RANK G1 + G1+, SEARCH 6.1, SEARCH 6.3, RECURRENCE E2 (snapshot part) | Torus T = 1024, 900 ep, n = 8 fresh seeds. Arms: shared r2; **r2 + clock slot** (identical to r2 at init, +64 params); **r2 drift-forbidden** (inverse-pair latents tied, observation steps 0); r2 ActOnly (observation steps 0, moves free); shared r4; optional r4 wd 0. Geometry snapshots every 10-50 epochs (check snapshots do not change the RNG stream, rule 13) | 5 arms x 8 = 40 runs, about 13-14 h wall; +r4 wd 0 about +2.7 h | Rank theory (4.6c, 4.7). Clock slot >= 6/8 supports "extra rank helps by giving the clock an exit"; <= 2/8 supports over-parameterisation (Burer-Monteiro). Drift-forbidden r2 solving like r4 shows the rank-2 failure is clock removal; failing shows a second obstacle. r4 wd 0 separates parking from removal (X2). Snapshots test P2 / P4 at scale | **yes** (the central rank test) |
| 2 | **G-2 Sign-route batch (ring + torus)** | SEARCH 6.4, 6.2; PAPER_LEVEL #3 | Ring, observation-step-free, n = 8: signed; |.|; **hybrid a = omega abs(W) z** (non-negative readout, signed latent: collective route restored); **|.| with tied latents z_y := -z_x** (clock 2 omega abs(w.z), never zero); **exp steps** (LRU parameterisation) and softplus, each at its best of 3 rates. Torus part: signed rank 4 with move latents frozen, from the default and the |.| start, n = 8 | ring about 1.5 h (hybrid / tie) + 3-4 h (exp / softplus rates); torus 16 runs about 4 h | Separates "non-negative readout" from "no collective route" (S3). Predicted: hybrid solves, tied |.| fails. Completes the R1 / R2 2x2 on the torus (frozen latents from the |.| start predicted <= 2/8). Gives "every non-negative parameterisation we tried" | **yes** (cheapest decisive item) |
| 3 | **G-3 Sign beyond attention: diagonal complex SSM** | PAPER_LEVEL #1 | One-layer h_t = exp(i omega Delta(x_t)) h_{t-1} + B x_t on Z_m (m 8, 32; T 128-512) with +1 / -1 / stay; signed, |.|, softplus, exp; n = 8; existence construction per Prop. 1 | 5-8 GPU-h; 1-2 days code and checks | The route account concerns the step table, so it predicts the same cost in an SSM. Moves the paper into the SSM / RoPE literature (Grazzi, Mamba-3, Shakerinava); answers "nobody uses |.|" (exp is the LRU parameterisation). A null result is an informative scope boundary | yes (generality) |
| 4 | **G-4 Torus best-rate sign sweep (+ rank)** | PAPER_LEVEL #2 | Observation-step-free torus, signed vs |.| at 1e-3 / 3e-3 / 1e-2, n = 8, selection-aware test as LRSWEEP; optional r2 vs r4 at 3e-3, 1e-2 | sign only 40-48 runs, about 12.5-15 h wall; + rank about 25 h | Removes the limitation every reviewer raised; fills the evidence table's LR column on the task that carries the accuracy | no (robustness), but gating for a main track |
| 5 | **G-5 Depth routes** | DEPTH E1, E3 (ring step) | (a) Ring T = 128, index 2 layers with and without BOS + absolute channel (does the count appear at 2 instead of 3?), n = 4-8. (b) Torus walks with run length fixed at 2 / 5 / 10, index 4L and path 1L, trained and tested on each; T = 512 version | (a) about 3 GPU-h; (b) about 24 GPU-h at T = 512 (48 at T = 1024) | Route B predicts accuracy is a function of runs spanned in every condition; route A predicts dependence on T and the gap. (a) tests whether a missing normaliser blocks route A | **yes** (between the two routes) |
| 6 | **G-6 Delta-freedom 2x2 (polar score)** | MAPPOPE #1 | Paper torus T = 128, r2, 300 ep: {index, path} x delta {pinned, free U(-2 pi, 0)}; optional MapPoPE with delta frozen random; n = 8 | about 2.5-3.5 h | Peak pinning (4.10) predicts a **sign reversal**: free delta helps the index arm (concentrated in R3+) and hurts or slows MapPoPE | **yes** (cheap) |
| 7 | **G-7 Long-walk text world + double dissociation** | LANGUAGE E4, E3 | (a) Text world at 4096 words (~530 moves): path vs RoPE 1-2 layers, n = 8. (b) A text world where some queries are count-keyed ("what did she see k moves ago") and the rest place-keyed: used channels should split into rho << 1 and >> 1, and each set's ablation should remove only its query type | (a) about 20-35 GPU-h; (b) about 3 GPU-h | (a) Removes the short-walk caveat that C-1 creates for contribution 5. (b) Turns the tick rule and the used-channel ratio into a within-model test | (b) yes; (a) strengthens |
| 8 | G-8 Rank x sign x passes factorial | RECURRENCE E1 | Torus T = 1024: {r2, r4} x {signed, |.|} x {1 pass, loop x4}, n = 8; primary = geometry class and one-pass reader quality | 64 runs, about 74 GPU-h (about 19 h wall); trimmed 48 runs about 55 GPU-h | "Readers cannot repair a code" on both axes, registered | yes (reader vs code) |
| 9 | G-9 Rank follow-ups | RANK G2, G4, G3 | Seeded escape (lift with clock off-plane share ~0.7 vs default), n = 8, 3.6 h; ring rank 1 vs 2 at T = 1024, 2 h; 3D grid 10 rank 5 vs 4, 5.3 h | about 11 h total | Escape mechanism; whether the 1D exemption is the short task; P9 (D + 2) | G2, G4 yes |
| 10 | G-10 Reviewer-level breadth | PAPER_LEVEL #4, #5, #6, #8 + DEPTH E2 | LSTM / GRU torus baseline (6-10 GPU-h); egocentric actions at T = 1024 (about 10 h); sign and rank at scale (about 27 h); index depth to 8 layers (DEPTH E2 / PAPER_LEVEL #8, about 35-44 h) | 80-90 h | Main-track objections: "why attention", scale, "more than four" turned into a number | no |
| 11 | G-11 Smaller items | LANGUAGE E1; SEARCH 6.5; PAPER_LEVEL #7 + RECURRENCE E5; MAPPOPE #2; RECURRENCE E3; DEPTH E4, E5 | Clock emergence snapshots in the text world (1.5 GPU-h); +1/+2 at 900 epochs (3 h); registered Match-Query rerun with destruction (5-10 h); pinned-peak MapWM at T = 1024 (about 1 day); pass count x8 (30-50 GPU-h); ring vs T (15-60 GPU-h); Dyck across D (about 20 GPU-h) | -- | appendix-level | mostly no |
| -- | dropped | RECURRENCE E4 (loop vs Level 1.5 under noise: CLAUDE.md live negatives, X13); TW_AMBIG (stopped 2026-10-11; low for this paper); TW_LANDMARK (live negative) | -- | -- | -- | -- |

Cheapest set that makes the theory registered: G-2 ring part (about 5 h) + G-6 (about 3 h) + G-1 (about 14 h), about
22 h wall. Add G-3 and G-4 for a main track.

### 5.2 CPU checks before drafting (no GPU)

| id | check | why | cost |
|---|---|---|---|
| CPU-1 | Finish the what/where re-analysis at T = 1024 and write its `.txt` | C-4; Discussion and App. K | done (2026-10-20; all 32 runs, `.txt` written) |
| CPU-2 | Retrace floor and NEEDS-MAP strata on the TW_NORMSTEP and TW_STATECHANGE batches (LANGUAGE E5) | C-1: no text-world number goes into the paper until done | about 1 h |
| CPU-3 | Harmonise the clock metric: recompute RANK's off-plane share and readgain with SEARCH's per-inverse-pair clocks on the same checkpoints; report clean and drifting heads separately | X2, X3; Fig. 9 | about 1 h |
| CPU-4 | Rank-3 heads: are the two pair clocks independent in the unsolved runs (S6(b): one parkable at r = 3)? | X14 | < 1 h |
| CPU-5 | Polarise MapWM r2 / r4 at eval (beta -> 0) at T = 1024, by stratum (MAPPOPE #3) | 7.4: a cheap check of Theory 2 | < 1 h |
| CPU-6 | Loss by stratum for the LOOP_RANK arms (RECURRENCE T3) and strata for the aids (PAPER_LEVEL CPU (c), mostly in RECURRENCE 5) | 7.4, App. M | < 1 h |
| CPU-7 | Exp-step existence construction (as softplus, P1(f)) | needed by G-2, G-3 | < 1 h |
| CPU-8 | The evidence table (PAPER_LEVEL 2.4) from existing files; MAPPOPE's T = 1024 probe at 100 walks instead of 40 | Sec. 7 table; App. L numbers | about 1 h |
| CPU-9 | Independent, results-blind code audit of the analysis scripts whose numbers enter the main text: `mappope/kernel_probe.py`, `recurrence/phase_reader_soft.py`, `geometry.py`, `pass_strata.py`, `depth/torus_depth.py`, `ring_depth.py`, `torus_probe.py`, `language/tw_retrace_floor.py`, `tw_strata.py`, `tw_step_account.py`, `tw_attention_asides*.py`, `usage_tick.py`, `crossover_usage.py`, `sign/sign_clock.py`, `rank_clock.py`, `path_profiles.py`, `rank/rank_geom.py`, `c1_runlevel.py`, `escape_start.py` | None had one (the reports say so). Rule 9: probes lie confidently. The reports' self-checks (logit-exact forwards, stored accuracies reproduced) are necessary, not sufficient | an agent, half a day |
| CPU-10 | Build Figs. T1, 8-11, 7d-e, G2, H7, J2 from stored outputs | Sec. 3.6 | half a day |
| CPU-11 | End-to-end read of draft 2 after assembly (rule 28); grep the generators for any withdrawn number (rule 18) | -- | -- |

---

## 6. Status table: every new claim draft 2 would carry

"New" = not in draft 1. R = registered, A = analysis, P = theory-proved, C = theory-conjecture, S = speculation.

| # | claim (as the paper would state it) | label | source |
|---|---|---|---|
| 1 | The integrator puts P_t in the rotation; one layer compares P_t - P_s at any gap | P | DEPTH 3.2 |
| 2 | Index attention can represent the count in two layers in principle (normaliser, 1/n precision, large width) | P (cited construction) | DEPTH 3.3; Liu et al. 2023 Lemma 6 |
| 3 | Exchange rates are learned rates, not representational bounds | P (from 2) + R (rates) | DEPTH 3.3 |
| 4 | Deep torus index models solve retraces (0.996 / 1.000) and decay with direction runs spanned (0.93 / 0.98 at 3-4, 0.74 / 0.65 at 5-8, 0.56 / 0.50 at 9-16) | A | DEPTH 4.1 |
| 5 | At a fixed gap of 32-127 moves the same decay appears (0.87 / 1.00 at 2 runs to 0.49 / 0.50 at 17+) | A | DEPTH 4.1 |
| 6 | 0.943 short-gap accuracy is exactly retrace plus partial 3-8-run returns | A | DEPTH 4.1 |
| 7 | No global count in torus index models (P_t / t R^2 <= 0.21); a recent-displacement code degrading with the window | A | DEPTH 4.3 |
| 8 | Three ring index layers are flat in runs spanned and carry a count (P_t / t R^2 0.69) | A | DEPTH 4.5 |
| 9 | One ring index layer learns exactly retrace and the one-lap clock | A | DEPTH 4.5 |
| 10 | Deep index models' last layer attends to the last visit on retraces (0.55-0.58) and not at long gaps (0.00-0.01; path 0.81) | A | DEPTH 4.2 |
| 11 | Run-length-shifted eval: two-run returns gap-invariant; runs set the level, the gap trims it | A (distribution-shifted) | DEPTH 4.4 |
| 12 | Clock / map decomposition; a clock serves gaps < pi / k only (signature) | P | SEARCH S5 |
| 13 | Tick rule: usable for place-keyed retrieval only if tick = 0 mod 2 pi / g | P | LANGUAGE T1 |
| 14 | Signed collective route {z_x + z_y = 0}: cancels every channel, map free, surrogate flow converges, generically unique | P | SEARCH S2 |
| 15 | Freezing move latents blocks the escape (1/8 vs 7/8) | R | ACCOUNTS T2 |
| 16 | |.| cone: clock >= map below pi; zero-clock non-blind points at 2 pi n | P | SEARCH S1 |
| 17 | |.| has no collective route for an inverse pair; <= 2r latent-cancellable channels at a generic readout | P | SEARCH S3 |
| 18 | +1/+2: |.| has two mirror collective zeros; frustration contracts to the blind origin when both coherence classes are populated | P (surrogate) | SEARCH S4 |
| 19 | Latent-reachable share of the needed change at init: signed 1.000, |.| 0.34 / 0.56 (unweighted / usage) | A | SEARCH S3 check |
| 20 | Straight-path loss to the solution: |.| rises 0.6-1.9 nats, signed 0.04-0.48 (complete separation); signed latent route alone 0.30 -> 0.72, |.| stays 0.30 | A | SEARCH 3.3 |
| 21 | With the content frozen, the start decides the step map's search; the signed escape from the same-direction start is not reproduced by the step map alone | A | SEARCH 3.4 |
| 22 | Failing arms end clock about the size of the map (1.1-1.9 |.|; 0.5-1.9 unsolved rank-2 heads); successes clock-free (<= 0.065) | A | SEARCH S5 table (harmonise, CPU-3) |
| 23 | |.| trained ring models: 0.30-0.36 of sub-pi channels at the cone boundary (0.056 at init) | A | SEARCH S4 check |
| 24 | On +1/+2 the |.| step code is learned (8/8 in used channels); unsolved runs fail outside the step code | A | SEARCH S4 |
| 25 | Rank D: no channel is clock-free and a full map while the clock is non-zero; rank > D: per-channel parking at zero map cost | P | SEARCH S6; RANK Prop. 1 |
| 26 | 1D exemption: no readout change alters the drift-to-map ratio | P | RANK Prop. 1(c) |
| 27 | Hiding is free to first order at D >= 2 (isotropic P) | P (under A1-A3) | RANK Prop. 2 |
| 28 | Hiding starves the global route (feedback, plateau) | C (P2 NOT SHOWN) | RANK Prop. 3 |
| 29 | Drifting rank-D heads drift along their least-resolved axis (57/62 runs vs 28.7 expected) | A (not discriminating) | RANK C1 |
| 30 | Solved rank > D heads hold the small leftover clock off the map plane (0.96-1.00 in 9 of 10 sets vs random 0.50-0.71) and barely read it | A | RANK C2 (re-check CPU-3) |
| 31 | A fresh rank-4 step map removes the drift in 16/16 heads, by parking; lifted runs solve in order of initial off-plane share (rank-sum p 0.018) | A | RANK C5 |
| 32 | Clock slot rescues a failing rank-2 start (one matched pair, T = 128) | A (pilot, n = 1 pair) | RANK C6 |
| 33 | Rank 2 + clock slot learns the T = 1024 torus like rank 3-4 | S (prediction; G-1) | RANK P7 |
| 34 | One-layer models are position-only readers of their own code (model - reader +0.028 rank 2, -0.002 rank 4) | P (structure) + A | RECURRENCE T1, 5.3 |
| 35 | Loop and depth leave the code unchanged (reader quality L - A +0.027, p 0.28; L4 - A +0.031, p 0.27) and read it better (+0.32-0.44 on long / wrap) | A | RECURRENCE 5.3 |
| 36 | The loop is a 4-step computation: one pass of its weights scores 0.381; retrieval forms at passes 2-4 | A | RECURRENCE 5.2 |
| 37 | Aids end in 1D clock-free (8/16) or drifting codes, never rank 4's 2D code (1/16 vs 8/8) | A | RECURRENCE 5.4 |
| 38 | The loop gives no larger or more coherent step-map gradient at init (Adam-invariant x2.9) | A | RECURRENCE 5.6 |
| 39 | Deep readers make the collapsed code a usable basin | S | RECURRENCE T4 |
| 40 | Peak pinning under beta = 0 | P (delta = 0 measured) | MAPPOPE Prop. 1 |
| 41 | Target coherence C 0.70 (MapWM) vs 0.995 (MapPoPE), T = 128; action-key attention 0.179 vs 0.041 | A | MAPPOPE Cor. 1a-1b |
| 42 | PoPE's index kernel cannot prefer offset 3 over 1 (K(3) > K(1) in 0% of pairs); 0.110 of the -0.126 gap is on R3+ revisits | P (Cor. 1c) + A | MAPPOPE Cor. 1c |
| 43 | About 80% of the registered +0.154 interaction is the index side | A | MAPPOPE |
| 44 | At T = 1024 the score repairs the readout term on short gaps (C +0.34, p 0.0002; C0 unchanged) and leaves the code term on long-gap and wrap revisits | A (40 walks; CPU-8) | MAPPOPE Theory 2 check |
| 45 | Free delta reverses the interaction | S (prediction; G-6) | MAPPOPE #1 |
| 46 | Text world: retrace 0.934; NEEDS-MAP revisits path 0.961 (fresh) vs RoPE 2L 0.138 | A | LANGUAGE 2.6 |
| 47 | Step table by word class: optional words 0.004-0.03 of a move; aside words large but cancelling to a 0.04-0.16 offset | A | LANGUAGE 2.1 |
| 48 | Collapsing asides onto the place drops learned models to 0.952-0.975 with 71-100% aside errors; offsetting DirOnly's asides lifts it to 0.999-1.000 (8/8) | A | LANGUAGE 2.4 |
| 49 | The text-world clock counts moves (Case II) in channels retrieval does not use; removal costs <= 0.009 at ceiling | A | LANGUAGE 2.2-2.3 |
| 50 | The tick alone predicts which channels drift (agreement 0.98-1.00, 32 models) | A | LANGUAGE 2.2 |
| 51 | Used-channel rho: torus 0.064-0.079, text world 0.06-0.23, code 4.5-6.6, TinyStories 9.9-12.3, recency 21.8-22.1 | A (code, TinyStories n = 3 pilots) | LANGUAGE 3.3 |
| 52 | The pooled growth exponent does not separate map from clock data | A | LANGUAGE 3.4 |
| 53 | "A token's offset depends on its role, not its identity" (leak, DirOnly and asides as instances) | C (light) + R (leak) + A | LANGUAGE 5 |
| 54 | What/where at T = 1024: signed r4 keeps 0.925-0.999 with token identity within type removed, 0.990-1.000 with a shared kernel x per-pair gain plus offset; |.| models lose 0.28 on average (7/8 runs >= 0.15); separation tracks success (Spearman 0.84, analysis) | A (complete; CPU-1) | `causal_whatwhere_T1024`, `scripts/whatwhere/whatwhere_stats.py` |
| 55 | EM's recency deficit is a different kind of search failure (closed mostly by 4x budget) | R pieces + A contrast (not registered) | PAPER_LEVEL 3.2 |
| 56 | The exact start-to-solution basin does not reach halfway along the straight |.| segment within 2000 step-map steps | A | SEARCH 3.4 |

---

## Appendix to this plan: what was checked here, and what was not

- The X3 resolution was checked on `rank_clock_out.txt` (per-head rows of the solved per-head r2 and block-diagonal
  runs).
- X2's definitional difference was checked against the docstrings of `sign/rank_clock.py` and `rank/rank_geom.py`. The
  0.004-vs-0.065 gap was not recomputed (CPU-3).
- The what/where numbers were first read from the JSON at about 02:00 on 2026-10-20 (Signed_r4 8/8, Abs_r4 8/8, Pos_r4
  5/8, RoPE 0/8); the run has since completed (32/32, `.txt` written) and Sec. 3, item "9 Discussion", above carries the final values.
- Every other number in this plan is copied from the named report or results file. None was re-derived.
