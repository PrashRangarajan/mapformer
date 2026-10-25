# Big picture, 2026-10-24: where the work is, what is ours, what to stop

Advisor read, no experiments run, nothing edited. Sources: CLAUDE.md, `project_state.md`, `WHERE_THINGS_STAND.md`,
`MAIN_DRAFT2.md` + notes, `CLAIM_MAP_clock_map.md`, `SYNTHESIS_DRAFT2_PLAN.md`, `OUTLINE_overall.md`,
`RUNBOOK_NEXT_EXPERIMENTS.md`, the registered results files named below, `docs/lit/LIT_*.md`,
`review_draft5/{PRIOR_ART,MOCK_REVIEW_draft5}.md`, `papers/INDEX.md`, the 2026-10-14/15/16 downloads, Semantic Scholar
citation lists (MapFormer: 0 citers indexed; Shakerinava 2026: 11) and web searches run today. Status words as in
`WHERE_THINGS_STAND.md`: REG, REG-sec, analysis (post hoc, eval-only), pilot.

In flight, not waited for: SIGNSEP (kalman), G-4 TORUS_LRSWEEP (koopman), RANKDECAY (koopman). Each is placed below.

---

## 1. The claim the evidence supports today

**One sentence.** On navigation tasks whose revisits cannot be served by a local rule (1024-move torus walks, also told in
English), one layer of attention whose rotary phase path-integrates a learned content-dependent step learns the map that
index attention does not learn within four layers; whether training finds that map is decided by two choices of the
step map, the sign of the increment and the per-head rank above the map's dimension, and both are costs of search, not
of representation: the constrained classes represent the map exactly and keep an installed solution, yet rarely find it.

Three qualifiers belong in the sentence or beside it: "within 300-1800 epochs" (every failure is budget-scoped and several
arms were still descending), "with AdamW weight decay 0.05" (rank 4 needs it: `RANKWD_RESULTS.md`), and "one recipe per
task, learning rate swept on the ring only" (G-4 in flight closes the torus half for sign; nothing closes it for rank).

**The four results that carry it.**

| # | result | numbers | status | file |
|---|---|---|---|---|
| 1 | One path layer solves the long-walk map; index attention does not, at one or four layers or looped; T = 128 is mostly map-free | T = 1024: path 0.998 (8/8) vs RoPE 0.731 (0/8), p 0.0002, retrace floor 0.843; depth: index 4L 0.860, loop x4 0.864 vs path 1.000 (n = 4, complete separations); T = 128 retrace 0.934 | REG (n = 8); FACTORIAL REG n = 4; floors analysis | `SIGN_MATCHED_RESULTS.md`, `GENERALITY_RESULTS.md`, `FACTORIAL_RESULTS.md` |
| 2 | Sign of the increment is a learnability cost in the rotary slot, in attention and in a diagonal complex SSM | torus 0/8 vs 8/8 (-0.177); ring at each step's best of four rates 2/8 vs 8/8 (p 0.007); exact non-negative solution held 8/8 (ring) / re-found 8/8 (torus); hybrid non-negative readout + signed increment 16/16; SSM 0/8 vs 8/8 (p 0.0002) | REG, 10+ batches | `SIGN_MATCHED`, `LRSWEEP`, `WRAP_EXISTENCE`, `GENERALITY`, `SIGNROUTE`, `G3_SSM` |
| 3 | Per-head rank above the map's dimension decides learnability on 2D/3D tori; the obstacle at rank D is the clock | per-head rank 2: 0/8, 2/8, 2/8; rank 3 6/8; rank 4 8/8 (D - C_bd p 0.0070); 2x budget 0/8 vs 7/8; rank-2 projection 0.9955 frozen, held 7/8; drift-forbidden rank 2 8/8 vs actions-only 1/8 (p 0.0014); 1D exempt (rank 1 7/8) | REG; clock-exit mechanism PARTIAL | `RANK_SEP`, `RANK_PROJ`, `LOOP_RANK_E1800_P1`, `RANK_ND`, `RANK_WRAP`, `RANK_TESTS`, `RANKCLOCK` |
| 4 | The same one layer learns the walk told in English, cancelling opposite direction words | fresh seeds 0.979 vs RoPE 1L 0.505, 6/6 vs 0/6 (p 0.0022); RoPE 2L 0.793; on the 13.4% of revisits retrace cannot reach: path 0.956 vs RoPE 2L 0.411 (below the 0.506 constant) | REG (A); stratum analysis | `TEXTWORLD_RESULTS.md`, `tw_strata_200walks_out.txt` |

Everything else in draft 2 (PoPE peak pinning, Selective RoPE, what/where, readers vs codes, the clock/map theory) is
support, mechanism or framing. The paper is positive: it says what one layer of path integration buys, measures it
against the right floor, and names the two design choices that decide whether training gets it.

---

## 2. Novelty audit, claim by claim

| contribution | prior art | ours | strength |
|---|---|---|---|
| **One-layer path integration vs index depth on long-walk maps** | MapFormer v4 Table 2 already shows one-layer MapWM solving 2D navigation at T = 128 where RoPE/TAPE/CoPE at up to 4 layers fall short; TEM-t (path-integrated position as the transformer's PE); depth-for-recurrence shortcuts (Liu et al. 2023, Yao 2021); Heo et al. 2026 (2603.05573) give a Lie-algebraic depth account for parallelizable models (expressivity, not learning); PaTH / Selective RoPE solve parity in one layer | (i) the floor critique: MapFormer's benchmark at T = 128 is 93% solvable by a map-free retrace rule, so "helps navigation" was mostly measured on local returns; (ii) the matched-length, matched-depth exchange rate on three tasks (about 3 index layers on Dyck-2 and the biased ring, more than 4 on the torus) with strata showing depth buys only short-gap revisits; (iii) the analysis that trained index models learn local return rules, not a count | **Solid and useful, incremental in kind.** The floor result is the sharpest new point and will be read as a correction to MapFormer's benchmark. The exchange rate is a measurement, budget-scoped, with the torus number at n = 4. Do not claim capacity (Liu's Lemma 6 + 8 covers the abelian torus in principle) |
| **Sign as a learnability cost in the rotary slot, now in an SSM** | Sarrof 2024 / Grazzi 2025 / Khavari 2025: representational sign cost for real eigenvalues; Grazzi Thm 3 and Mamba-3 Sec. 3.2: non-negative angles represent Z_m; Shakerinava et al. ICLR 2026 (2603.01959): "solutions exist, training does not reach them" for input-dependent rotations in SSMs, non-Abelian groups, near-solution init recovers; Helfrich 2018 (scoRNN: rotation orientation fixed at init); Woodworth 2020 / Amid-Warmuth 2020 (a reparameterised weight cannot cross zero); LRU phase init; MapFormer Figs. 3d/9b (antipodal latents end state); MapFormer attributes CoPE's failure to its positive gate (representational reading, which Prop. 3 shows is wrong for a rotary step) | the sign of a content-dependent rotary step isolated as one operation at matched length; rule-14 existence design (constructed, frozen, trained, at trained scale); the relative-sign gauge (Lemma 1) for content-dependent steps; the causal route tests (frozen latents 1/8 vs 7/8; hybrid readout 16/16); best-rate test; replication in a diagonal complex SSM with the identical task and init | **Defensible; nobody has done it.** Two blunt weaknesses: (a) no published method uses a |.| or softplus increment, so "allow a sign" is already practice (Selective RoPE, Mamba-3); the practical hook is that published MONOTONE content phases (CoPE, CARoPE, GRAPE-AP, and Selective RoPE's state-tracking bias) fail the map, and the exp arm (the LRU/Mamba parameterisation) was unstable at 3e-3 and unmeasured at 1e-3; (b) the mechanism half (routes vs barrier) is still an account: SIGNROUTE did not separate them, SIGNSEP is the attempt. The scientific content that survives any mechanism outcome: a constraint that removes no function still costs training, in attention and in the SSM. That is the sentence to lead with |
| **Per-head rank D vs D + 1, and the clock account** | MapFormer ablates r (rank 1 fails 2D; rank 2 solves 2D at T = 128; Table 6 5D collapse); LieRE's generator-density sweep (interior optimum); Larson 2026 (2609.12259, Sept.): a "causal rank law", gradient descent recruits exactly the minimal faithful dimension d_min on group-composition tasks, forced rank d_min clears the bar; Burer-Monteiro-style over-parameterisation folklore | per-head rank D fails and D + 1 succeeds on 2D and 3D tori at T = 1024 while a rank-D solution exists and is held (search, not capacity); not hard in 1D; drift-forbidden rank 2 solves 8/8 (the clock is the obstacle, registered); the hiding/parking account | **The empirical D vs D + 1 result is ours and novel, and in tension with Larson's rank law, which must be cited and reconciled** (his memories have a fixed decoder and dense state supervision; ours a sparse revisit loss and a shared readout across heads). The mechanism is weaker: the registered clock-exit test was PARTIAL, Conjecture 1 (hiding) is untested, the latent is shared where MapFormer's is per head, n_heads = 2, one recipe, and RANKWD shows the rank-4 side is optimiser-conditional (8/8 at wd 0.05, 1/8 at wd 0) while rank 2 was never given its own recipe search. Say "rank D + 1 with decay"; drop "rank D is hard" as a law |
| **Maps from words** | TEM-t and MapFormer's naturalistic-data claims; CoPE counts selected tokens; nothing exact | one path layer learns a scripted torus walk in 58 words (REG); opposite direction words cancel; role-not-identity offsets for asides; the per-move clock in unused channels | **Medium-weak as a contribution, good as a demonstration.** A reviewer will say it is the torus with synonyms and fillers, and that retrace scores 0.934 on it; the map evidence is a 13.4% stratum (analysis). The aside-offset finding ("only actions move" is the wrong separation in language) is the one genuinely new idea here. Keep it as a section, not a headline |
| **PoPE peak pinning** | PoPE (Gopalakrishnan et al. 2025) defines the polar score; nobody gives the kernel-peak account | Prop. 7 (kernel peaks at zero phase difference given delta = 0) and the registered delta-free test (+0.122 index, path at ceiling) | **Novel, small, narrow audience.** Mechanism of a +0.15 interaction at T = 128 on a task mostly solvable without a map. Appendix in the main paper; a short note elsewhere if wanted |
| **Selective RoPE fails the map** | Selective RoPE (Movahedi et al. ICLR 2026) tests parity and LM, not navigation | faithful module, 0/8 vs 8/8 within 900 epochs, beats index +0.236, observations step 0.9x moves (the leak at full size), state-tracking config is effectively non-negative and fails like |.| | **Modest, budget-scoped (every run descending).** Valuable as design guidance ("keep observations from stepping the phase") and as the link between the sign result and a published signed method. Do not write "cannot" |
| **What/where separation** | MapFormer Fig. 9 (observation steps -> 0); TEM; Whittington 2023 (factorised codes from non-negativity + efficiency); Song & Zhong 2024, Barbero 2025 (positional vs semantic heads in trained RoPE models) | score-level decomposition at T = 1024: the solving models do not need the content x position interaction (cost 0.032), failing ones do; separation tracks success | **Not a contribution; a discussion paragraph.** Post hoc, one batch per arm, the Holm-corrected contrast is 0.08. "Separation is learned" is prior art |
| **Howard Case I/II/III = constant / monotone / signed rotary rate** | Howard 2014; GridPE (VCO -> RoPE); the Tiganj lab keeps s real; RoMem (Howard-motivated monotone phase rate) | the three-way identification for rotary attention, and the remark that in the imaginary slot Cases II and III coincide representationally and differ in learnability | **Framing, not a result.** State it once in Background and in the CCN abstract. A reviewer can call it obvious once Howard, Mamba and Selective RoPE sit side by side |

Scoop check (web, 2026-10-24): no paper found that isolates the sign of a content-dependent rotary step, that reports a
per-head-rank learnability threshold for path-integrated attention, or that follows up MapFormer (Semantic Scholar indexes
no citers). The nearest neighbours are Shakerinava 2026 (cite as the precedent for "exists but not learned" in rotation
SSMs), Larson 2026 (rank law; cite and contrast), Howe 2026 (2609.18966, exact-init arms pulled off by Adam; concurrent,
methodological kin), and Heo et al. 2026 (depth, Lie algebraic; cite in Sec. 6). None undercuts the claims; two
(Shakerinava, Larson) shrink the novelty of the framing and must be in Related Work.

---

## 3. The five strongest reviewer attacks on draft 2

| # | attack | existing data | in flight | needs new |
|---|---|---|---|---|
| A1 | **"Not found within budget" means slow, not unable.** Rank 2: 5/8 still descending at 900 and at 1800 epochs; |.| 3/8 descending on the torus; every SRoPE run descending; most clock-slot runs descending. The paper's own Limitations say so | 1800-epoch rank batch (0/8 vs 7/8); RING_BUDGET 3x (3/8 -> 3/8, no cancelling code found); flat STALLED classes on the ring; existence-held results (a solution 30 epochs away is kept, so the failing runs are not near one) | G-4 adds rates, not epochs | A loss-curve figure with the registered STALLED/DESCENDING classes per arm (CPU) and the phrase "search within budget" on every failure claim. A 3x-budget torus batch for rank 2 and |.| (~1 GPU-day) would settle it; recommended only if a TMLR reviewer asks |
| A2 | **Optimiser confound (rule 15).** Sign: swept on the ring only. Rank: never swept; the rank-4 side needs wd 0.05 (RANKWD), so "rank D + 1 suffices" is conditional on a recipe that rank 2 never got to choose | LRSWEEP (ring, sign); ROBUST (wd 0, lr/3, lr x3 UNMEASURED); RANKWD | **G-4** answers the torus half for sign. **RANKDECAY** says which parameters' decay rank 4 needs; it does not answer the rank-2 side | **A registered rank-2 recipe search** (rank 2 at lr 3e-3 and wd 0.1 / 0.2, with rank 4 anchors; ~24 runs, ~8-10 h). This is the one new batch the rank claim needs. Without it, write "at the recipe under which rank 4 learns" |
| A3 | **Practical relevance / "nobody uses |.|".** Selective RoPE and Mamba-3 are signed already; the parameterisation that is used (exp, LRU/Mamba's softplus Delta) blew up at 3e-3 in the SSM and was not run at 1e-3 | SIGNROUTE breadth (exp, softplus 1/8-3/8 at two rates, budget-scoped); CoPE / CARoPE / SRoPE-ST monotone failures; the SSM |.| result | none | **G-3 block A2 (exp at 1e-3, 40 runs, ~2 h).** Cheap and it turns "exp was unstable" into a statement. Then lead the sign section with the published monotone phases, not with |.| |
| A4 | **The theory is post hoc bookkeeping and its one registered mechanism prediction (clock exit) was PARTIAL; Conjecture 1 is untested; the route account did not separate from the barrier (SIGNROUTE Amendment 1).** Mock review W8 said this of draft 5 and it still holds | DF 8/8 vs AO 1/8 (the clock is the obstacle, registered); frozen latents 1/8 vs 7/8; hybrid 16/16; CARoPE cannot cancel (proved) | **SIGNSEP** separates route from barrier for sign (four cells from one start). Whatever it returns, the main claim stands; Sec. 7.3's "why" paragraph changes | Cut, do not add. Demote Conjecture 1, hiding and parking to one remark plus appendix; keep in the main text only proved statements and registered causal tests. "We identify a route the signed model uses" is supported; "we identify the reason" is not |
| A5 | **Synthetic, tiny, and MapFormer already showed it helps; the words are a scripted torus; on ordinary text it equals RoPE (n = 3). What does this teach about models people use?** Plus: no recurrent baseline, and an RNN integrates trivially | floors (MapFormer's own benchmark is 93% map-free), SRoPE/CoPE/CARoPE comparisons, the clock-ratio diagnostic on text; TinyStories pilot | none | For TMLR: an LSTM/GRU torus baseline at T = 1024 (~8 GPU-h; it must store a redrawn map in its state, so the outcome is not obvious). FACTORIAL to n = 8 (+4 seeds, ~8 h) so the headline "more than four" rests on more than complete separations at n = 4. Scope sentence: "at our scale" on every text claim |

Secondary attacks to pre-empt in the text: the shared-latent deviation from MapFormer (say it at every rank comparison);
Holm 0.054 on rank 3; the analysis numbers have one batch per arm and no fresh-seed replication; the text-world map
evidence is a 13.4% stratum; the CANCEL primary landed in no branch (the exchange rate is a declared secondary).

---

## 4. One paper or two

**Two, plus a CCN abstract. Not a navigation/learnability split.**

The navigation half without the learnability axes is a measurement note (MapFormer owns "helps"; ours would be floors,
exchange rates and words), and the learnability half without the navigation regime loses the task that makes the clock
visible. The main paper must carry both, as draft 2 does. The sign line, however, is self-contained, dense (ten registered
batches, its own existence constructions, the SSM arm) and speaks to a different audience (Grazzi / Shakerinava / Mamba-3
readers), and it already exists as `DRAFT6_workshop.md`.

| paper | content | venue | why |
|---|---|---|---|
| **Main** | draft 2 trimmed: floors and strata; one layer vs depth; sign and rank as search costs with the registered causal tests (DF, frozen latents, hybrid, SSM); words; theory cut to proved statements and one remark on conjectures; PoPE, what/where, readers-vs-codes in appendices | **TMLR** (mid-December target is realistic for text; see Sec. 5 for what must land first) | no page limit; the venue rewards exactly what this project has (pre-registration, floors, honest scoping) and tolerates small models; rolling, so a January submission costs nothing |
| **Sign companion** | draft 6 cut to 4 pages: representable (Prop. 3), gauge (Lemma 1), the cost at matched length and best rates, exists/held/re-found, hybrid readout, the SSM result, SIGNSEP's reading if it lands cleanly | **ICLR 2027 workshop** (~1 Feb; Sci4DL-type if it recurs, else ICBINB/NFAM), non-archival; CPAL Spotlight (18 Jan) as a parallel option | the genre (controlled falsification of a learnability hypothesis) fits; non-archival keeps TMLR clean; the SSM result makes it a sequence-model paper, not a navigation one. Check each venue's dual-submission text |
| **CCN abstract** | the cognitive-map framing: Howard's three cases as rotary rates, what/where separation as a property of the solution, the clock as the obstacle to a map, maps from words | **CCN 2027** 2-page (1 Apr) or 8-page (4 Feb) | the only audience for which TEM/Howard is the core; cheap; the 2-page track collides with nothing |

Double reporting between main and companion is acceptable because the companion is non-archival; the main paper cites
it. If the ICLR workshop list (29 Nov) has no fit, send the companion to CPAL Spotlight and fold its content into TMLR.

---

## 5. Experiment triage

Each item is marked by what it serves. "Drift" means it came from a post hoc reading of the last batch and serves a
sentence, not the claim (`feedback_follow_on_drift.md`).

| item | status | serves | verdict | what it could change |
|---|---|---|---|---|
| **G-4 torus best-rate sign sweep** | in flight | main claim 2 on the task where the accuracy numbers live (attack A2) | **needed**; let it finish | if |.| at 3e-3 or 1e-2 solves >= 4/8 on the torus, the sign cost becomes "ring + speed on the torus": the abstract's torus sign sentence is rewritten and the companion keeps its ring title. Do not extend the batch (pilot said no) |
| **SIGNSEP** | in flight, ~07:00 | mechanism of claim 2 (attack A4); companion paper's "why" | **nice-to-have for main, needed for the companion's mechanism paragraph** | W solves and F fails -> route account; F solves and W fails -> barrier; both fail -> both needed; both solve -> neither account as stated. The main claim is unchanged in every cell; Sec. 7.3 and Prop. 4 wording change |
| **RANKDECAY (koopman)** | in flight | the wd qualifier on claim 3 | **borderline drift**; let it finish, report in Limitations, do not chain further | if decay on omega alone suffices, the limitation becomes a mechanism sentence ("decay pulls the ladder to low frequencies"); if not, the qualifier stays "decay on everything" |
| **G-3 block A2 (exp at 1e-3, SSM)** | queued, ~2 h | attack A3; the SSM statement about the parameterisation people use | **needed, cheap** | exp solves -> "the cost is specific to |.|/softplus starts, exp learns" (a scope the paper can state); exp fails cleanly -> the SSM sentence covers the LRU parameterisation |
| **Rank-2 recipe search** (not in the plan; my addition) | not built | attack A2 on claim 3 | **needed for the rank claim** (~24 runs, ~8-10 h) | if rank 2 solves at its own recipe, "rank D + 1 is needed" becomes "rank D is harder at the default recipe"; the DF result (clock is the obstacle) survives either way |
| **FACTORIAL + 4 seeds** | not built, ~8 h | attack A5, the n = 4 headline | **nice-to-have, cheap** | none expected; it buys a p-value floor below 0.029 |
| **LSTM/GRU torus baseline** (G-10 part) | not built, ~8 h | attack A5 for TMLR | **nice-to-have**; do it if TMLR is the target and time allows | an LSTM that solves the torus at T = 1024 reframes "why attention"; one that does not strengthens the one-layer result |
| **G-2 torus part** (frozen move latents on the torus) | not built | mechanism breadth | **drift** unless SIGNSEP returns the route account and the ring-only scope becomes the objection | -- |
| **G-5 depth routes** | not built, ~27 h | the index-attention mechanism (Sec. 6 analysis) | **drift for this paper**; a different paper's question | -- |
| **G-7 long text world** (4096 words) | not built, 20-35 h | removes the retrace 0.934 caveat on claim 4 | **nice-to-have, expensive**; the NEEDS-MAP stratum already carries the claim | -- |
| **G-8 rank x sign x passes** | not built, ~55-74 h | "readers cannot repair a code", registered | **drift** | -- |
| **G-9 other rank follow-ups** (seeded escape, 3D rank 5, ring at T = 1024) | not built | mechanism | **drift** | -- |
| **G-10 rest** (egocentric, scale, index to 8 layers) | not built, 80-90 h | breadth | **drop for submission** | -- |
| **G-11** | -- | appendix | **drop** | -- |
| G-7b double dissociation, TW_AMBIG, TW_LANDMARK, RANK_NOWRAP | stopped / postponed | -- | **keep stopped** | -- |

**Order.** (1) Fold in G-4, SIGNSEP, RANKDECAY as they land; (2) G-3 A2 (2 h); (3) the rank-2 recipe search (8-10 h);
(4) FACTORIAL + 4 seeds (8 h); (5) LSTM baseline (8 h) if TMLR; then stop. About 26-28 GPU-hours beyond what is running.

**Stopping point.** "Enough" is: every main-text contrast registered at n = 8 on the task it is quoted for, the recipe
objection answered on that task for both axes (G-4 for sign, item 3 for rank), the SSM statement clean for exp, and the
theory section reduced to proved statements plus registered causal tests. After (1)-(3) the companion is submittable; after
(1)-(4) the main paper is. Everything after that is revision material, to be run only if a reviewer asks.

What to stop doing: chaining a new mechanism batch off each result's descriptive table (RANKDECAY is the current instance),
building more clock-exit variants, and polishing analysis numbers at the fourth decimal. The paper's weakness is scope and
recipe dependence, not precision.

---

## 6. After this paper

The line points at one question that is larger than positional encoding: **when does a parameterisation that removes no
function from a model's class still cost gradient descent the solution, and can the cost be predicted from the geometry
of the step map alone?** Both axes here are instances. A non-negative increment keeps every map code (rotations wrap) and
loses the path through zero; rank D keeps the rank-D map and loses the spare direction a clock can be parked in. Larson's
2026 rank law says gradient descent recruits exactly the dimension the task's algebra demands; ours says that on a sparse,
indirect loss it needs one more to get there. Shakerinava's non-Abelian gap, Howe's parasitic attractors and the Dale's-law
results (Li et al. 2023: the spectrum at initialisation, not the sign constraint, decides) are the same question in other
coordinates. A theory that takes a step map's class (signed or not, rank, group), a loss (dense state supervision or sparse
revisits) and a start, and returns which codes training finds and by which route, would unify them; the route inventory
and the clock/map decomposition are a first vocabulary for it, and the rule-14 design (construct, install frozen, install
trainable, train from scratch) is the measurement protocol. The neuroscience version is sharper still: a map code is a
phase that an out-and-back returns to zero, a clock is one it does not, and the brain's integrators have to remove their
clock too; whether the one spare dimension that makes removal cheap in these models has a counterpart in grid-cell modules
(more modules or dimensions than the space being coded) is a testable question, and the natural CCN follow-up.
