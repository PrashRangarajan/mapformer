# The paper as a whole: verdict, framing, additions, experiments, venues (2026-10-20)

Scope: `docs/paper/MAIN_DRAFT1.md` (+ `MAIN_APPENDIX1.md`, `APPENDIX_theory_v3.md`) read as a reviewer for TMLR, a strong
ICLR/NeurIPS workshop and CCN. Sources read: the outline v2, `DRAFT6_workshop.md`, `review_draft5/` (mock review,
synthesis, prior art, repro audits), CLAUDE.md's citable table and rank section, `docs/WHERE_THINGS_STAND.md`,
`VENUES_2026-10-15.md`, `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`, and for the EM question `EM_WM_STATE.md`,
`SEARCH_RESULTS.md`, `PAIRSPLIT_RESULTS.md`, `RECENCY_EM_RESULTS.md`, `WARM_RESULTS.md`, `NOLEAK_RESULTS.md`,
`SPREAD_RESULTS.md`. CPU only; no file other than this one was written; nothing committed. Every number below is copied
from the draft or the file named beside it. GPU costs are estimates scaled from measured timings (Sec. 4, "calibration").

## 0. Top three recommendations

1. **Make one claim and organise the paper around it.** The claim: *in content-dependent rotary models, per-head rank
   and the sign parameterisation of the step decide whether the map is found, not whether it can be represented.* The
   advantage results (floors, exchange rate, words) then show why finding the map matters: index depth does not buy it.
   Use revisit strata (short-gap / long-gap / wrap-only) as the measuring tool from Sec. 3 onward, not as a late Sec. 6.3.
   New title, abstract and contributions in Sec. 2.
2. **Run the sign test outside attention.** Use a one-layer diagonal complex SSM (Mamba-3-style input-dependent rotation)
   on Z_m tracking with inverse moves, with signed, |.|, softplus and exp (LRU-style) phase steps, plus an exp arm in the
   attention ring test. About 6-10 GPU-h and 1-2 days of code and checks. This is the one experiment that turns a
   MapFormer study into a claim about content-dependent phases, in the literature (Grazzi, Mamba-3, Shakerinava) that
   so far asks only what such phases can represent.
3. **Close the torus learning-rate gap.** On the torus, signed vs |.| has been run at one learning rate only (and rank
   too). Fix: each arm at its best of three rates, n = 8, about 15 h wall (sign only) or about 25 h (sign + rank). Every
   reviewer raised the learning rate. The ring answer (2/8 vs 8/8 at best rates) does not yet cover the task that
   carries the accuracy evidence.

EM's recency deficit (question 2) is **not** a clean third case of "representable but not found". It fits the
pattern only after changing the model, and a larger budget mostly closes it (Sec. 3.2). Use it as a contrast in the
Discussion and an appendix, not as a third axis.

---

## 1. Verdict on Draft 1

**Overall, as a reviewer.** TMLR: accept with minor revisions. The claims match the evidence, the controls are
unusually careful, and TMLR weighs correctness over significance. Strong ICLR/NeurIPS workshop (Sci4DL type): accept,
but only as the 4-page sign paper; Draft 1 is three times that length. CCN 8-page: borderline. The cognitive-map
content (words, what/where, rank D+1) is there, but the core of the paper is ML learnability, and CCN reviewers will
ask about egocentric actions and neural data. ICML/NeurIPS main track: weak reject as is. The reasons would be one
model family at 1-4 layers, d 128-256 and 2 heads, all tasks synthetic, the sign result only in attention, and no
recurrent baseline. Each of these has a costed fix (Sec. 4).

**The single most compelling story.** Path-integrated attention does something depth does not: one path layer holds a
1024-move torus map that four index layers or a looped block do not (0.860 / 0.864 vs 1.000). Each extra index layer
buys only short-gap revisits (to 0.94); long-gap and wrap-only revisits stay at the constant floor. So the map has to
come from the step map, and whether training finds it depends on two choices that do not limit what the model can
represent: per-head rank and the sign of the step. In both, the solution exists, training keeps or re-finds it when it
is installed, and from scratch it is not found. Every failing model, index or path, ends with the same code: local
revisits served, long-gap and wrap-only revisits at the floor. No other paper has this story. MapFormer owns "path
integration helps"; Grazzi and Mamba-3 own representability; Shakerinava owns "optimisation, not expressivity" for
non-Abelian SSMs (`PRIOR_ART.md`: the learnability cost of a non-negative rotary step and Lemma 1 were not found
anywhere).

**Is the structure delivering it? Partly.**

- *Two papers in one.* Title and abstract close on "optimisation, not expressivity", but Secs. 4, 5 and 7 (about 1,350
  words body) never refer to it. The connection is implicit: depth does not substitute, so the map must be found. The
  introduction should say this in one sentence, and Sec. 5 should end on it.
- *The best tool is introduced too late.* Strata (short-gap / long-gap / wrap-only) are the paper's clearest way to
  tell a map from a local rule. They appear in Sec. 4, return in Sec. 5 and are unified only in Sec. 6.3 (129 words).
  Define them once in Setup, and make Fig. 2 the figure the paper keeps returning to. Sec. 6.3 can then be cut to one
  sentence closing Sec. 6.
- *The two failure axes are presented as parallel, but their evidence is not.* For rank, the solution is held (7/8) and
  2x budget leaves 0/8, but there is no learning-rate test. For sign, the solution is held on the ring, perturbed and
  re-found on the torus, and the best-rate test is ring only. Neither has a budget test on the torus with an in-batch
  control for both arms. A reviewer will build this table; build it first (Sec. 2.4) and let the experiments in Sec. 4
  fill its gaps.
- *Sec. 6.2 dominates.* It has 779 words of body, 684 words of captions and tables, and three of seven figures. Fig. 6
  (the mechanism) can move to the appendix in an 8-page version, with one sentence left.
- *Background (678 words) is long for what it carries.* The Howard paragraph is used once in the Discussion (mock review
  W8). Sec. 2.4's list of orientation-fixing parameterisations belongs in Related Work at the end, or in the Discussion.
- *The protocol sentence over-generalises.* "Every verdict was pre-registered" sits beside the ring exchange rate,
  which is a declared secondary (the registered CANCEL primary landed in no branch; `CANCEL_RESULTS.md`), and beside
  "(analysis)" floors. Change it to "every contrast in Tables 1-3 and Figs. 1, 3-5 was pre-registered; the rest is
  marked". Check the figure list against App. A.
- *The what/where paragraph uses the wrong regime.* The checks ran on the T = 128 paper-torus checkpoints
  (`docs/WHAT_WHERE_CHECKS.md`), where the paper's own Sec. 4 argues a map is not needed (retrace 0.934). Re-run them on
  the stored T = 1024 checkpoints (eval only, CPU; Sec. 3.1), or move the paragraph to the appendix.
- *Positioning against MapFormer.* Mostly right. Say once, early, that MapFormer already shows path integration helps,
  and that this paper's contributions are the floors, the matched-depth rate, the learnability axes and the words.

---

## 2. Improved title, abstract, contributions

### 2.1 Title

Recommended: **Representable, kept, but not found: what decides whether path-integrated attention learns a map**

- "Kept" names the leg that separates this work from a plain optimisation failure (installed solutions survive
  training). "What decides" names the two variables. It does not undersell Secs. 4-5, because the abstract opens with
  them.

Alternatives:
- *Depth does not buy the map, and training must find it: path-integrated rotary attention.* This puts the advantage
  first. Use it only if the torus depth series is extended (Sec. 4, E8); "does not buy" is otherwise measured only up
  to four layers.
- *Rank and sign are learnability variables: path-integrated rotary attention learns maps that index depth does not.*
  For an ML main track once the SSM generalisation (E1) has landed; then retitle to "content-dependent phases".

### 2.2 Abstract (about 215 words; for the current evidence, no new experiments)

> Content-dependent rotary position encodings let each token add a learned step to a running phase; when the steps
> are driven by actions, attention compares positions on a map. We study what path-integrated attention (MapFormer)
> learns and what decides whether training finds the map. Against measured map-free floors at matched length, one
> path-integrated layer solves 1024-move torus navigation (0.998, 8 of 8 seeds), while index RoPE stays below a rule
> that retraces the last run of moves (0.731 against 0.843), and four index layers or a looped block lift only
> short-gap revisits (0.860 and 0.864). At matched depth one path layer is worth about three index layers on Dyck-2
> and on a biased 1D walk, and the map is learned from navigation told in English (0.979 against 0.505), with opposite
> direction words cancelling. Two choices in the content-to-angle map decide whether it is found: its per-head rank
> (rank 2: at most 2 of 8 seeds; rank 4: 8 of 8) and the sign of its step (non-negative |.|: 0 of 8; signed: 8 of 8).
> In both, the constrained model represents the map exactly, and training keeps or re-finds an installed solution;
> from scratch the gap persists at twice the budget (rank) and at each parameterisation's best learning rate (sign).
> Every failing model, index or path, ends with the same code: short-gap revisits served, long-gap and wrap-only
> revisits at the floor. Rank and sign are learnability variables, not expressivity variables.

If E1 lands positive, replace the last sentence: "The same cost of a non-negative phase step appears in a diagonal
complex SSM tracking Z_m, so it is a property of content-dependent phases, not of attention."

### 2.3 Contributions

1. *Measuring map use.* On MapFormer's 128-move walks a map-free retrace rule scores 0.934. We measure at 1024 moves
   (retrace 0.843) with revisits split by stratum (short-gap / long-gap / wrap-only); only long-gap and wrap-only
   revisits show a map. There one path layer scores 0.998 and index RoPE 0.731 (8/8 vs 0/8 SOLVED, p 0.0002).
2. *Depth does not substitute on a 2D map.* At matched depth, one path layer is worth about three index layers on
   Dyck-2 (+0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers) and on a biased 1D walk. On the torus it is worth more than
   four: index depth and a looped block lift short-gap revisits to 0.94 and leave long-gap and wrap-only revisits at
   the floor (0.46-0.51).
3. *Two learnability variables.*
   - Rank: with per-head rank 2 the torus is solved on 0-2 of 8 seeds (rank 3: 6/8; rank 4: 8/8; Fisher p 0.0070 at
     matched sharing). A rank-2 solution exists (0.9955) and training keeps it (7/8).
   - Sign: a non-negative step represents the map exactly (Proposition 1) and fixes only relative signs (Lemma 1).
     Training keeps an installed solution (ring 8/8) or re-finds it (torus 8/8), yet from scratch |.| solves 0/8
     against 8/8 for signed steps. The gap holds at each parameterisation's best of four learning rates (2/8 vs 8/8,
     p 0.007).
   - Signed steps reach the map collectively, through zero: freezing the move latents gives 1/8 vs 7/8.
4. *Maps from words.* One path layer learns the torus told in English (0.979 vs 0.505, 6/6 vs 0/6 fresh seeds,
   p 0.0022), learns which words move it, and cancels opposite direction words on every seed.

The CoPE/CARoPE results (CoPE can represent the ring map yet fails it; CARoPE provably cannot cancel) go inside
contribution 3, as "published monotone phases fail too, for a learnability (CoPE) or a representational (CARoPE)
reason". They are not a separate contribution.

### 2.4 The evidence table the paper should carry (new; CPU, from existing files)

One row per axis and task, one column per leg. Cells give the status from the results files; "--" means not run.

| axis, task | exists (frozen) | kept / re-found when installed | from scratch, 1x | 2x-3x budget | best LR |
|---|---|---|---|---|---|
| rank 2, torus | 0.9955 (`RANK_PROJ`) | kept 7/8 | 0-2/8 vs 8/8 | 0/8 vs 7/8 at 1800 ep (5/8 still descending) | -- (E2) |
| \|.\|, ring | exact (Prop. 1; `WRAP_EXISTENCE`) | kept 8/8 | 0-2/8 vs 8/8 | 3/8 at 900 vs 3/8 at 300 ep, no in-batch signed arm (`RING_BUDGET`) | 2/8 vs 8/8 (`LRSWEEP`) |
| \|.\|, torus | exact | re-found 8/8 (`GENERALITY` E2) | 0/8 vs 8/8 | -- | -- (E2) |
| softplus, torus | exact (Prop. 1f) | -- | 1/8 vs 8/8 | -- | -- |

The empty cells are what reviewers will ask about. E2 fills the LR column; E6 fills the budget column at scale.

### 2.5 Structure and page budget (8-page version; TMLR adds the bracketed items)

| section | pages | change from Draft 1 |
|---|---|---|
| 1 Introduction | 0.9 | one sentence joining the halves: depth does not buy the map, so the step map must find it |
| 2 Background | 0.5 | keep 2.1-2.3 and 2.5; move 2.4 to a closing Related Work (TMLR) or cut to two citations |
| 3 Setup, floors and strata | 0.7 | define strata here; protocol sentence scoped |
| 4 Measuring map use (Table 1) | 0.7 | as now, shorter |
| 5 Depth does not substitute (Figs. 1, 2a) | 0.9 | end on the bridge sentence |
| 6 Learnability: rank, sign (Table 2, evidence table, Figs. 3, 4, 5) | 2.4 | Fig. 6 to appendix [TMLR: keep]; 6.3 cut to one closing sentence; add SCORE_RANK and LOOP_RANK sentences (Sec. 3.1) [+ E1 SSM paragraph 0.3] |
| 7 Maps from words (Fig. 7) | 0.5 | clock-seed detail to App. J |
| 8 Discussion and limitations | 0.6 | EM contrast (2 sentences); egocentric-action boundary (2 sentences); TinyStories moved to limitations as a pilot |
| total | 7.2 + refs | leaves 0.8 for figure slack |

---

## 3. Existing results: add, keep out

### 3.1 Add

| result | where | why it strengthens | status / caveat |
|---|---|---|---|
| SCORE_RANK: PoPE's score leaves rank 2 at 3/12 SOLVED vs MapWM 3/12 at T = 1024 (rank 4 8/8 in both) | Sec. 6.1, one sentence; App. L | Shows the rank failure lives in the step map, not the score rule. Rules out the obvious alternative account | registered, NO RESCUE |
| LOOP_RANK search aids: rank 2 + loop x4 2/8 (0.973), 4 real layers 5/8 (0.990), plain 0/8 (0.894), rank 4 8/8 | Sec. 6.1, one sentence; App. G | More search (depth) partly recovers rank 2, as a search account predicts; three loss regimes | registered, verdict UNMEASURED; depth adds 4x params, so not "search at constant capacity". Say so |
| What/where checks re-run on the T = 1024 checkpoints (SIGN_MATCHED, RANK_SEP: path, index, \|.\|, rank 2) | Discussion + App. K | Moves the TEM "where apart from what" claim into the regime where a map is needed. Tests whether failing models also lose separation | new eval-only CPU job (~1-2 h); the current checks are T = 128 only, where retrace scores 0.934 |
| Rotation actions and allocentric recoding (+0.050 -> +0.488, 8/8) | Discussion, as a boundary (2 sentences) + appendix | The scope of an Abelian (cumsum) phase: egocentric turn/forward actions need a heading-dependent step; recoding to allocentric moves restores it. This is the head-direction / grid-cell link a CCN audience expects | 16-epoch recipe, index arm on the floor, not registered. Cite as a boundary only, or rerun (E5) |
| EM recency, as a contrast case | Discussion (2 sentences) + App. M (one table) | Sharpens what "search failure" means; see 3.2 | registered pieces; the 4x-budget contrast is not registered |
| MapFormer v4 Table 2 reproduction | already App. A4 | keep | -- |

### 3.2 EM's recency deficit: a third case of "representable but not found"? Not a clean one

The three legs, from the files:

- **Exists: yes.** Single-p0 MapEM with the rewind installed and frozen scores 1.000 on 8/8 at T = 1024 and 2048
  (`WARM_RESULTS.md` W1, registered).
- **Kept: only after changing the model and the install scale.**
  - Installed at the 1/64 scale, the trainable model loses it (0.642).
  - At 8x the scale it reaches 0.941 (4/8 >= 0.95).
  - It is fully kept (1.000, 8/8) only when `w_in`'s content columns are held at zero, a restricted model
    (`NOLEAK_RESULTS.md`). Rank and sign were kept in the same class, within about 5x of trained scale.
- **Not found from scratch: partly.**
  - From scratch P0 scores 0.600, 0/8 to loss < 0.5 (`RECENCY_EM_RESULTS.md`).
  - But `SEARCH_RESULTS.md` shows EM does find the rewind, per query token and wrapped, for about half the offsets k.
  - With one k it finds a 63-symbol rewind on 7/8 seeds (0.985), faster than WM.
  - The obstacle is queries per token (`SPREAD2`). Four times the budget moves the 64-offset task from 0.578 to 0.928
    (6/8 near 1.000; `SPREAD_RESULTS.md`, an arm of a registered batch, but the e1200-vs-e300 contrast was not
    registered).
- **Other differences from the main paper.**
  - The failing model is single-p0 MapEM, an ablation. Paper-faithful MapEM scores 0.837.
  - The task is k-back recency, a clock task, not a map.
  - The line carries a withdrawal ("0/40 never found", withdrawn by SEARCH).

**Verdict.** It belongs to the same family: the solution is representable, and training from scratch is the problem.
But it is a different kind of search failure. More data or budget mostly closes it (exposure per query token). Rank,
by contrast, stays at 0/8 at twice the budget, and |.| on the ring stays at 3/8 at three times. As a third main-text
axis it would add a model, a task and a mechanism, and weaken "kept". Use it as a **contrast** that defines the main
claim more sharply:

> "Not every search failure is of this kind: single-origin MapEM represents k-back recency exactly and finds it per
> query token, and four times the budget closes most of its deficit (0.578 -> 0.928), whereas rank 2 stays at 0/8
> with twice the budget."

Caveat for that sentence: at 1800 epochs 5/8 rank-2 runs were still descending, so "stays" is budget-scoped.
Appendix M: one table with the WARM, NOLEAK, SEARCH and SPREAD rows.

### 3.3 Keep out (dilutes, or is not citable as needed)

| result | reason |
|---|---|
| What-to-where leak and NormStep (+0.0107) | Effect about 1%, partly training speed (r(loss, acc) -0.94); the accuracy gain did not replicate on fresh seeds (+0.0045, p 0.46); off-thesis |
| Loop on Match-Query (+0.346) | Cross-batch and asymmetric; per-seed drift 0.185 across batches; checkpoints missing (`REPRO_AUDIT_nonsign.md` note 8). The torus factorial already carries the loop |
| Match-Query "necessary for in-context maps" (0.730 vs 0.154; destruction 0.918 -> 0.074) | Not registered; checkpoints and the destruction script are missing, so it cannot be reproduced. Cite only after a registered rerun (E7) |
| Map-extent threshold (-0.010 / +0.015 / +0.305) | Not registered, n = 5, measured noise floor 0.150, index arms still descending. Its message (small maps need no map) is already carried, better, by the retrace floor. At most one appendix line |
| CSCG stitching control (+0.131) | n = 3, a probe, not a capability result |
| Parallel-scan timing (2.6-3.3x) | Engineering, and MapFormer's claim; one Setup clause at most |
| MapPoPE pair / gain grain / gain phase | Paper torus T = 128 at ceiling, or about 1% effects; App. L already has what the Dyck pairing needs |
| TW_STATECHANGE | Its strongest reading (state semantics) is withdrawn; what remains (location + recency) adds little to Sec. 7 |
| TinyStories "token clock" | Pilot, n = 3, with withdrawn sub-claims; one limitations sentence, labelled pilot |
| Clock/map crossover (-0.280 / -0.004) | Out of distribution on both tasks |
| Code / Bach / Indirect replications | Off-thesis |

---

## 4. Experiments that would raise the paper's level, ranked

**Calibration.**
- 1-layer torus at T = 1024 (2,048 tokens): about 2 s/epoch alone, and about 10 s/epoch at 4 jobs per GPU under load
  (`RANK_NOWRAP_PREREG.md` pilot timing). A 900-epoch run is therefore about 2.5 slot-hours, with 8 slots on the two
  4090s.
- 2-layer d 256 torus: 3.6 s/epoch, GPU-bound at one run per GPU (`ROBUST_PREREG.md`).
- 4-layer index torus: 7.4 s/epoch alone (`LOOP_RANK_E1800_PREREG.md`).
- Ring (32 cells, T = 128, 300 epochs): 64 runs take about 5.3 GPU-h, about 2.7 h wall on two idle GPUs.

**Every batch below** needs, per rules 7, 11, 12 and 29, a pre-registration, a CPU task gate, pilots on seeds outside
the batch and a results-blind verification agent. Add about half a day of CPU work per batch. The user is
cost-sensitive and shares the GPUs: check `nvidia-smi` owners before launching.

| rank | experiment | design | cost | claim it unlocks | risk if it goes the other way |
|---|---|---|---|---|---|
| **1** | **Sign beyond attention: diagonal complex SSM** | One-layer diagonal complex recurrence h_t = exp(i omega Delta(x_t)) h_{t-1} + B x_t, readout from Re/Im. Task: Z_m position tracking with +1 / -1 / stay tokens (m = 8 and 32; T = 128-512; label = current position). Steps: signed, \|.\|, softplus, exp. 8 seeds. Existence construction per Prop. 1 (exp and softplus: annihilated tokens step exp(0) = 1 or ln 2, made a full turn by omega). Prediction from attention: non-negative steps rarely learn from a small start; a past-pi start rescues | about 64-96 tiny runs, about 5-8 GPU-h, about 3-4 h wall; 1-2 days of code (new model; `model_baselines_extra.MambaLikeBaseline` exists from the withdrawn lm200 era, but has no input-dependent phase) | The learnability cost is a property of content-dependent phase parameterisations, shown in both attention and SSMs, on the state-tracking benchmark where Grazzi and Mamba-3 prove only representability. Moves the paper from "a MapFormer study" to the SSM/RoPE literature; answers mock review W5 ("no one uses \|.\|"), since exp is the LRU parameterisation | If \|.\| learns fine in the SSM, the cost is attention-specific: a scope boundary worth one paragraph, and an informative contrast (the recurrence gives gradient through time) |
| **2** | **Torus learning-rate sweep (sign, + rank)** | Observation-step-free torus, signed vs \|.\| at lr 1e-3 / 3e-3 / 1e-2, n = 8 fresh seeds, selection-aware best-rate test copied from `LRSWEEP`. Optional rank arms: shared r2 vs r4 at 3e-3 and 1e-2 | sign only: 48 runs, about 120 slot-h, **about 15 h wall**; + rank: 80 runs, about 25 h wall; trimmed (signed 2 rates, \|.\| 3): 40 runs, about 12.5 h | "The cost holds at each arm's best rate" on both tasks, for sign and (optionally) rank. Removes the limitation every reviewer raised; fills the evidence table's LR column | If \|.\| catches up at 1e-2 on the torus, the torus claim becomes "needs larger steps". Still publishable, but the abstract's "best learning rate" becomes ring-only |
| 3 | **Exp (and sigmoid-rate) steps in attention** | Ring, observation-step-free; signed, \|.\|, softplus, exp at each arm's best of 3 rates, n = 8. Can share a batch, code gate and verification agent with #1 | about 72-96 ring runs, about 6-8 GPU-h, about 3-4 h wall | "Every non-negative parameterisation we tried", including the one used in the literature (LRU exp) | exp may behave like softplus (likely); little downside |
| 4 | **Recurrent baseline on the torus** | LSTM/GRU, 1 and 2 layers, d 128 / 256, T = 1024, n = 8. `model_baselines_extra.LSTMBaseline` exists but was used only in the withdrawn lm200 era: re-verify it | about 16-24 runs, about 6-10 GPU-h (cuDNN LSTM is fast; untimed at 2,048 tokens, so pilot first) | Answers "why attention?". Expected: a recurrent model can integrate but must store the map (16 objects over hundreds of cells) in its state, so long-gap revisits fail for memory. That separates path integration from map storage, the TEM "where" vs "what" in architecture | If an LSTM solves it, the paper must say path attention is one of two solutions. Honest, and it reframes Sec. 5 as "versus index attention" |
| 5 | **Egocentric actions, registered, at T = 1024** | Torus T = 1024; path MapWM with {translate, turn/forward commanded, turn/forward allocentric} + index RoPE (turn/forward); n = 8 | 32 runs, about 80 slot-h, about 10 h wall | The scope of an Abelian phase, measured where a map is needed: commanded turns break it, allocentric recoding (what head-direction cells supply) restores it. A CCN headline; an ML limitation turned into a result | If commanded turns are learned at T = 1024 (the old result was at 16 epochs), the boundary moves; report it |
| 6 | **Scale for sign and rank** | (a) sign, 2 layers d 256, 1800 epochs, n = 8 per arm: does the quarter-size gap (6/8 \|.\| still descending) close? (b) per-head rank 2 / 3 / 4 at 8 heads d 256 | (a) 16 runs at about 1.8 h each GPU-bound, about 15 h wall; (b) 24 runs, about 12 h wall; together about 27 h | "The variables still decide at width 256 / 8 heads / 2x budget", or an honest "at scale the sign cost is speed". Fills the evidence table's budget column; answers the "1-layer d 128" objection | (a) may close the gap: the sign headline would then be scoped to small models. Decide before submitting to a main track |
| 7 | Registered Match-Query rerun with destruction ablation | path vs index, n = 8, + context destruction; time a pilot (cost unmeasured; estimate 5-10 h) | 5-10 h (estimate) | Restores "the map is built in context" with a reproducible record; a second 2D task beside the torus | Low novelty: torus evaluations already redraw the map, so every number is transfer |
| 8 | Torus index depth to 8 layers, n = 8 | index 6 and 8 layers + loop x8, and L1 / L4 / path at n = 8 | about 16 deep runs at about 3.7 h each GPU-bound, about 60 GPU-h + about 8 h: **about 35-40 h wall** | Turns "more than four" into a number or "more than eight"; licenses "depth does not buy the map" | Expensive. If 8 layers solve it, the rate is 8, still a strong number |
| 9 | Real navigation data | -- | unscoped | -- | Not recommended now: Jericho was infeasible (6/57 games clean, near-trees); Talk the Walk is unscoped. The text world is the honest substitute |

**Cheapest set that changes the venue tier:** #1 + #3 in one batch (about 8-12 GPU-h) and #2 sign-only (about 15 h
wall), about two days of GPU in all. Add #4 (about 8 GPU-h) for any ML main-track submission. Add #5 for CCN. Then
#6, if a main-track reviewer is expected to press on scale.

**CPU-only additions (no GPU).**
- (a) The what/where checks on the T = 1024 checkpoints (Sec. 3.1).
- (b) The evidence table (Sec. 2.4).
- (c) Strata for the rank-2 + loop / depth arms (LOOP_RANK), to show that extra search lifts the long-gap stratum and
  not only short gaps.
- (d) An existence construction for exp steps (as for softplus), needed by #1 and #3 anyway.

---

## 5. Venue plan

All dates are from `VENUES_2026-10-15.md` unless marked. ICML / NeurIPS 2027 dates are **UNVERIFIED** (from past
editions); check them before planning.

| track | paper | venue, deadline | pages | condition |
|---|---|---|---|---|
| **A (recommended)** | full paper, Secs. 1-8 + appendix | **TMLR**, rolling; submit about mid-Dec 2026 after #2 (and ideally #1 + #3) | no limit; aim for 12-14 pages main (all 7 figures, mechanism kept, EM contrast, what/where at T = 1024) + appendix A-M | TMLR forbids concurrent archival submission; non-archival workshops are fine |
| A, parallel | sign short paper (Draft 6, cut to Secs. 4.1-4.3 + Lemma 1 + loss curves + existence figure) | **ICLR 2027 workshop** (Sci4DL-type; list 29 Nov 2026; papers about 1 Feb 2027) | 4 pages + unlimited appendix | Non-archival. Sci4DL asks that previously accepted work add new results; the SSM arm (#1) would serve |
| A, parallel | cognitive-map abstract: depth does not buy the map, maps from words, rank D+1, what/where (+ #5 if run) | **CCN 2027 2-page** extended abstract, 1 Apr 2027 | 2 pages | Non-archival, lightly reviewed; collides with nothing |
| A, parallel (optional) | same as TMLR | CPAL 2027 Recent Spotlight, 18 Jan 2027 | 250-word abstract + manuscript | Non-archival; concurrent submissions allowed |
| **B (reach)** | full paper reframed per Sec. 2.1, alternative 3 ("content-dependent phases") | **ICML 2027 main**, expected late Jan 2027 (**UNVERIFIED**) | 8 pages + unlimited appendix; use the 8-page budget in Sec. 2.5 | Only if #1, #2 and #4 land by early January. Fallback: NeurIPS 2027 (expected mid-May, UNVERIFIED) with #6 added, or TMLR |
| not recommended | full paper | CCN 2027 8-page proceedings (4 Feb 2027) | 8 | Archival (DOI in 2026), so it blocks TMLR/ICML for the same content. The ML learnability core is not CCN's centre. Use the 2-page track instead |

**Order of work.**
1. CPU now: reframe the text (Secs. 1-2 here), build the evidence table, re-run what/where at T = 1024, and add the
   SCORE_RANK / LOOP_RANK sentences.
2. GPU next: #1 + #3 in one batch, then #2 sign-only.
3. Decide A vs B in the first week of January, on #1's outcome. If the SSM shows the cost, B is worth trying. If not,
   the SSM result is a scope paragraph and A is the plan.

**Page budgets by venue, in brief.**
- TMLR: 12-14 pages main.
- ICML: 8 pages, per the table in Sec. 2.5.
- Workshop: 4 pages, sign only.
- CCN: 2 pages, three claims and one figure (the exchange rate or the text world).
