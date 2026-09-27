# Where things stand -- 2026-09-27

Orientation for a fresh session. Read `.claude-memory/project_state.md` first (what is running,
what the user must decide), then this. `CLAUDE.md` holds the conventions, the citable table and the
withdrawal list and is the authority on all three; this file is the shape of the project around them.

## The thesis, as it now stands

This began as a reproduction of Rambaud et al.'s MapFormer -- a transformer whose rotary angle is a
path-integrated, content-dependent phase rather than the token index -- and turned into a study of
when a learned "where" separates from the "what". What the project has actually established is
mostly a negative with a sharp edge: **nearly every positional-encoding effect measured here turns
out to be robustness to distribution shift rather than capability, and it closes once the training
distribution is matched on the axis the claim is about.** Code extrapolating past 512 gave -3.694
bpc; trained at 2048 it is -0.0030, unmeasured, and the composition claim reverses. Dyck at 4
layers gave +0.168 at nesting depth 12; trained at depth 12 every arm is at ceiling and the effect
is +0.002. Length was not enough -- depth had to be matched too, and the general form is: match the
training distribution on *every* axis the task varies, or you are measuring extrapolation.
Three positives survive that test: navigation on the torus at training length, depth-substitution
on Dyck, and the rank result, which is the one place a matched-distribution control made the effect
*sharper* rather than killing it. The published taxonomy and the content-dependent rotation itself
are not ours (GRAPE 2512.07805, Mamba-3 2603.15569, Selective RoPE 2511.17388); what is ours is
empirical -- the rank of the content-to-angle map, and the navigation regime.

## What survives

Numbers verified against their results files on 2026-09-27. `CLAUDE.md`'s citable table is the full
list with scopes; these are the load-bearing ones.

| result | numbers | file |
|---|---|---|
| Path integration helps on the torus **at training length** | converged recipe: position **+0.243** (MDE 0.038, 8/8); index RoPE 0.805, path 0.971; grows to +0.359 at 8x length | `PAPER2X2_RESULTS.md` |
| ...and is necessary for in-context maps | Match-Query 0.730 +/- 0.247 vs index 0.154, chance 0.0625; context destruction 0.918 -> 0.074 | `MATCH_QUERY_SCALE.md`; the destruction pair is in `MATCH_QUERY_RESULTS.md` |
| Dyck: path integration is worth ~3 layers of attention, **at matched depth** | trained AND tested at L32 D12: +0.353 / +0.130 / +0.045 / +0.024 at 1/2/3/4 layers (8/8 each, A2f, floor 0.594). Mixture training over D 4..12 keeps +0.110 at D12 | `DYCK_MDEPTH_RESULTS.md` |
| **Rank: it is the PER-HEAD rank of the content-to-angle map** | per-head rank 2 solves 0/8, 2/8, 2/8; per-head rank 4 solves 8/8, 8/8. Decisive contrast D - C_bd, Fisher and permutation p 0.0070. Sharing and `W_out` scale UNMEASURED | `RANK_SEP_RESULTS.md` |
| ...and it is SEARCH, not capacity | a rank-2 solution exists (0.9955 frozen, 8/8) and is held under training | `RANK_PROJ_RESULTS.md` |
| ...but search aids do not close it | r=2 + loop x4 2/8 solved / 0.973; 4 real layers 5/8 / 0.990; plain r=2 0/8 / 0.894; r=4 8/8 / 0.998. Registered verdict **UNMEASURED** | `LOOP_RANK_RESULTS.md` |
| Boundary: map extent, a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Boundary: rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8) | `KNOB_SWEEP_n8.md`, `H12_BUDGET_CURVE.md` |
| A shared block looped x4 helps path integration | Match-Query +0.346 unpaired (t 3.75); matches 3 real layers at 1/3 the params | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8); installed rewind frozen 1.000 | `EM_WM_STATE.md`, `SEARCH_RESULTS.md` |

Everything else in `CLAUDE.md`'s table is real but narrower. Everything in its **Withdrawn** list is
not to be cited, and the list is long for a reason -- most of it died to a control that was added
later, not to a mistake in the run.

## What is open, ranked

Ranked by what would change the story per GPU-hour. Costs are estimates from comparable batches.

1. **The three never-controlled OOD claims** -- the largest unpaid debts, because each is currently
   stated as robustness and would become capability if it survived. All three trained at T=128 and
   read at T=512/1024.
   - **Sign of the increment** (`SIGN_ABLATION.md`): signed beats index +0.123 / +0.195 at T=512 /
     1024 (12/12). At *training* length monotone scores 0.90-0.98 against index 0.80, so the
     accuracy cost is extrapolation-only; the matched-length evidence is in the loss. **10-20 GPU-h.**
     The most likely of the three to survive, and it is a replication of Sarrof / Grazzi / Selective
     RoPE in a new regime, so the payoff is a clean regime claim rather than a novel one.
   - **InEKF / Level 1.5 and the forget gate** (`L15_ABLATION.md`, `FORGET_CONTROL.md`): the filter
     line is already a live negative (stabilisation, not inference); the forget gate's +0.086 has no
     identified mechanism. **Low priority** -- run only if the mechanism question is revived.
   - **PoPE-wrapping**: never had a matched-length control. Cheap to bundle with the sign batch.
2. **Rank follow-ups** -- the one line where controls have been making the result *sharper*.
   - **Rank 3 per head** (~4 GPU-h). The current split is 2 vs 4 with nothing between it. If 3
     solves 8/8 the story is "rank 2 is special"; if it solves 0-2/8 it is "rank must exceed the
     task's degrees of freedom". Registered as untested in `RANK_SEP_RESULTS.md`.
   - **A longer budget for the H1 arms** (~6 GPU-h). Several `L` and `L4` runs in
     `LOOP_RANK_RESULTS.md` were still descending at 900 epochs, and the registered 0.05 cutoff sits
     inside their spread -- at 0.08 or above H1 would have passed. Extending the budget is the
     honest way to convert UNMEASURED into a verdict (rule 4: prefer extending the budget to
     convergence-conditioning arguments). Cheapest decisive item on this list.
   - **More loops** (8, 16) and the loop at rank 4 (~4 GPU-h). Tests whether the loop saturates.
   - Still unmeasured from `RANK_SEP_RESULTS.md`: cross-head sharing (D - C) and `W_out` per-entry
     scale (C_bd - B). Both need a design that moves one without the other; neither is cheap.
3. **H3, the "cancellation knob": a within-task dose-response** (~10 GPU-h, design sketched
   2026-09-27, not pre-registered yet). *Motivation:* this project's cross-task generalisations have
   failed repeatedly -- an effect established on the torus, transferred to Dyck or to code, has
   closed every time a matched control arrived. Within-task dose-response has not failed that way.
   So instead of asking "does path integration help on task X", vary the property that path
   integration is supposed to exploit, continuously, inside one task, and measure the response.
   *Design:* a walk whose steps are +/-1 with a bias parameter `p`. At `p = 0.5` the increments
   cancel in expectation and the accumulated phase is a genuine position -- the regime a signed,
   content-dependent phase is for. At `p = 1` every step is +1, the accumulator is the token index,
   and the model is a clock: index RoPE should match it exactly. Sweep `p` and measure the
   depth-substitution exchange rate (how many attention layers buy one layer of path integration,
   the +0.353 -> +0.024 curve of `DYCK_MDEPTH_RESULTS.md`) as a function of `p`. *What it would
   settle:* whether "signed = map, monotone = clock" (`project_clock_vs_map.md`) is a dichotomy or a
   continuum, and whether the exchange rate is a property of the task's cancellation structure
   rather than of the task. *Before launching:* it needs a pre-registration with branch boundaries
   set against the measured noise floor, an action-stream n-gram gate (rule 11), and a check that
   the readout is invariant to the sign and scale gauges (rule 8).
4. **The documents rewrite** (no GPU). See below. It is not research, but five documents currently
   assert things the results files contradict, which is how a retraction propagates.

## Known stale

The five `.tex` documents and `RESULTS_INDEX.md` were corrected on **2026-09-25**, in the pass whose
scripts are in `docs/prepared/tex_edits_2026-09-25/`. Both `DYCK_MDEPTH_RESULTS.md` and
`RANK_SEP_RESULTS.md` landed *after* that pass, and `LOOP_RANK_RESULTS.md` two days later, so all
six files still carry superseded claims. **Do not do this rewrite piecemeal** -- it is a
citation-by-citation pass, and `rep.py` (which refuses any edit that does not match exactly once) is
the tool for it. Rebuild each PDF from source afterwards (rule 27: no mid-line `%` comments in .tex).

Two errors recur and account for most of the work:

- **The Dyck framing.** Anywhere a document presents `+0.168` at 4 layers (or the `+0.290 / +0.209 /
  +0.159 / +0.168` ladder at L32 D12) as the language line's positive result, it is quoting depth
  extrapolation from models trained at D4. The replacement is depth-substitution at matched depth:
  `+0.353 / +0.130 / +0.045 / +0.024` at 1/2/3/4 layers, plus mixture training over D 4..12 keeping
  `+0.110`. Any sentence saying the index arms *plateau*, or that depth closes some fraction of the
  gap "and then stops", is also dead: at 3x budget they keep climbing and reach ceiling by 4 layers.
- **The rank caveat.** Every sentence of the form "per-head rank, cross-head sharing and `W_out`'s
  per-entry scale are not separated" is now false. They were separated on 2026-09-25: per-head rank
  fires (D - C_bd, Fisher and permutation p 0.0070); sharing and scale are individually UNMEASURED,
  which is a different and weaker claim than "unseparated". The five-arm table belongs wherever the
  three-arm one currently is.

Per-file list, to be filled in by the audit below. Line numbers are as of commit `414b60a`
(`positional_review.tex`, `axes_measured.tex`, `mapformer_math.tex`) and `542a5f0`
(`report/report.tex`, `report/report_short.tex`, `RESULTS_INDEX.md`); re-grep before editing.

Two structural facts shrink the job: the Dyck ladder numbers appear in **exactly one** of the six
files (`RESULTS_INDEX.md` line 48) -- `positional_review.tex` and `axes_measured.tex` contain no
Dyck content at all, and `mapformer_math.tex`'s one mention is a passing list of formal languages.
And the "not separated" sentence exists in **thirteen near-identical copies**: draft the replacement
wording once and propagate it. The `base=10000` confound is not mentioned in any of the six, so its
closure creates no stale text.

**`positional_review.tex`** (lightest: 3 rank statements, no Dyck exposure)
- 175 (Intro, "What is claimed, and what is not", C3) -- "which property of the bottleneck is responsible is not separated" -> it is per-head rank, D - C_bd p 0.0070.
- 1108 (Table `tab:capability`, row *absolute location in a bounded map*) -- "signed, **and** a shared $r\ge4$ to be found reliably" -> sharing is the UNMEASURED factor; should read "per-head rank >= 4". Same cell's "$0/8$ and $2/8$ against $8/8$" -> the five-arm split 0/8, 2/8, 2/8 | 8/8, 8/8. **Duplicated verbatim at `mapformer_math.tex` 2142; change together.**
- 1186-1195 (Open questions, first `\item` "The input-side rank") -- a listed open question the 2026-09-25 batch answers; rewrite as settled, keeping only "why rank 2 is rarely found" open, and add `LOOP_RANK_RESULTS.md` there as the first probe of it.
- 59-65 (abstract) and 1176-1179 (Conclusion): "Corrected 2026-09-25" stamps need a 2026-09-27 line.

**`axes_measured.tex`** (4 copies of one rank claim, no Dyck exposure)
- 46-47 (abstract), 241 (claim **F2**), 398-403 (`sec:rankmeas`, "At matched length it is search"), 1026-1027 (Conclusion) -- all four say the three candidates are not separated -> per-head rank fires, sharing and `W_out` scale UNMEASURED.
- 400-402 -- "the initial scale of the angle increments is matched" is now weaker than the truth: the worry is *answered* (C_bd std 0.208 vs A 0.335 / B 0.328, which fail identically), so low initial scale is not necessary for failure.
- `sec:rankmeas` has no table; the five-arm table has no home in this paper and F2 is the claim that now has a clean design without one.
- 1002 (Table `tab:combinations`, row "rank r=4 + weight sharing across depth", caption says "None has been built") -- `LOOP_RANK_RESULTS.md` partially builds that cell on the torus.

**`mapformer_math.tex`**
- 582-584 (the `\begin{quote}` box at 575, "Tested at matched length (2026-09-24)") and 1556-1558 (Prior art, `\item` on A6) -- the "not separated" sentence -> per-head rank.
- 593 (that box's source list) -- add `RANK_SEP_RESULTS.md`; re-date the box.
- 1928-1932 ("(ii') A6 is thinner than a blank column suggests, but still open") -- the heading "still open" and the two-arm contrast are outdated.
- 2142 (capability table) -- twin of `positional_review.tex` 1108.
- 2510 (Empirical anchors, **Bottleneck rank**) -- "solved 8/8 against 0/8" is incomplete; the citable form is the five-arm split.
- 2511 (Empirical anchors, **Recursion**) -- "matches or beats 3x params" now has a torus counter-instance: 4 real layers beat the matched-parameter loop (+0.017, perm p 0.0009, 5/8 vs 2/8). Scope it to parity / Match-Query or carry the counter-instance.

**`report/report.tex`** (heaviest: 6 rank, 3 recursion, 6 Dyck)
- Rank: 69-71 (abstract, "Findable solutions"), 1620-1626 (`sec:c3-rank`), 1642 (Table `tab:rank-mi` **caption**, whose whole "Reading:" clause needs rewriting), 1658-1660 (Scope, "Budget and initialisation": `W_out`'s per-entry scale is now measured, C_bd - B, and does NOT account for the split). Table `tab:rank-mi` **body** at 1645-1651 has three arms; add `Vanilla_r4mibd` (block-diagonal r=4, 2/8, 0.948) and `Vanilla_r4ph` (per-head r=4, 8/8, 0.999) plus a shared/per-head column. 2206-2209 (Limitations, "Our bottleneck is shared across heads") is accurate but is where "sharing is now measured (D - C, p 1.00 / 0.59)" belongs.
- Recursion: 1719-1720 -- "On the torus a one-layer path-integrated model is already near ceiling, so a loop has nothing measurable to add there" is **contradicted** at T=1024, r=2 (+0.079, perm p 0.0034, 2/8 vs 0/8); scope it to T=128 / r=4. 1748 -- "the loop against three real layers is +0.032: not distinguishable" is Match-Query-scoped and has a torus counter-instance. 1772-1776 (Prior art, "What this report adds") should include the torus loop batch and its threshold-sensitivity note.
- Dyck (`sec:dyck`, 1966-2052): 1977 + 1982 (Table `tab:dyck` caption "the split appears only out of distribution", and the two `D12` column headers) -- those columns are 3x the **training depth**; label them depth-extrapolation. 2013 + 2022 (Table `tab:dyck-std`) -- the bracket-memory gap is read at 3x depth and 4x length. 2037-2042 ("the paper's real claim survives and sharpens ... index models need a second layer") -- the one-layer half survives (+0.353, 8/8) but the framing should become depth-substitution, i.e. parameter efficiency, not something depth cannot buy. The section nowhere says the effect CLOSES at matched depth, nor carries the mixture-training survivor (+0.110), the death of the index-arm plateau (3x - 1x = +0.021, 8/8) or of "depth closes 40% then stops".
- 1938-1940 (`sec:length`, "Where a matched-length control has been run...") and 2194-2197 (Limitations, "Out-of-distribution length") -- add Dyck matched-*depth* as the third and clearest instance. **2197 is duplicated at `report_short.tex` 562.**
- 2188-2191 (Limitations, "Language modelling only at small scale") -- the one sequence result that got a matched-distribution control found its headline closing; say so.
- 2250-2252 (Conclusion) -- needs a 2026-09-27 correction line.

**`report/report_short.tex`**
- 428-430 -- the "not separated" sentence. 422-427 -- three of five arms; add the two that make the claim, and note the initial-angle-scale worry is answered rather than merely matched. 47 (abstract) -- "against 8/8 at a shared $r=4$" should read "at per-head rank 4".
- 504 (Claim 5, Recursion) -- same "+0.032, not distinguishable" scoping as `report.tex` 1748. 505-507 -- "on the torus the loop's gain is +0.052 raw, +0.006 matched-loss" is T=128 / r=4; add the T=1024 r=2 result.
- 562 (Limitations) -- "(code, rank)" -> "(code, rank, Dyck depth)". 585-587 -- where "sharing is now measured" belongs. 604-606 (Conclusion) -- needs a 2026-09-27 line.
- **Checked and NOT stale:** 518-520 (Claim 6, "on Dyck-2 a one-layer path-integrated model learns the bracket stack") is the L32 D4 training cell and survives at matched depth.

**`RESULTS_INDEX.md`**
- 1 (title) -- "regenerated 2026-09-24; rank, documents and statistics updated 2026-09-25": re-date.
- 17-19 (Documents paragraph) -- "all brought into line with this index on 2026-09-25" becomes false the moment the index is fixed and the documents are not; say which lag.
- 27 (rank row) -- claim cell "is unseparated" -> per-head rank (Fisher and perm p 0.0070, Holm 0.028); status cell "r=4's smaller `W_out` init ... is unseparated" -> measured (C_bd - B) and not the cause; key-number cell -> all five arms; file cell -> add `RANK_SEP_RESULTS.md`, `RANK_SEP_PREREG.md`.
- 48 (**NEEDS CONTROL** table, "Dyck ladder at L32 D12") -- both missing controls have now been run. The row should leave NEEDS CONTROL entirely and be restated as a closed headline pointing at `DYCK_MDEPTH_RESULTS.md`.
- 28 -- the index has no row for what survives on Dyck: add the matched-depth ladder and the mixture result.
- 33 ("Paper replications", Hewitt +0.064 at L128 D12, status SOLID) -- that cell is 3x depth and 4x length; label it depth-extrapolation rather than bare SOLID.
- 30 (loop row) -- or a new Open bullet: H1's registered UNMEASURED verdict.
- 57-64 (Open bullets, "Rank, the old numbers", line 61 "`RANK_MI_RESULTS.md` is the readable test") -- point at `RANK_SEP_RESULTS.md`.
- 127 -- "Top-level `*.md` only (444 files)"; there are **455** (verified 2026-09-27).
- Catalogue counts and missing entries: 138 "Rank, generator and accumulator (44)" is missing the four `RANK_SEP*` files (-> 48); 150 "Match-Query, loop and algorithmic tasks (38)" is missing the four `LOOP_RANK*` files (-> 42); 166 "Dyck-2 (16)" is missing the three `DYCK_MDEPTH*` files (-> 19). Regenerating with `docs/tools/catalog_results_index.py` fixes all of these at once (it currently reports one unclassified file, `FAST_ATTN_RANK.md`, which needs a group pattern).

Line numbers are from a read of the files on 2026-09-27; the `.tex` files are unchanged since
commit `414b60a` and `RESULTS_INDEX.md` since `e6eb22d`, but re-grep before editing rather than
trusting a number.

Also stale, lower stakes:

- `RESULTS_INDEX.md` was last *regenerated* 2026-09-11 and is missing the Dyck, Bach, code,
  rank-matched, rank-separation and loop-rank files entirely. Regenerate the catalogue with
  `docs/tools/catalog_results_index.py`, then correct the two rows above by hand.
- No document mentions `LOOP_RANK_RESULTS.md` at all. It belongs beside the rank claim in each:
  the search account has partial support, and the matched-parameter loop is not the clean
  "search at constant capacity" demonstration it was designed to be.

## The working method, in five lines

1. **Pre-register** before the batch: the primary readout, the branch boundaries set against the
   measured noise floor, and the task's floor (the better of an n-gram and a constant, per cell).
   Report the registered verdict afterwards even when the result makes it look silly.
2. **One batch.** Every arm retrained together, never against a stored checkpoint.
3. **Eight seeds.** Below n=6 no distribution-free test reaches p < .05; n <= 3 is not a point
   estimate. Below the MDE the word is "unmeasured", not "null".
4. **Matched distribution.** Train at the length, the depth, and the setting of whatever axis the
   claim is about. An effect seen only outside that is robustness until a matched control says
   otherwise.
5. **Convergence before comparison,** and verify what a probe measures by reading its code. Most of
   the withdrawal list is runs that had not converged or probes that measured something else.
