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

Nothing in the six document files, as of **2026-09-27**. The five `.tex` documents and
`RESULTS_INDEX.md` were rewritten that day against `DYCK_MDEPTH_RESULTS.md`, `RANK_SEP_RESULTS.md` and
`LOOP_RANK_RESULTS.md` (scripts in `docs/prepared/tex_edits_2026-09-27/`, applied with `rep.py`), and
all five PDFs rebuilt from source. The "not separated" rank caveat is now "per-head rank fires; sharing
and `W_out` scale unmeasured" everywhere; the Dyck matched-depth closure and depth-substitution ladder
are in `report.tex` / `report_short.tex` / the index; the torus loop-rank batch is in every document
that carries the rank claim. The catalogue was regenerated (457 top-level `*.md`). Not covered by that
pass: `report/language_summary.html` (the shared report), `README.md`, and the `paper/`, `paper_rank/`
drafts.

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
