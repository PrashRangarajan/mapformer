# Where things stand -- 2026-10-03
> **Update 2026-10-09:** results since 10-03 (dropout re-score, MapPoPE separated, SCORE_RANK, GAIN_GRAIN, neuro review, word-class
> breakdown) and the running / queued / ready batches: `docs/SESSION_2026-10-04_to_10-09.md`, `.claude-memory/project_state.md`, CLAUDE.md rows.

Orientation for a fresh session. Read `.claude-memory/project_state.md` first (what is running, what the user
must decide), then this. `CLAUDE.md` holds the conventions, the citable table and the withdrawal list and is the
authority on all three; this file is the shape of the project around them. This week's narrative (what was asked,
run, audited, corrected): `docs/SESSION_2026-09-27_to_10-03.md`. Every number below was re-read from the file
beside it on 2026-10-03; a correction or amendment block at the top of a results file supersedes its body.

Status words used throughout: **REG** = pre-registered, one batch, 8 seeds, verdict as registered (with its
amendment); **REG, no branch** = registered but the result fell in no registered branch, so the numbers are
reported as they fell; **not pre-reg.** = an older batch with no pre-registration file; **POST HOC** = eval-only
analysis of stored checkpoints; **PILOT** = n=1-2, not a result.

## The thesis

This began as a reproduction of Rambaud et al.'s MapFormer -- a transformer whose rotary angle is a
path-integrated, content-dependent phase rather than the token index -- and turned into a study of when a learned
"where" separates from the "what". What it has established is mostly a negative with a sharp edge: **nearly every
positional-encoding effect measured here was robustness to distribution shift, not capability, and closed once
training matched the test on the axis the claim is about.** Code extrapolating past 512 gave -3.694 bpc; trained
at 2048 and scored on the full val file the encoding effect is -0.0033, unmeasured. Dyck's +0.168 at 4 layers and
nesting depth 12 was trained at depth 4; trained at depth 12 every arm reaches ceiling (+0.002). The general form:
match the training distribution on every axis the task varies, or you are measuring extrapolation.

**Sign is the one exception so far**: the first never-controlled OOD claim to get its matched-length control
survived it (T=1024, monotone 0/8 and 1/8 solved vs signed 8/8). What survives matched-distribution tests:
navigation on the torus at training length; depth substitution (one path layer ~ three attention layers, on Dyck
and on a biased 1D walk); the per-head rank of the content-to-angle map -- the one line where controls made the
effect sharper (rank = task dimension is hard in 2D and 3D; one spare direction per head fixes it on large tori);
sign; navigation told in English. The newest line asks what/where directly: trained path models do not need the
content x position interaction in their score (post hoc), and removing the small leak of object identity into
the step gains exactly the accuracy that leak costs, +0.0107 in distribution (registered; partly training speed).

What is ours is empirical. The taxonomy and the content-dependent rotation are published (GRAPE 2512.07805,
Mamba-3 2603.15569, Selective RoPE 2511.17388); sign is a replication in a new regime; the context-step
mechanisms, "separation is learned" and transfer to unseen codes are prior art (`docs/lit/`). Ours: the rank
result, the navigation regime, and measurements in it (the causal score form, the leak and its cost).

## What survives

`CLAUDE.md`'s citable table is the full list with scopes; these are the load-bearing rows.

| result | numbers | status | file |
|---|---|---|---|
| Path integration helps on the torus **at training length** | converged recipe: position **+0.243** (MDE 0.038, 8/8); index RoPE 0.805, path 0.971; +0.359 at 8x length (OOD) | REG | `PAPER2X2_RESULTS.md` |
| ...and is necessary for in-context maps | Match-Query 0.730 +/- 0.247 (n=5) vs index 0.154, chance 0.0625; context destruction 0.918 -> 0.074 | not pre-reg., n=5 | `MATCH_QUERY_SCALE.md`, `MATCH_QUERY_RESULTS.md` |
| Dyck: path integration is worth ~3 layers of attention, **at matched depth** | trained AND tested at L32 D12: +0.353 / +0.130 / +0.045 / +0.024 at 1/2/3/4 layers (8/8 each, floor 0.594); mixture training over D 4..12 keeps +0.110 at D12 | REG | `DYCK_MDEPTH_RESULTS.md` |
| ...the same exchange rate on a biased 1D walk (H3) | 1 path layer solves all 32 cells; index needs 3 layers at p_plus 0.5 / 0.75 / 0.9, 1 at 1.0. At 0.9 the 2-layer gap is 0.0115 vs a 0.01 threshold: **knife-edge**. Fresh seeds (s2-s7; s0-s1 were the pilot) agree | REG, no branch (primary non-monotone; exchange rate is a registered secondary) | `CANCEL_RESULTS.md` |
| **Rank: the PER-HEAD rank of the content-to-angle map decides it** (torus, T=1024, 900 ep) | per-head rank 2 solves 0/8, 2/8, 2/8; per-head rank 4 8/8, 8/8. D - C_bd Fisher and permutation p 0.0070. Sharing and `W_out` scale UNMEASURED | REG | `RANK_SEP_RESULTS.md` |
| ...it is SEARCH, not capacity | a rank-2 solution exists (0.9955 frozen) and is held under training (7/8) | REG | `RANK_PROJ_RESULTS.md` |
| ...rank 3 sits with rank 4 | per-head r=3 6/8, 0.987 (r=2 0.885, r=4 0.999); 3 - 2 +0.102, perm p 0.027, Holm 0.054 (at its boundary on every count); 4 - 3 UNMEASURED | REG | `RANK3_RESULTS.md` |
| ...2x the budget does not rescue rank 2 (H1 part 1) | 1800 epochs from scratch: r=2 0/8 (0.908) vs r=4 7/8 (0.994), Fisher p 0.0014. 5/8 rank-2 runs still descending: "never" not shown | REG | `LOOP_RANK_E1800_P1_RESULTS.md` |
| ...search aids help but do not reach rank 4 | r=2 + loop x4 2/8, 0.973; 4 real layers 5/8, 0.990; plain r=2 0/8, 0.894; r=4 8/8, 0.998 | REG, verdict UNMEASURED | `LOOP_RANK_RESULTS.md` |
| ...**rank = D is hard in 2D and 3D; D+1 suffices on large tori** | 2D grid 32: rank 2 1/8 vs rank 3 8/8 (Fisher 0.0014). 3D grid 10: rank 3 1/8, rank 4 4/8 (UNMEASURED vs rank 3); 3D grid 18: rank 4 **8/8, 1.000** (vs grid 10 +0.153, perm p 0.0085). Failures sit on wrap-only revisits (0.53-0.80 vs 0.91-0.97 elsewhere). A 100-cell 2D map is memorised (own 0.986, unseen 0.273), so the registered 2D wrap contrast is not a wrap test | REG (`RANK_ND`: no branch; `RANK_WRAP`: WRAP DRIVES IT, clean only in 3D) | `RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md` |
| Sign of the increment, **at matched length** | trained and tested at T=1024: Abs - Signed -0.177 (perm p 0.0002), 0/8 vs 8/8 solved; opposition 0.06 vs 1.92-1.97. Monotone arms stalled (budget-scoped); a replication in a new regime | REG | `SIGN_MATCHED_RESULTS.md` |
| Navigation told in words: PATH WINS IN WORDS | path 1L 0.969 vs RoPE 1L 0.505 (floor 0.505, reversal-copy 0.597), RoPE 2L 0.772; fresh seeds +0.474, 6/6 vs 0/6, p 0.0022. Step table: registered B did not fire; opposites cancel on 8/8 after a common component, which on 4/8 seeds is a real per-step clock. Scripted grammar, context-free steps | REG (A); B no branch, read from declared secondaries | `TEXTWORLD_RESULTS.md` |
| **What-to-where leak: removing it buys exactly what it costs** | new-object task (unseen object codes, T=1024, 900 ep): MapWM 0.9890, ActOnly 0.9997, NormStep 0.9998; remedy - MapWM **+0.0107** each (perm p 0.0002, MDE ~0.0035) = MapWM's own leak. Remedies 16/16 SOLVED, MapWM 8/8 still descending (r(loss, acc) -0.944): partly training speed. The registered x4 robustness test could not fail (scale invariance by construction) | REG, verdict REMEDY read through Amendment 1 | `LEAK_RESULTS.md`, `docs/NORMSTEP_NOTES.md` |
| Trained path models do not need the content x position interaction | paper-torus checkpoints: object identity removed from the score, converged path arms keep 0.974-0.989 (cost 0.011-0.025, below the MDE); one shared kernel x a content gain 0.988-1.000; RoPE / PoPE fall below the 0.598 floor; token TYPE (action vs observation) is needed. Separation also on a never-redrawn 32x32 map | POST HOC | `docs/WHAT_WHERE_CHECKS.md` |
| Boundary: map extent, a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing (index arms behind +0.305 were still descending) | not pre-reg. | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Boundary: rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8), at the 16-epoch recipe | not pre-reg.; needs the converged recipe | `KNOB_SWEEP_n8.md`, `H12_BUDGET_CURVE.md` |
| A shared block looped x4 helps path integration | Match-Query +0.346 unpaired (t 3.75); matches 3 real layers at 1/3 the params | not pre-reg. | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8); installed rewind frozen 1.000 | REG | `EM_WM_STATE.md`, `SEARCH_RESULTS.md` |

Everything else in `CLAUDE.md`'s table is real but narrower. Everything in its **Withdrawn** list is not to be
cited; most of it died to a control added later, not to a mistake in the run.

## Pilots and analyses that are not results

- **Context-dependent step** (direction words used without moving; `CTXSTEP_PILOT1-3.md`, `CTXSTEP_HS_RECIPE.md`,
  `CTXSTEP_HSR_PILOT.md`; PILOT). Window-limited steps (context gate, Selective-RoPE generator) suppress decoys
  with the cue 1-3 tokens away and fail at 6-13. The hidden-state step reaches far cues when it learns a step
  (7/14 runs do); HSR (word step + alpha * LN(h1), alpha init 0) learned one 4/4 and reached 0.966-0.998. The
  registered batch (`CTXSTEP_PREREG.md`) was STOPPED before any result (~31 GPU-h; the window hypothesis is
  near-guaranteed by construction). Every mechanism here is published (`docs/lit/LIT_CONTEXT_STEPS.md`).
- **New-object transfer** (`runs/newobj_pilot`, PILOT, n=1): path arms 0.976-0.982 on unseen object codes vs index
  0.39-0.44. The test could not fail (unseen iid codes through a linear encoder transfer: Chen 2019).
- **What/where analysis** (`docs/WHAT_WHERE_ANALYSIS.md`, ANALYSIS + descriptive probe, section 6 corrected
  2026-10-03): every scheme's score from the code; trained path models put most score variance in one shared
  kernel; PoPE separates at init but not more at the end of training.
- **NormStep notes** (`docs/NORMSTEP_NOTES.md`, ANALYSIS, weights checked on 2 seeds): robustness to code norm is a
  theorem; zeroing NormStep's observation steps removes a per-move gauge, not leak; its object-identity leak is ~5x
  smaller than MapWM's. Not provable that it must be. Predicted risk on language: the LN bias step becomes a
  word-count clock.
- **Literature reviews** (`docs/lit/LIT_CONTEXT_STEPS.md`, `LIT_WHAT_WHERE.md`, `LIT_NEW_OBJECTS.md`): what is
  prior art, and costed proposals at the end of each.

## What is open, ranked

By what would change the story per GPU-hour; costs from comparable batches. Nothing is running.

1. **NormStep on the text world, with a bias-free NormStep arm** (`docs/NORMSTEP_NOTES.md` section 5). Tests the one
   predicted failure of the leak remedy: with a variable number of tokens per move, the LN bias step is a
   word-count clock. Decides whether NormStep is a general recommendation. **~3 GPU-h.**
2. **Separation vs data at matched map size** (`docs/lit/LIT_WHAT_WHERE.md` P2; fresh vs fixed vs 4-map pool,
   MapWM r=4 vs MapPoPE r=4). Whittington's "separation needs factorised data" is not refuted, only confounded with
   map size. **48 runs, ~3.5 h.**
3. **The window limit as a cue-distance curve** (`LIT_CONTEXT_STEPS.md` P3; CG at k=2/4/8, SR, HSR, pad drawn
   per decoy). Replaces a by-construction claim with a measured transition. **~10-12 h.** Then **a Mamba-3-style
   gate at depth vs HSR** (P1, the comparison a reviewer will ask for), **~13 h**; decomposing the HSR fix (P2)
   ~9 h.
4. **The leak, a test that can fail**: anisotropic training codes with a SWAP shift (`LIT_NEW_OBJECTS.md` E2; gate
   on CPU first), **~10 h**; or train MapWM longer / penalise ||W A|| to see whether it reaches NormStep's leak.
5. **Rank follow-ups.** H1 part 2 (loop x4 and 4 real layers at rank 2, 1800 ep; the registered H1 primary),
   **~6 GPU-h**. Cross-head sharing (D - C) and `W_out` scale (C_bd - B) are still unmeasured and need a design
   that moves one without the other.
6. **Never-controlled OOD claims**: InEKF / Level 1.5, forget gate, PoPE-wrapping, rotate/allocentric. Low
   priority unless the mechanism question is revived; PoPE-wrapping is the cheapest.
7. **A real dataset.** Jericho is not a good test (6/57 games clean, near-trees;
   `/home/prashr/jericho_data/feasibility/table_final.txt`). Talk the Walk is the only candidate left. Unscoped.
8. **Trainer consolidation** (no GPU): `train_cancel`, `train_textworld`, `train_ctxstep{,2,3}`, `train_newobj` were
   cloned by sed; merge into one trainer, verified loss-exact on one seed (rule 19), before the next batch.
9. **Documents** (no GPU): see below.

## Known stale

As of **2026-10-03**:
- The five `.tex` documents (`positional_review`, `axes_measured`, `mapformer_math`, `report/report`,
  `report/report_short`) were brought into line 2026-09-27 and corrected 2026-09-30 where later results
  contradicted them (sign; code full-val). They are **incomplete, not wrong**: none carries rank 3, H1 part 1, H3,
  the text world, the context step, 3D rank / wrap, the what/where checks or the leak.
- The shared report (`report/language_summary.html`, https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc) was
  republished as v10 on 2026-09-30. It lacks 3D rank / wrap, what/where and the leak. Republish WITH `url=`.
- Not covered by any pass: `README.md`, the `paper/` and `paper_rank/` drafts, older summaries (`REPORT*.md`,
  `RESULTS_SUMMARY_*`).
- `RESULTS_INDEX.md` catalogue regenerated 2026-10-03 (zero unclassified, docs-level notes included).

## The working method, in six lines

1. **Pre-register** the primary readout, branches set against the measured noise floor, and the task's floor
   (the better of an n-gram and a constant, per cell). Report the registered verdict even when it looks silly.
2. **Audit before reading**: an independent, results-blind code-verification agent per batch; its findings go
   into an amendment before any result is read (it found unfailable tests, missing branches, probe errors).
3. **One batch, eight seeds**, pilots on seeds outside the batch. Below n=6 no distribution-free test reaches
   p < .05; below the MDE the word is "unmeasured".
4. **Matched distribution**: train at the length, depth and setting of the axis the claim is about.
5. **Convergence before comparison**; r(final loss, accuracy) beside every gain (the leak gain is partly speed).
6. **Verify what a probe measures** by reading its code; check gauges (NormStep's zeroing readout measured one).
