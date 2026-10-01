# Where things stand -- 2026-09-30

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
bpc; trained at 2048 and scored on the full val file it is -0.0033, unmeasured. Dyck at 4 layers
gave +0.168 at nesting depth 12; trained at depth 12 every arm is at ceiling and the effect is
+0.002. Length was not enough -- depth had to be matched too, and the general form is: match the
training distribution on *every* axis the task varies, or you are measuring extrapolation.
**The one exception so far is sign**: the first of the never-controlled OOD claims to get its
matched-length control survived it (trained and tested at T=1024, monotone arms 0/8 and 1/8 solved vs
signed 8/8, `SIGN_MATCHED_RESULTS.md`). What survives the matched-distribution test: navigation on
the torus at training length; depth substitution (Dyck, and the same 1 ~ 3 layer exchange rate on a
biased 1D walk, H3, knife-edge at its most biased cancelling cell); the per-head rank result, now
with rank 3 sitting with rank 4 and rank 2 not rescued by twice the budget -- the one line where
controls made the effect *sharper*; sign; and path integration on the torus walk told in English
(PATH WINS IN WORDS). The published taxonomy and the content-dependent rotation itself are not ours
(GRAPE 2512.07805, Mamba-3 2603.15569, Selective RoPE 2511.17388); what is ours is empirical -- the
rank of the content-to-angle map, and the navigation regime (sign is a replication in a new regime).

## What survives

Numbers verified against their results files on 2026-09-27; rows added 2026-09-30 checked against theirs. `CLAUDE.md`'s citable table is the full
list with scopes; these are the load-bearing ones.

| result | numbers | file |
|---|---|---|
| Path integration helps on the torus **at training length** | converged recipe: position **+0.243** (MDE 0.038, 8/8); index RoPE 0.805, path 0.971; grows to +0.359 at 8x length | `PAPER2X2_RESULTS.md` |
| ...and is necessary for in-context maps | Match-Query 0.730 +/- 0.247 vs index 0.154, chance 0.0625; context destruction 0.918 -> 0.074 | `MATCH_QUERY_SCALE.md`; the destruction pair is in `MATCH_QUERY_RESULTS.md` |
| Dyck: path integration is worth ~3 layers of attention, **at matched depth** | trained AND tested at L32 D12: +0.353 / +0.130 / +0.045 / +0.024 at 1/2/3/4 layers (8/8 each, A2f, floor 0.594). Mixture training over D 4..12 keeps +0.110 at D12 | `DYCK_MDEPTH_RESULTS.md` |
| **Rank: it is the PER-HEAD rank of the content-to-angle map** | per-head rank 2 solves 0/8, 2/8, 2/8; per-head rank 4 solves 8/8, 8/8. Decisive contrast D - C_bd, Fisher and permutation p 0.0070. Sharing and `W_out` scale UNMEASURED | `RANK_SEP_RESULTS.md` |
| ...and it is SEARCH, not capacity | a rank-2 solution exists (0.9955 frozen, 8/8) and is held under training | `RANK_PROJ_RESULTS.md` |
| ...but search aids do not close it | r=2 + loop x4 2/8 solved / 0.973; 4 real layers 5/8 / 0.990; plain r=2 0/8 / 0.894; r=4 8/8 / 0.998. Registered verdict **UNMEASURED** | `LOOP_RANK_RESULTS.md` |
| ...nor does twice the budget (H1 part 1) | at 1800 epochs from scratch: r=2 0/8 (0.908) vs r=4 7/8 (0.994), Fisher p 0.0014. 5/8 rank-2 runs still descending, so "never" is not shown | `LOOP_RANK_E1800_P1_RESULTS.md` |
| ...and rank 3 sits with rank 4 | per-head r=3 6/8 solved, 0.987 (r=2 0.885, r=4 0.999); 3 - 2 +0.102, perm p 0.027, Holm 0.054 -- registered RANK 3 SUFFICES at its boundary on every count; 4 - 3 UNMEASURED | `RANK3_RESULTS.md` |
| Sign of the increment, **at matched length** | trained and tested at T=1024: Abs - Signed -0.177 (perm p 0.0002), solved 0/8 vs 8/8; opposition 0.06 vs 1.92-1.97. Monotone arms stalled (budget-scoped); a replication in a new regime | `SIGN_MATCHED_RESULTS.md` |
| Depth substitution on a biased 1D walk (H3) | 1 path layer solves all 32 cells; index needs 3 layers at p_plus 0.5 / 0.75 / 0.9 and 1 at 1.0. At 0.9 the 2-layer gap is 0.0115 vs a 0.01 threshold: **knife-edge**. Registered primary in no branch. Seeds 0-1 were the pilot; fresh seeds agree | `CANCEL_RESULTS.md` |
| Navigation told in words: PATH WINS IN WORDS | path 1L 0.969 vs RoPE 1L 0.505 (constant floor 0.505, reversal-copy 0.597), RoPE 2L 0.772; fresh seeds +0.474, 6/6 vs 0/6. Step-table branch did not fire; opposites cancel on 8/8 after a common component, which on 4/8 seeds is a real per-step clock. Scripted grammar, context-free steps | `TEXTWORLD_RESULTS.md` |
| Boundary: map extent, a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Boundary: rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8) | `KNOB_SWEEP_n8.md`, `H12_BUDGET_CURVE.md` |
| A shared block looped x4 helps path integration | Match-Query +0.346 unpaired (t 3.75); matches 3 real layers at 1/3 the params | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8); installed rewind frozen 1.000 | `EM_WM_STATE.md`, `SEARCH_RESULTS.md` |

Everything else in `CLAUDE.md`'s table is real but narrower. Everything in its **Withdrawn** list is
not to be cited, and the list is long for a reason -- most of it died to a control that was added
later, not to a mistake in the run.

## What is open, ranked

Ranked by what would change the story per GPU-hour. Costs are estimates from comparable batches.
Done since 2026-09-27 and removed from this list: sign at matched length, rank 3, H1 budget (part 1),
H3.

1. **A context-dependent step** (`CONTEXT_STEP_DESIGN.md`, pilots `CTXSTEP_PILOT1-3.md`,
   `CTXSTEP_HS_RECIPE.md`; pilots, no result). The text world's steps are context-free; real language
   uses "north" without moving. Pilot 3 at n=1 per cell: window-limited steps (context gate,
   Selective-RoPE generator) suppress decoys when the cue is 1-3 tokens away and fail past their
   window -- ready to pre-register (exclude or declare seeds 0, 1, 5; the SR arm is not one-knob).
   The hidden-state step (HS) reached far cues on 5/5 runs that learned a step, but half its far-cue
   runs never learned one; the proposed fix (step from `emb + LN(h1)`, starting as the context-free
   model) is untested. Pilot the fix before registering HS. **~2 GPU-h pilot, then ~10-20 GPU-h.**
2. **InEKF / Level 1.5, forget gate, PoPE-wrapping, rotate/allocentric** -- still never had a
   matched-length control. The filter line is a live negative; the forget gate has no mechanism.
   **Low priority** unless the mechanism question is revived; PoPE-wrapping is the cheapest.
3. **Rank follow-ups.**
   - **H1 part 2** (loop x4 and 4 real layers at rank 2, 1800 ep; the registered H1 primary), deferred
     by Amendment 1. ~6 GPU-h.
   - **A 3D torus** (`environment_nd.py`): rank 3 succeeded on a 2-DOF task; if the threshold tracks
     DOF, rank 3 should fail where rank 4 succeeds. ~6 GPU-h.
   - Still unmeasured from `RANK_SEP_RESULTS.md`: cross-head sharing (D - C) and `W_out` per-entry
     scale (C_bd - B). Both need a design that moves one without the other; neither is cheap.
4. **A real dataset.** Jericho is not a good test (6/57 games clean, near-trees;
   `/home/prashr/jericho_data/feasibility/table_final.txt`). Talk the Walk is the only real dataset
   still worth a look. Unscoped.
5. **Trainer consolidation** (no GPU). `train_textworld.py`, `train_ctxstep{,2,3}.py` and
   `train_cancel.py` were cloned by sed; the 2026-09-30 audit proposes one trainer. Do it before the
   next registered batch, verifying loss-exact on one seed (rule 19).
6. **The documents** (no GPU): the .tex papers lack the new results (see below).

## Known stale

As of **2026-09-30**:
- The five `.tex` documents (`positional_review`, `axes_measured`, `mapformer_math`, `report/report`,
  `report/report_short`) were corrected 2026-09-30 only where a later result contradicted them (sign
  listed as never controlled or extrapolation-only; code encoding -0.0030 -> full-val -0.0033) and the
  four touched were rebuilt (`mapformer_math` had no contradicted statement and was left as is). They are
  **incomplete, not wrong**: none carries rank 3, H1 part 1, H3, the text world or the context step,
  and sign at matched length appears only as a correction, not as a result with its table.
- `report/language_summary.html` (shared report source) was updated 2026-09-30 (H1 part 1, H3
  knife-edge, code n=3, a text-world section) and has NOT been republished; the live link shows the
  previous version until it is republished WITH `url=`.
- `RESULTS_INDEX.md` catalogue regenerated 2026-09-30 (478 top-level `*.md`, zero unclassified);
  its hand-written rows cover the 2026-09-30 results.
- Not covered by any pass: `README.md`, the `paper/` and `paper_rank/` drafts, and older summaries
  (`REPORT*.md`, `RESULTS_SUMMARY_*`).

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
