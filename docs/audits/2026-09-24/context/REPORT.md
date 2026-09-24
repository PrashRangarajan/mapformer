# Context / token-usage audit -- 2026-09-24

Read-only audit of what `/home/prashr/mapformer` puts into every Claude session. Tokens are
chars/4 (no tokenizer is installed -- VERIFIED `tiktoken` absent); number-dense markdown usually
tokenizes denser than that, so true counts are probably 10-20% higher (JUDGED). Every claim is
tagged **VERIFIED** (measured or grepped here) or **JUDGED** (from reading). Scripts:
`classify.py` (per-line classification), `unique_check.sh` (only-in-CLAUDE greps), `chunks.tsv`.

Deliverables in this directory: `REPORT.md` (this), `CLAUDE_proposed.md` (drop-in
replacement), `HISTORY_PLAN.md` (where the log goes + facts that exist only in CLAUDE.md),
`MEMORY_proposed.md` (tightened memory index).

## 1. What is loaded every session, and what it costs (VERIFIED sizes)

| file | how it enters context | chars | ~tokens |
|---|---|---|---|
| `CLAUDE.md` | auto: every session, every subagent, every reload after compaction | 183,065 (3,272 lines) | **45,766** |
| `.claude-memory/MEMORY.md` | auto (memory index; the dir is a symlink target, VERIFIED) | 5,247 | 1,312 |
| **auto-loaded total** | | 188,312 | **47,078** |
| `.claude-memory/project_state.md` | "read first" in both CLAUDE.md and MEMORY.md | 23,667 | 5,917 |
| `RESULTS_INDEX.md` | "START HERE" pointer in CLAUDE.md | 24,440 | 6,110 |
| **total if the read-first instructions are followed** | | 236,419 | **~59,100** |

No `~/.claude/CLAUDE.md` or `/home/prashr/CLAUDE.md` exists (VERIFIED), so nothing else stacks
on top. The other 34 memory files (~150K chars, ~37K tokens) load only on demand.

**Subagents pay it too (VERIFIED).** This auditor's own injected context contains CLAUDE.md --
an older snapshot that lacks on-disk lines 3-125 (the "2026-09-23 (later)" block), i.e. the
injected copy is fixed at session start and mid-session edits do not reach running
sessions/subagents (mechanism JUDGED). With five parallel auditors, this audit alone spent
~5 x 43K = ~215K tokens on CLAUDE.md before any work.

**Growth (VERIFIED from git):** 54.7K chars (2026-05-15) -> 67.7K (08-10) -> 82.2K (08-25) ->
129.8K (09-05) -> 159.2K (09-12) -> 183.6K (09-24). +53.8K chars in the last 19 days, about
**+700 tokens per day**; at that rate it passes 60K tokens by mid-October (JUDGED extrapolation).

## 2. CLAUDE.md by class

Classes: **CURRENT** = needed every session (live frame, citable result, invariant);
**RULE** = standing method/ops rule; **POINTER** = tells you which file to read;
**HISTORICAL** = superseded, retracted, or session narrative whose results live in a results
file; **DUPLICATE** = currently valid but said elsewhere (memory file, `EM_WM_STATE.md`,
`LANGUAGE_LANDSCAPE.md`, `project_state.md`). Per-line labels by section with overrides for
the three mixed sections (`classify.py`); sizes VERIFIED, labels JUDGED.

| class | chars | ~tokens | share |
|---|---|---|---|
| HISTORICAL | 118,673 | 29,668 | **64.8%** |
| DUPLICATE | 25,247 | 6,311 | **13.8%** |
| RULE | 19,839 | 4,959 | **10.8%** |
| CURRENT | 17,261 | 4,315 | **9.4%** |
| POINTER | 2,046 | 511 | 1.1% |

Even the 9.4% CURRENT is written as dated narrative, and most of it is repeated nearly
verbatim in `project_state.md`'s LATEST blocks (JUDGED from reading both). The RULE share is
~5K tokens but scattered across ~15 places (section 3, finding 3).

Per top-level section (class mix in % of the section's chars; HIST/DUPL/RULE/CURR/POIN):

| # | line | ~tok | class mix | section |
|---|---|---|---|---|
| 0 | 3 | 2365 | HIST38 CURR23 RULE19 DUPL18 | 2026-09-23 (later) -- shared report, bf16, MAESTRO plan, the PoPE/MapFor |
| 1 | 126 | 346 | CURR100 | 2026-09-23 -- Dyck depth ladder: the last positive result SURVIVES; C1 c |
| 2 | 145 | 505 | CURR100 | 2026-09-22 -- PoPE ablation: bound account DEAD; the "non-replication" w |
| 3 | 177 | 1304 | HIST100 | LANDED then RETRACTED, 2026-09-21 -- code modelling |
| 4 | 255 | 1200 | CURR46 POIN22 RULE19 HIST12 | LATEST, 2026-09-15..20 -- the Dyck-2 and PoPE-paper line (read `.claude- |
| 5 | 320 | 320 | DUPL91 CURR8 | LATEST, 2026-09-12..15 (read `.claude-memory/project_state.md` LATEST bl |
| 6 | 342 | 1221 | DUPL87 POIN12 | START HERE (updated 2026-09-11) |
| 7 | 415 | 82 | CURR100 | Project in one sentence |
| 8 | 423 | 888 | DUPL68 CURR31 | IMPORTANT (2026-08-27): the MapFormer paper is now at v4 and DID languag |
| 9 | 485 | 653 | RULE100 | STANDING RULES 8-12, all bought by the 2026-08-26..28 retractions |
| 10 | 529 | 482 | HIST100 | Current state (what's implemented and working) |
| 11 | 559 | 263 | CURR100 | Things that are true (verified) and must be preserved |
| 12 | 580 | 270 | HIST100 | Architectural choices that matter |
| 13 | 600 | 304 | HIST100 | Landmark experiment (added in latest session) |
| 14 | 628 | 185 | HIST100 | Clone-structure analysis (added in latest session) |
| 15 | 644 | 272 | HIST100 | Main empirical finding |
| 16 | 668 | 202 | HIST100 | Things that didn't work / why |
| 17 | 683 | 287 | HIST100 | Open questions / natural next experiments |
| 18 | 704 | 266 | HIST100 | Quick reproducibility commands |
| 19 | 731 | 176 | HIST100 | Filesystem map |
| 20 | 745 | 131 | RULE100 | Authoring style / preferences |
| 21 | 755 | 95 | HIST100 | Level 2 InEKF results (autonomous addition) |
| 22 | 765 | 123 | HIST100 | Level 1.5 InEKF (compromise between Level 1 and Level 2) |
| 23 | 776 | 242 | HIST100 | Matched-compute verification: settled (L1.5 wins architecturally) |
| 24 | 800 | 285 | HIST100 | Parallel work while orchestrator runs (2026-04-22) |
| 25 | 822 | 1315 | HIST100 | Session 2026-04-22 evening — honest framing + WM vs EM + Level15EM + pap |
| 26 | 935 | 1110 | HIST100 | Session 2026-04-24 — final results, Level15EM init pathology + fix |
| 27 | 1027 | 361 | HIST100 | Session 2026-04-26 — Level15PC + Grid + GridL15PC findings |
| 28 | 1061 | 1318 | HIST100 | Session 2026-04-27 — Kalman = stabilisation, R-saturation diagnosis, NoB |
| 29 | 1170 | 335 | HIST100 | Session 2026-04-28 — v3 / v4 PC isolation + length diagnostic |
| 30 | 1203 | 1258 | HIST100 | Session 2026-04-29 — multi-seed v4 + PC/Kalman duality + Sorscher Option |
| 31 | 1308 | 921 | HIST100 | Session 2026-04-30 — Fix 8 audit, RNG-control, MiniGrid wrapper, SE(n) g |
| 32 | 1391 | 1074 | HIST100 | Session 2026-05-01 — DoG bug, continuous nav, stochastic-transition fram |
| 33 | 1474 | 2975 | HIST100 | Session 2026-05-10 — TEM fixes, β / dropout discovery, EM/WM mechanism,  |
| 34 | 1731 | 548 | HIST100 | RETRACTION: all lm200 results (2026-07-16) |
| 35 | 1785 | 2767 | HIST97 RULE2 | Session 2026-07-25 — koopman clone, compositional multi-seed, model rena |
| 36 | 1964 | 1294 | HIST65 CURR18 RULE7 POIN7 | Session 2026-08-09 — Match-Query verified; three task lines voided |
| 37 | 2059 | 2694 | HIST51 RULE38 DUPL10 | Session 2026-09-03/04 -- rank is the story; Selective RoPE is not |
| 38 | 2252 | 2336 | HIST46 DUPL38 RULE15 | Session 2026-08-19/20 — the headline changed: the ENVIRONMENT decides |
| 39 | 2406 | 2103 | HIST75 RULE24 | Session 2026-08-29/30 -- the aliasing story dies; recursion; two retract |
| 40 | 2560 | 628 | CURR100 | Loop x path integration on Match-Query (2026-08-31) -- they DO compose |
| 41 | 2605 | 798 | HIST100 | The InEKF does not pay even where its premise holds (2026-09-02, two tak |
| 42 | 2659 | 249 | HIST100 | Fixing power: lr 1e-3 on the torus (2026-09-02, RECIPE_POWER.md) |
| 43 | 2675 | 163 | HIST100 | MoR routing: nothing to route on here (2026-09-02, LOOP_DEPTH_STRATA.md) |
| 44 | 2685 | 904 | HIST66 RULE33 | Filter x loop: NOT complementary (2026-09-01, n=12) |
| 45 | 2746 | 2419 | HIST68 DUPL20 RULE10 | Session 2026-08-31/09-01 -- refinement is dead; the loop is the surprise |
| 46 | 2917 | 984 | HIST68 DUPL31 | Session 2026-09-05/06 -- the theory is mostly published; rank and naviga |
| 47 | 2979 | 1179 | DUPL40 CURR26 RULE19 HIST12 | Session 2026-09-06 -- the sign axis, the clock/map reframe, and a review |
| 48 | 3060 | 1874 | HIST53 CURR26 RULE19 | Session 2026-09-07/08 -- the crossover lands; two documents become three |
| 49 | 3180 | 1666 | DUPL85 RULE14 | Session 2026-09-09/10 -- Neuron [11] read; EM/WM kernel theory built, te |

Notes on the table (JUDGED): section 7 "Project in one sentence" is labelled CURRENT but
describes the April project ("reproduction plus three extensions: parallel InEKF, sequential
InEKF, predictive coding"), which is no longer what the project is. Sections 10-28 ("Current
state", "Landmark experiment", "Main empirical finding", "Filesystem map", "Quick
reproducibility commands", ...) are April-era state presented under present-tense headings;
the lm200 parts are under the 2026-07-16 retraction. Section 33 (2026-05-10, 2,975 tokens) is
the single largest block and is mostly void results.

## 3. Findings, ranked by tokens at stake

1. **90% of the auto-loaded file is not needed each session** (VERIFIED sizes, JUDGED labels):
   78.6% is HISTORICAL or DUPLICATE. The file is a diary with corrections appended rather than
   applied -- it contains **83 correction markers** (RETRACTED 11, WITHDRAWN 13, CORRECTED 22,
   FALSIFIED 11, VOID 13, REFUTED 10, SUPERSEDED 3; VERIFIED grep), so a reader must load both
   the claim and its retraction to learn that nothing is there.
2. **The live state is stale in three places at once** (VERIFIED against `git log`): CLAUDE.md's
   top block and `project_state.md` both say the rank 900-epoch pilot is "running"; MEMORY.md and
   `project_rank_and_selective_rope.md` say the matched-length test is "pending / not yet run".
   In fact the 8-seed 900-epoch batch finished (eca6d2c: r=4 8/8 solved, r=2 0/8, verdict
   UNREADABLE), and Amendment 3 and a warm-start stability test were committed by 01:16 today
   (238a971, aa046d5). About 20 further "in flight / queued / PID nnn / pending" lines in
   CLAUDE.md describe April-September jobs long finished (VERIFIED grep). State that is
   copied into several files goes stale in all of them within hours.
3. **The standing rules exist in three numbering schemes and 15+ places** (VERIFIED): CLAUDE.md
   numbers 8-12, 13-16, 17-20, 21-26, 27-34 plus ~12 unnumbered "Rules bought" lists;
   `RESULTS_INDEX.md` has its own 1-28 (20 and 22 missing) and C27-C33; `project_state.md`
   says "33 numbered rules". They collide: CLAUDE.md's rule 8 is "measure the noise floor",
   RESULTS_INDEX's rule 8 is "apply a retraction to the generators"; both files acknowledge the
   27-28 clash. Most rules also have a `feedback_*.md` memory file (e.g. the LinearLR rule is in
   CLAUDE.md, RESULTS_INDEX rule 10, `feedback_convergence_first.md`,
   `feedback_recipe_before_architecture.md` and `project_miniworld_flip_negative.md`). Merged and
   deduplicated they fit in 28 rules of 1-3 lines (`CLAUDE_proposed.md`).
4. **The "start here" pointers point at stale files** (VERIFIED): `RESULTS_INDEX.md` was last
   regenerated 2026-09-11 and has zero mentions of DYCK, JSB, CODE_, INDIRECT, CROSS_RESULTS,
   RANK_MATCHED, BF16, T2_RESULTS or AUG_RESULTS; it still says "Use r=4" without the OOD-only
   caveat and that the MapPoPE r=4 "batch is running" (`MAPPOPE_R4_RESULTS.md` has landed).
   The "START HERE (updated 2026-09-11)" section sits at line 342, below 340 lines of later
   narrative.
5. **`project_state.md` (5.9K tokens, "read first") mostly duplicates CLAUDE.md** (JUDGED from
   headings and the 09-23 block, which is a condensed copy of CLAUDE.md's): its 2026-09-23,
   09-21, 09-15..20 and 09-12..15 blocks mirror CLAUDE.md's same-dated blocks, and its
   "In flight (2026-09-11)" / "Open, and ranked" sections are superseded by its own LATEST
   block. The shared-report link is in four files (CLAUDE.md, MEMORY.md, project_state.md,
   reference_shared_report.md; VERIFIED).
6. Smaller: CLAUDE.md cites 4 files that exist nowhere (`RESULTS_LEVEL2.md`,
   `RESULTS_LEVEL15.md`, `RESULTS_LEVEL15_CLEAN.md`, `STOCHASTIC_TRANSITION_RESULTS.md`) and 3
   April `paper/0x_*.md` drafts that are gone (VERIFIED). The "UNCONTROLLED CONFOUND"
   (index `base=10000`) that the 2026-09-21 block says costs "~20 GPU-min to settle" was
   settled -- base 32/128 move the index arms by -2%..+3% (`DYCK_DEPTH_RESULTS.md` lines
   120-126, VERIFIED) -- but CLAUDE.md never says so.

## 4. The proposed replacement (`CLAUDE_proposed.md`)

**18,317 chars, ~4,580 tokens (-90% against 45,766; VERIFIED size).** Contents: what the project
is (one paragraph, rewritten for 2026-09); binding conventions with the single-author /
no-`Co-Authored-By` rule stated as binding and overriding session defaults; a where-things-live
table; 18 citable results in one table with numbers and files, led by the matched-vs-mismatched
frame; rank marked OPEN; live negatives on one paragraph; 26 withdrawn items one line each with
file; 28 merged rules (measurement 1-10, design 11-19, operations 20-28), each with the number
that bought it; paper-faithfulness invariants, the v4 reproduction target and the environment
gotchas. It deliberately carries **no in-flight state** -- that lives only in `project_state.md`
so it cannot go stale in two places.

Every number in it was taken from CLAUDE.md, RESULTS_INDEX.md or the cited results file and
every cited file exists (VERIFIED `ls`). Cross-check before applying: the rank line
("r=4 8/8, r=2 0/8, UNREADABLE") reflects `RANK_MATCHED_RESULTS.md` as of aa046d5 and will need
updating when the warm-start test lands.

## 5. What would be lost, and where the log goes

See `HISTORY_PLAN.md`. Summary: `git mv CLAUDE.md docs/LOG.md` keeps everything verbatim (not
auto-loaded). Of ~65 specific facts and rules grepped across every top-level `*.md`, the memory
files and both archives, the ones with **no other carrier** are: the setsid / 2-minute-SIGTERM
rule, `wait` returning on dead children, atomic `mv` replacement of a running driver, the
`local a=$1` expansion trap, the TF32 non-equivalence figure (6.0e-01) and the MapEM SDPA
exclusion, the v4 Table 2 targets -- all kept in `CLAUDE_proposed.md` -- plus four that only the
log would keep and that deserve a real home: the **run-directory map for the Dyck / Bach /
Indirect line** (CLAUDE.md says no results file names them), the note that
**`DYCK_T3_RESULTS.md` is an empty artefact**, the rest of the **v4 diff** (Mamba/MAmPa/RoPE-4L
revisions, TAPE/PathAtt), and the **`--fast-attn` 2.56x / 37%-memory** figure.

## 6. Memory directory audit (briefer)

VERIFIED: 36 files (35 + MEMORY.md), ~156K chars. Every file is indexed and no index link
dangles. Only MEMORY.md (1.3K tokens) is auto-loaded, so the memory files cost tokens only when
read -- but `project_state.md` is read at every start.

| finding | evidence | proposal |
|---|---|---|
| Rank entries stale | MEMORY.md "matched-length test pending"; `project_rank_and_selective_rope.md` frontmatter "audited GO, not yet run" vs git eca6d2c/238a971/aa046d5 (VERIFIED) | update both |
| `project_state.md` duplicates CLAUDE.md and itself | 09-12..15, 09-15..20, 09-21 blocks mirror CLAUDE.md; "In flight (2026-09-11)" and "Open, and ranked" superseded; "33 numbered rules" (JUDGED/VERIFIED) | cut to live state only, <= 6K chars (~1.5K tokens); history to `docs/LOG.md` |
| `feedback_action_noise_framing.md` contradicted | "our +10pp Level 1.5 win ... is a lower bound; structured real noise should give larger wins" (VERIFIED line 18) vs `MQ_NOISE_2X2*.md` (benefit flat/negative as drift grows) and `NOISE_REFINE.md`; the +10pp is also an OOD-length number | keep the vocabulary advice, delete the claim |
| Layered corrections inside memory files | `project_clock_vs_map.md` 16.9K chars with 10 WITHDRAWN markers; `project_hierarchy_negative.md` 14.1K with 3 RETRACT markers (VERIFIED counts) | rewrite each to the surviving claim (~3K chars) |
| Overlapping feedback files | recipe_before_architecture ~ convergence_first; premise_before_test + prelaunch_audit ~ validate_task_first; verify_before_relaying ~ probe_verification; verify_before_destructive + cwd_and_module_paths ~ scheduler traps (JUDGED) | merge: 35 -> 25 files |
| Historical files still indexed as method | `feedback_lm200_stuck_baselines.md` (covered by CLAUDE rule 12 + withdrawn list), `feedback_pc_kalman_duality.md` (April; belongs with the correction line) (JUDGED) | fold into `feedback_convergence_first.md` / `project_loop_and_correction.md` |
| Documents reference incomplete | `reference_review_documents.md` has no `report.pdf`, `report_short.pdf` or shared-report entry (VERIFIED grep) | merge `reference_shared_report.md` and `reference_paper_corpus.md` into it |
| Link repeated 4x | shared-report URL in CLAUDE.md, MEMORY.md, project_state.md, reference_shared_report.md (VERIFIED) | keep in CLAUDE.md + the documents reference only |

`MEMORY_proposed.md`: 3,274 chars (~820 tokens, -37%), 25 entries, the rule-detail files mapped
to the CLAUDE.md rule numbers so the two stop drifting.

## 7. Estimated savings per session (JUDGED from the VERIFIED sizes)

| | now | proposed | saved |
|---|---|---|---|
| auto-loaded (CLAUDE.md + MEMORY.md) | ~47.1K | ~5.4K | **~41.7K (-89%)** |
| plus `project_state.md` read at start | ~53.0K | ~6.9K | ~46K |
| plus `RESULTS_INDEX.md` (if followed) | ~59.1K | ~6.9K (index becomes optional) | ~52K |

Multipliers: the auto-loaded saving recurs on every compaction reload and every subagent
spawn -- a five-agent fan-out like this one saves ~200K tokens. It also stops the ~700
tokens/day growth, provided the "no narratives in CLAUDE.md" convention is kept.

## 8. Applying it (for the main session; nothing here was applied)

1. `mkdir -p docs && git mv CLAUDE.md docs/LOG.md` (+ one header line); copy
   `CLAUDE_proposed.md` to `CLAUDE.md`; refresh its rank line from `RANK_MATCHED_RESULTS.md`.
2. Rewrite `project_state.md` to live state (move its dated blocks to `docs/LOG.md`); add the
   run-directory map there.
3. Apply `MEMORY_proposed.md` with the merges listed in section 6; fix the rank frontmatter and
   the action-noise claim.
4. Regenerate `RESULTS_INDEX.md`; delete its private rule list in favour of CLAUDE.md's numbers.
5. Commit single-author, **without** a `Co-Authored-By` line -- the project rule is binding and
   overrides the attribution reminder that sessions arrive with.
