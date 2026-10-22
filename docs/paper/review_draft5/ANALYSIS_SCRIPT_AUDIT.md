# Analysis-script audit for main draft 2 (CPU-9)

2026-10-21. Independent, results-blind code audit (rule 29 applied to post hoc analyses) of every script under
`docs/paper/expansion/scripts/{mappope,recurrence,rank,sign,depth,language}/` plus
`docs/audits/2026-10-20/causal_whatwhere_T1024.py`, `docs/audits/2026-10-16/{sign_floors,ring_floors}.py` and
`docs/paper/figs/{torus,ring}_strata_floors.py`. Six auditors (one per group), CPU only (`CUDA_VISIBLE_DEVICES=""`, 4
threads), every re-run from a scratch copy with output paths patched. Nothing in the repo was written or committed by the
audit; `git status` was checked before and after by each auditor. `docs/paper/MAIN_DRAFT2.md` / `MAIN_APPENDIX2.md` were
written and committed (45abed7) by another session during the audit; **all line references below are to 45abed7**.

Parts 1-6 below are the six auditors' sections, merged verbatim. Their scratch helper scripts (paths under
`.../scratchpad/script_audit/<group>/`) are session-ephemeral; each section names the check it ran, so they can be
regenerated. Every BLOCKER and most MAJORs were spot-checked by the coordinating auditor against the code or the committed
output (marked **[verified]**); the rest rest on the group auditor's own re-computation (marked **[auditor]**).

Verdict scale: REPRODUCED (re-run output identical or within float noise; definitions match the report) / CORRECT WITH
FIX (code right, a definition, label or claim built on it must change) / WRONG / COULD NOT RUN. Severity: BLOCKER (a
number in draft 2 is wrong) / MAJOR (a definition or claim must change) / MINOR.

## 0. Bottom line

- **No script computes a wrong number from its own definition.** Every committed output re-derives exactly (byte- or
  bit-identical, or within 2e-5 CPU noise), and the one exception is `sign/step_basin.py`, which is not deterministic.
  Every rebuilt forward that was checked reproduces the model's own logits exactly (max |dlogit| 0.0; 104 mappope /
  recurrence, 32 causal, 16 torus + 128 ring depth, 5 text-world checkpoints; step tables / angles to <= 4.5e-6 on 384
  rank and sign checkpoints). No duplicated checkpoint (md5 over every file loaded) and no pilot seed pooled anywhere.
  Strata, retrace and revisit labels match the registered code (`eval_rank_strata.kinds_vec`, `sign_floors.py`).
- **Four BLOCKERs**, all definitions attached to correct arithmetic: text-world NEEDS-MAP gloss (B1), the T = 1024
  coherence-by-stratum head choice (B2), Table 4 mixing two clock definitions (B3), and "None of the 16 runs" against the
  script's own 1/16 (B4).
- **About 20 MAJORs**, mostly claims that outrun the analysis (parking, global count, "code unchanged", "every
  success/failure", the signature) and two subset / label issues.
- No WRONG and no COULD NOT RUN verdict.

## 1. Verdict per script

| group | script | verdict | worst finding |
|---|---|---|---|
| mappope | kernel_probe.py | CORRECT WITH FIX | B2 |
| mappope | probe_stats.py | CORRECT WITH FIX | M-MP2 |
| recurrence | common.py | REPRODUCED | -- |
| recurrence | phase_reader_soft.py | CORRECT WITH FIX | M-RC1 |
| recurrence | geometry.py | REPRODUCED (B4 is draft text) | B4 |
| recurrence | pass_strata.py | CORRECT WITH FIX | M-RC2 |
| recurrence | phase_code.py, summarize.py, attn_offsets.py, training_curves.py, grad_at_init.py (seeds 0-1), index_loop_passes.py | REPRODUCED | MINOR only |
| rank | rank_geom.py | REPRODUCED | M-RK1 |
| rank | sweep_final.py, summarise_final.py | REPRODUCED | M-RK2, random baseline MINOR |
| rank | escape_start.py | REPRODUCED | M-RK2, M-RK3 |
| rank | analyze_snap.py | REPRODUCED | M-RK4 (appendix) |
| rank | c1_runlevel.py, init_frame.py, grad_hiding.py | REPRODUCED | MINOR |
| rank | train_snap.py, train_nd_cpu.py | not re-run (training); code read | MINOR |
| sign | sign_clock.py | CORRECT WITH FIX | M-SG1, M-SG2, M-SG3 |
| sign | rank_clock.py | CORRECT WITH FIX | B3 |
| sign | path_profiles.py, latent_geometry.py, latent_reach.py, aobal_geometry.py, prop_checks.py, lattice_p.py, basin_summary.py | REPRODUCED | MINOR |
| sign | step_basin.py (+ run_basin.sh) | code correct, NOT DETERMINISTIC | M-SG6 |
| depth | depthlib.py, torus_depth.py, torus_probe.py, torus_runlen.py, walk_stats.py, summarize_torus.py, dyck_distance.py, dyck_crosstab.py | REPRODUCED | M-DP4 (uncommitted generator) |
| depth | ring_depth.py | CORRECT WITH FIX | M-DP1 |
| depth | summarize_ring.py | CORRECT WITH FIX | M-DP2, M-DP3 |
| language | tw_lib.py, tw_step_account.py, tw_attention_asides.py (subset), tw_attention_asides_dironly.py (subset), tw_clock_unit.py, tw_common_used.py, tw_alpha_drift.py, usage_tick.py, crossover_usage.py | REPRODUCED | MINOR |
| language | tw_retrace_floor.py | CORRECT WITH FIX | uncommitted third line (MINOR) |
| language | tw_strata.py | CORRECT WITH FIX | B1, M-LG1 |
| floors / causal | causal_whatwhere_T1024.py, sign_floors.py, ring_floors.py, torus_strata_floors.py, ring_strata_floors.py | REPRODUCED | MINOR only |

## 2. BLOCKERs

**B1. Text-world NEEDS-MAP is not "places reached by another route"** (`language/tw_strata.py`) **[verified]**.
`retrace_ok` predicts NOTHING when the retrace rule does not apply and counts the revisit correct when that equals the
target; the rule is never wrong when it applies (0 cases). So NEEDS-MAP = "rule does not apply AND target is an object";
non-retrace revisits on blank cells are filed as retrace-correct.
- MAIN_DRAFT2 l. 628-630 "the 5.9% that retrace gets wrong, places reached by a route other than undoing the last run":
  places reached another way are 12.3% of targets (100 walks) / 13.4% (200), best constant 0.524 / 0.506 ('nothing').
- Fix (a), relabel: "the revisits retrace gets wrong: places reached by another route that hold an object (on blank
  cells retrace's default is right); the best constant there excludes 'nothing' by construction"; report RoPE 1L's 0.002
  as "it predicts 'nothing'". Or fix (b), report the whole non-retrace stratum (200 walks): 13.4%, best constant 0.506,
  path re-scored 0.956 (fresh seeds 2-7 0.978; eval mode 0.934), RoPE 2L 0.411 (0.34-0.55, below the constant), RoPE 1L
  0.454.
- Also: MAIN_DRAFT2 l. 77 (Sec. 2 item 5), Table 7 caption and column header l. 633-640, MAIN_APPENDIX2 l. 160;
  LANGUAGE.md 2.6, 6.1, summary item 6; plan X9, C-1, Sec. 2 item 5, claim 46.

**B2. T = 1024 target coherence C / C0 by stratum are read on the wrong head** (`mappope/kernel_probe.py`, `summarise`
l. 270: `h = argmax(R["mass_cell"].mean(0))` over all revisits, then used for every stratum) **[verified: code; values
auditor]**. In rank-4 models the wrap-only revisits are carried by the other head. Re-read on the head with most same-cell
mass in each stratum:

| stratum | MapWM r2 C0 | PoPE score r2 C0 | MapWM r4 C0 | PoPE score r4 C0 |
|---|---|---|---|---|
| gap >= 128 | 0.814 -> 0.795 | 0.837 -> 0.905 | 0.993 | 0.993 |
| wrap-only | 0.457 -> 0.399 | 0.411 -> 0.482 | 0.840 -> 0.967 | 0.554 -> 0.970 |

PoPE-score r2 long-gap C 0.840 -> 0.908 (p 0.029 -> 0.005); r4 - r2 wrap C0 MapWM +0.383 (p 0.032) -> +0.568 (p 0.0009),
MapPoPE +0.143 (p 0.51) -> +0.488 (p 0.012). The theory's verdict strengthens; numbers and the caveat change.
- MAIN_DRAFT2 l. 612: "(0.81-0.84, against 0.99 at rank 4) and wrap-only revisits (0.41-0.46, against 0.84 at rank 4)"
  -> "(0.80 / 0.91, against 0.99 at rank 4) and wrap-only revisits (0.40 / 0.48, against 0.97 at rank 4 under either
  score)".
- MAIN_APPENDIX2 L2 table rows (l. ~1669-1670) and the paragraph ending l. 1675 ("C0 measured on one head is not a
  sufficient statistic"): replace values; delete the caveat (artefact of the head rule), replace by "wrap-only revisits in
  both rank-4 models are carried by the head that is not the short-gap head (C0 0.97 there)".
- MAPPOPE.md Summary 4, Theory 2 bullets and caveat; plan X7 and claim 44; Fig. 11 (c) regenerate. Code fix: choose `h`
  per stratum (`argmax(R["mass_cell"][sel].mean(0))`).

**B3. Table 4 puts two clock definitions in one column** (`sign/sign_clock.py` vs `sign/rank_clock.py`) **[verified]**.
sign rows: map = wrap(a_y), heads pooled (`sign_clock.py` l. 47); rank rows: map = unwrapped half-difference per head
(`rank_clock.py` l. 69-75). They differ by about 2x on failing codes. Under rank_clock's definition: |.| small starts
3.14-3.55, frozen-latent 2.27, signed 0.001-0.017. Under sign_clock's: unsolved rank-2 runs 0.51-0.72 (shared 0.664,
per-head 0.717, block-diagonal 0.512), solved r3/r4 0.043-0.050.
- Fix MAIN_DRAFT2 Table 4 (l. 393-404) and l. 383-386 to one definition. With sign_clock's made symmetric (M-SG1): |.|
  1.25-1.57, frozen 1.18, unsolved rank 2 0.51-0.72. With rank_clock's: |.| 3.1-3.6, frozen 2.3, unsolved rank-2 heads
  0.51-1.91. Ordering survives either way; successes <= 0.05 under both. "A clock about as large as its map" (l. 53, 382,
  698) fits rank 2 only under rank_clock's definition. Plan Table 5 / claim 22 likewise; Fig. 9 must use one definition.
- Related: SEARCH S1 Check / Sec. 0 / S4 and plan Table 5 / claim 22 "|.| 1.10-1.68" / "1.1-1.9" are no aggregate of
  the output (arm means 1.22-1.68). MAIN_DRAFT2 l. 385 already says 1.22-1.68; fix SEARCH and the plan.

**B4. "None of the 16 runs finds rank 4's two-dimensional code"** (MAIN_DRAFT2 l. 592) **[verified]**: `geometry.py`'s
own output (`GEOMETRY.txt` l. 20, 37) classifies L4 s1 as a 2D map (dim2 0.749, cancel 0.06). -> "Only 1 of the 16 runs
(L4 s1) finds a two-dimensional code (rank 4: 8/8); half end with x and y moving the phase along one direction."
RECURRENCE.md 6.1; plan claim 37 "never" -> "rarely (1/16 vs 8/8)".

## 3. MAJORs

### Rank (`rank/`)
- **M-RK1 Off-plane share holds only for the combined clock c = a + o** **[verified]** (re-ran the auditor's
  decomposition). In solved rank > D heads the mean move step a and the mean observation step o are each 0.017-0.028 of a
  map step, lie in the plane (off-plane 0.14-0.21 and 0.03-0.08) and are anti-parallel (cos -0.87 to -0.999); only the
  residual (0.003-0.008) is off-plane (0.96-1.00). SEARCH's per-pair clocks (0.034-0.058) are in-plane (0.14-0.20).
  Exceptions: lifted arm E (pair off-plane 0.97), 3D grid 18 (residual 0.0007, off-plane 0.81). Fix MAIN_DRAFT2 l. 447-449:
  add "the mean move and observation steps (each ~0.02 of a map step) lie in the plane and cancel; the small residual lies
  off it". For CPU-3: never cite a per-pair off-plane share for parking. *Coordinator's note:* the rank auditor concludes
  this "resolves X2 in SEARCH's favour"; I disagree. The drift a walk accumulates per move-plus-observation is a + o, so
  RANK's combined clock is the one that drifts the model. SEARCH's per-pair clock leaves out o and overstates the drift.
  The two agree only on what is read.
- **M-RK2 End states do not distinguish parking from removal** **[auditor]**. The latent clock shrank 14-86x (escape F),
  20-500x (E solved), 5-180x (pilot r4). MAIN_DRAFT2 l. 452 "each time by parking it off the plane" -> "the latent clock
  shrinks 14-86x; its residual (0.002-0.013 of a map step) lies off the plane (0.91-1.00)"; l. 447 "show it being used"
  -> "are consistent with it"; RANK_THEORY C2 PASS -> "consistent, not discriminating"; drop P3 "parked, not removed";
  plan claims 30-31.
- **M-RK3 Rank-sum p 0.018 depends on a post hoc statistic** **[auditor]** (MAIN_DRAFT2 l. 452-453; plan claim 31).
  Off-plane clock relative to the map step instead: solved runs rank 1st / 2nd / 8th, p 0.29; s1 also solves under plain
  continuation, without it p 0.048. Move to App. G with both caveats, or drop.
- **M-RK4 (appendix only) R2Clock's "off-plane 1.00" and its P4 count hold by construction** **[auditor]** (0.98-1.00 at
  step 0; latent clock cannot fall 10x). RANK_THEORY C6, MAIN_APPENDIX2 l. 1151, Fig. G2: drop R2Clock from the count,
  annotate its panel.

### Sign (`sign/`)
- **M-SG1 The map denominator uses one move only (rule 8)** **[verified: code l. 47; values auditor]**. With a symmetric
  map: MECH AO 1.683 -> 1.471, AO_absmag 1.632 -> 1.571, GAUGE AO_m 1.549 -> 1.559, torus AO4 1.219 -> 1.248, frozen
  latent 1.394 -> 1.181; signed unchanged. MAIN_DRAFT2 l. 399 and 542 "1.39" -> 1.18; l. 385 / Table 4 "1.22-1.68" /
  "1.55-1.68" -> 1.25-1.57 (or state the convention and give both).
- **M-SG2 "Every model that succeeds ends with clock-free heads"** (l. 382-383; SEARCH 5.1) is false without "from a small
  start" **[auditor]**: solved SO4_high 1.81 / 1.78, AO_mid s801 0.71, AO_high 0.11-0.76, SO_high 0.00-0.47. Add "from a
  small start"; plan Table 5 "Signed solutions 0.001-0.018" -> "signed, small starts". Same for l. 386 "0.001-0.072": arm
  means 0.001-0.051; individual solved heads <= 0.14 (per-head r4 s0 0.134 / 0.138).
- **M-SG3 The clock signature does not follow from S5 at |k| ~ pi** **[auditor]**. SO4_high s1206 is SOLVED with 0.99 of
  usage on channels whose per-pair clock is ~pi; revisit accuracy 1.000 in all four parity strata (3,135 revisits), though
  the correct-key score flips sign with parity. MAIN_DRAFT2 l. 206 and 388-389 ("By Proposition 2 the failing codes
  therefore serve short-gap revisits and lose long-gap ...") and l. 699; SEARCH Sec. 0, 5.1: state S5 as a bound on the
  position factor for n_x <= pi / K, drop "loses the rest" (or exclude |k| near pi), and rest the claim on the measured
  strata (l. 389-390), which are unaffected.
- **M-SG4 "Every failure ends clock-coupled" holds for arm means only** **[auditor]**. 5/22 unsolved rank-2 runs have both
  heads' clock < 0.15 (shared s1; per-head s5, s6; block-diagonal s1, s7); block-diagonal median head 0.084 vs mean 0.509;
  per-head mean 1.91 is driven by one head (12.54), median 1.13. MAIN_DRAFT2 l. 53, 382, 698, Table 4 l. 402: give counts /
  medians and name the collapsed class (plan X4).
- **M-SG5 Rank-2 "in-span 0.91-0.98" is 1 by construction unwrapped** **[auditor]** (1.0000 on all 40 r = 2 heads); it
  measures only wrapping. SEARCH S7: drop or mark "by construction"; informative at r = 3/4 only.
- **M-SG6 `step_basin.py` is not deterministic** **[auditor]**. End accuracy: S_abs s801 0.944 stored, reruns 0.509 /
  0.939; A_abs s803 0.548 -> 0.450; A_high s804 0.921 -> 0.839. SEARCH 3.4 per-seed values, "acc >= 0.9" counts and "2/8
  signed runs move off the start" do not reproduce. Fix: deterministic algorithms + 1 thread, or 2-3 reruns per cell with
  the spread. MAIN_DRAFT2 Sec. 7.3's qualitative sentence may stay, labelled exploratory; plan claims 21 and 56 likewise.
- **M-SG7 |.| clock / map >= 1 is largely by construction** (S1) for unwrapped |.| channels **[auditor]**; Table 4 caption
  should say the informative rows are frozen-latent signed and rank 2.

### Mappope / recurrence
- **M-MP1 "Code term unchanged" on short gaps is head-dependent** **[auditor]**: C0 0.900 vs 0.960 p 0.16 on the chosen
  head, +0.058 p 0.013 on the max head. MAIN_DRAFT2 l. 611 "with the code term unchanged" -> "with the code term moving
  +0.06"; MAPPOPE.md Theory 2 "entirely" -> "mostly (C +0.34 vs C0 +0.06)".
- **M-MP2 "4x less attention on action keys" is a retrieval-head statement** **[auditor]**: other head 0.25-0.31 on 6/8
  seeds; head-mean 0.223 vs 0.147 (p 0.15). MAIN_APPENDIX2 l. 1653 add "(retrieval head)"; MAPPOPE.md Summary 2, Cor. 1b;
  plan claim 41.
- **M-RC1 "Loops and depth leave the code unchanged / no better" is an unmeasured null** **[auditor]**: reader quality
  +0.027 / +0.031 vs exact-t MDE 0.080 / 0.087; long-gap MDE 0.29-0.30; and `geometry.py` shows the class shifts (1D
  cancelling 8/16 vs 0/8, Fisher p 0.022). MAIN_DRAFT2 l. 72-73 "never the code ... on an unchanged code" -> "never change
  the code family; the learned codes stay far from rank 4's (analysis)"; l. 590 "Their codes are no better than plain rank
  2's (... p 0.27-0.28)" -> "not measurably more readable (0.893 / 0.897 vs 0.866, below an MDE of 0.08)"; RECURRENCE Sec.
  0 item 1, Sec. 4; plan Sec. 2 item 4, 4.9, claim 35.
- **M-RC2 `pass_strata.py` per-head columns averaged by head index (rule 8)** **[auditor]**: at L pass 2 the retrieval head
  is head 1 on seeds 4, 6, 7. Best head per seed: "0.03 -> 0.36 -> 0.46 -> 0.52 (head 0)" -> 0.03 -> 0.50 -> 0.64 -> 0.63;
  "L4's last layer finds the right cell 64%" -> 81%; same-cell action-token mass 0.01-0.12 -> 0.02-0.15. MAIN_APPENDIX2 l.
  1711-1712; RECURRENCE Sec. 0 item 2, 5.2 table. Claim 36 survives.

### Depth (`depth/`)
- **M-DP1 The ring "global count" is a window code of ~64 moves** **[auditor]**. On T = 128 walks a t >= 8 probe cannot
  separate them; at t >= 96 displacement older than 64 moves has R^2 0.014 (3L); 3L P_t R^2 0.534 equals what the true
  d64 gives (0.508); at t >= 64 2L's P/t falls 0.60 -> 0.33. MAIN_DRAFT2 l. 344-345 "carry a global count in the residual
  stream (linear-probe R^2 0.69 for P_t / t, against 0.18 at one layer)" -> "carry a displacement code over the last ~64
  moves (R^2 0.72); displacement older than 64 moves is not decodable (R^2 0.01), so no global count is detectable on the
  ring either"; also l. 46, 62, 368 if they rest on the ring count; MAIN_APPENDIX2 F (~l. 976-977); DEPTH summary 5, 4.3,
  4.5; plan claim 8. Route A then has no positive instance.
- **M-DP2 The ring 0.75-dip "reproduction" is an identity; the script's own test fails** **[verified in
  `out/ring_depth_out.txt` l. 66-71]**: p-pooled class accuracies predict 0.678 / 0.683 / 0.749 / 0.870 vs observed
  0.710 / 0.677 / 0.824 / 1.000; retrace accuracy itself falls 0.797 -> 0.692. MAIN_APPENDIX2 F (~l. 979-980), DEPTH
  summary 6 and 4.5: drop "reproduced from class shares and class accuracies"; App. F only, descriptive.
- **M-DP3 "the floor on everything else" is false at p_plus 0.9** **[auditor]**: "neither" class (25.6%) 0.592 vs
  best-constant 0.502. MAIN_DRAFT2 l. 342 -> "and the floor on everything else at p_plus 0.5 / 0.75 (0.487 vs
  0.491-0.493); at 0.9 an unidentified +0.09 above it"; DEPTH summary 5, 4.5.
- **M-DP4 Two quoted results have no committed generator (rule 7)** **[auditor; recomputed correct]**: the fixed-gap
  segments table (MAIN_DRAFT2 l. 347 "0.874 at 2 runs and 0.493 at 17 or more") and "98.5%; median 37" (l. 349). Add both
  to `summarize_torus.py`. The table says "pooled over seeds" (MAIN_APPENDIX2 l. 902) but holds seed means: relabel, or
  switch to pooled (0.874 -> 0.896, 0.832 -> 0.785).

### Language (`language/`)
- **M-LG1 Strata on 100 walks beside a 200-walk floor** **[verified: `G = [gen(te, 1024) for _ in range(100)]`]**. On
  the 100 walks retrace scores 0.941, not 0.934 (1 - 0.934 = 6.6%). Re-run on all 200 (< 1 min): NEEDS-MAP share 5.9% ->
  6.6%; best constant 0.112 -> 0.088; path all 8 0.920 -> 0.925; fresh 2-7 0.961 -> 0.967; seeds at ceiling 0.944-1.000 ->
  0.936-1.000; RoPE 2L 0.138 -> 0.129; RoPE 1L 0.002 -> 0.004; Table 7 "all" 0.994 / 0.779 / 0.507 -> 0.993 / 0.772 /
  0.505. MAIN_DRAFT2 l. 77, 628-640; MAIN_APPENDIX2 l. 160; plan C-1, claim 46. (The Table 7 caption discloses the 100
  walks; the text's "below retrace" and "0.934" use the 200.) Apply together with B1.

## 4. MINOR (selected; full list in Parts 1-6)

- Causal: "92.5% of the score's variance" (MAIN_DRAFT2 l. 718) is SS(P)/SS(S); R^2 = 0.923 -> "92% (R^2)" or label the
  ratio. "-0.032" (l. 717): add "8/8 seeds lower, sign-flip p 0.008, uncorrected (analysis); Holm over 24 contrasts 0.08".
  Spearman 0.84 (l. 720): commit its computation. Plan l. 299-306 and 523 stale: softplus 0.780 / 0.476 / 0.675, RoPE
  0.731 / 0.479 / 0.439 (intact / identity removed / gain). Claim 54 "|.| models depend" -> "7/8 |.| runs lose >= 0.15".
  Intervention names: "token identity within type removed"; "per-pair gain plus offset".
- `APPENDIX_draft6.md` l. 156-157 and its draft-2 counterpart: maps 10-25 are not "the training maps" (SIGN_MATCHED trains
  on 0-7; refit gives 0.5747 vs 0.5749).
- Rank: random baseline is an RMS, not a median: MAIN_DRAFT2 l. 449 "0.50-0.71" -> "0.40-0.71 (median)". C1 double count:
  l. 450 "57 of 62 runs, against 28.7" -> "50 of 55, against 24.7" (p unchanged); r_beta medians 0.79-1.10. C4 14 / 3 ->
  11 / 5. C7: quote p 0.073, not 0.012.
- Depth: l. 351 "exactly the measured 0.943" is an identity -> "decomposes as". Fig. 8: ring rows include wrap-only
  revisits (2L without them 0.910 / 0.828 / 0.742 vs 0.891 / 0.802 / 0.678); its data come from uncommitted `.npz`, not
  the JSONs the plan names; label the 1L curve eval mode.
- Language: Spearman +0.905 needs p (exact permutation 0.0046); LANGUAGE 3.4 "0.01-0.07" -> 0.003-0.074; "converged
  seeds" label on the text-world rho range -> "23 of 24 models (tw s0 2.5)"; LANGUAGE 2.4 "none of that kind" false for
  MapWM s13; the post hoc list in LANGUAGE 8 should add the 0.98 cut, the 100-walk subset, the blank exclusion and the
  lift sizes.
- Mappope: kernel argmax ran over even offsets where no observation key sits; on odd D RoPE peaks 3 (569), 7 (342), 11
  (156), 15 (21) -> MAIN_APPENDIX2 l. 1659 "3 (414), 7 (277), 10-11 (147) and 2 (168)" -> those values; ~60 contrasts
  without Holm (the p 0.03-0.04 ones do not survive); "replicating the registered +0.115" -> "consistent with" (same
  checkpoints, 40 of the same walks).
- Recurrence: reader under-fits on some runs (L long-gap margin probably 0.03-0.05 too large); model on 100 walks vs
  reader on 50 (<= 0.004); "one pass 0.381" is a mean (7/8 below the floor).
- Sign: basin_summary's 3 trailing p-values are not script output (they reproduce); under Holm two fail 0.05;
  path_profiles "monotone in 5/8" -> 4/8; lattice_p tolerance 0.05 not 0.3; latent_geometry's MECH init values unsourced
  (correct); SEARCH Sec. 0 "unsolved signed runs have the relation in place" false for s1300 (0.68).
- MAIN_DRAFT2 l. 390 "70-93% / 2-10%" is not from these scripts; its source is `fig_wrap_share` (MAIN_APPENDIX2 l.
  1246-1253, Fig. H1), outside this audit's scope and not audited.

## 5. Post hoc choices to state (rule 7 / CLAUDE.md 6)

Already flagged by the reports: LANGUAGE's 0 mod pi rule. To add: the best-head rule (kernel_probe), 40 of 100 walks at T
= 1024 (labelled); r_beta < 0.5 cutoff, the 10x criteria and the escape-start statistic (rank); geometry.py's class
thresholds (stable over wide ranges); sign_clock's 0.1 / 0.05 rad cuts; lattice_p tolerance; path_profiles' rise
statistic; the depth run-length bins; the text-world 0.98 ceiling cut, 100-walk subset, NEEDS-MAP blank exclusion and lift
sizes; the causal Spearman. None of the BLOCKERs depend on a threshold choice.

---

# Part 1. Section: mappope_recurrence

## Script audit: expansion/scripts/mappope and expansion/scripts/recurrence (CPU-9, results-blind, 2026-10-21)

Auditor scope: every script in `docs/paper/expansion/scripts/mappope/` and `.../recurrence/`; reports MAPPOPE.md,
RECURRENCE.md, SYNTHESIS_DRAFT2_PLAN.md (PAPER_LEVEL.md quotes none of these numbers). During the audit
`docs/paper/MAIN_DRAFT2.md` and `MAIN_APPENDIX2.md` appeared (untracked, written 17:08-17:09 by another session, not by
this audit); they already carry several of the affected numbers, so their line numbers are given beside each fix.
Nothing in the repo was written: all re-runs used copies in this scratch directory with output paths patched
(`git status` before and after shows only the two `runs/srope_torus_pilot/*.md` plus the two new draft files above).
Checkpoints: md5 of all 104 checkpoints loaded (paper2x2 4 arms x 8, score_rank 40, rank_mi/rank_matched_e900 A and C,
loop_rank L and L4): **no duplicates**. Seeds are the registered ones (paper2x2 0-7; SCORE_RANK r2 10-21, r4 10-17, no
pilot dirs in `p0`; LOOP_RANK 0-7, A -> `rank_matched_e900` as registered). Scratch helper scripts: `mp/perhead_all.py`,
`mp/perhead_p2.py`, `mp/odd_argmax.py`, `rc/spot_phases.py`, `rc/reader_sens.py`.

### Summary

| script | verdict | worst finding |
|---|---|---|
| mappope/kernel_probe.py | CORRECT WITH FIX (re-run bit-identical) | BLOCKER: T=1024 C/C0 by stratum read on the head chosen over ALL revisits; wrap-only revisits are served by the other head. r4 wrap C0 0.840 / 0.554 -> 0.967 / 0.970; the "C0 not sufficient" caveat is an artefact |
| mappope/probe_stats.py | CORRECT WITH FIX (re-run identical) | MAJOR: "4x less attention on action keys" holds only in the retrieval head (head-mean 0.223 vs 0.147, p 0.15) |
| recurrence/common.py | REPRODUCED | strata = eval_rank_strata.kinds_vec exactly; forward exact |
| recurrence/phase_reader_soft.py | CORRECT WITH FIX (re-run, max diff 1.9e-5) | MAJOR: "code unchanged / no better" is a null at n=8 (MDE 0.08 all, 0.29-0.30 long); the geometry shows the learned code class DOES shift |
| recurrence/geometry.py | REPRODUCED (identical) | BLOCKER (text, not code): draft says "None of the 16 runs finds rank 4's 2D code"; the script's own table has L4 s1 = 2D map (1/16) |
| recurrence/pass_strata.py | CORRECT WITH FIX (2 seeds x 2 arms identical) | MAJOR: per-head columns averaged by head INDEX across seeds (rule 8); "0.03 -> 0.36 -> 0.46 -> 0.52 (head 0)" -> 0.03 -> 0.50 -> 0.64 -> 0.63 (best head) |
| recurrence/phase_code.py | REPRODUCED (identical) | -- |
| recurrence/summarize.py | REPRODUCED (identical) | MINOR: model on 100 walks vs reader on 50 (effect <= 0.004) |
| recurrence/attn_offsets.py | REPRODUCED (identical) | MINOR: entropy column from 10 walks beside 100-walk columns |
| recurrence/training_curves.py | REPRODUCED (identical) | -- |
| recurrence/grad_at_init.py | REPRODUCED (seeds 0-1 identical) | -- |
| recurrence/index_loop_passes.py | REPRODUCED (identical, matches FACTORIAL to 1e-6) | -- |

---

### mappope/kernel_probe.py

**Verdict: CORRECT WITH FIX.** Computation reproduces bit-identically; one definition (head selection) gives wrong
numbers for the T = 1024 stratum table.

- What it computes vs claims. Layer-0 rebuild (scores, softmax, V, FFN, readout) for RoPE / PoPE / MapWM / MapPoPE(-Pair).
  Checked by reading: RoPE complex pairing (0::2, 1::2) matches `_apply_rope`; PoPE score `sum mq mk cos(dphi - delta)`
  matches `model_pope.py`; retrace class is line-for-line `docs/audits/2026-10-16/sign_floors.py` (opposite action =
  a ^ 1, matches ACTION_DELTAS; target 2j steps back, token offset 4j-1, verified by hand); T=1024 strata use
  `eval_rank_strata.kinds_vec` (the registered SCORE_RANK code; start cell never in `seen`, consistent with the
  environment). Coherence C = sum A cos(dphi + psi) / sum A on the most recent same-cell observation key, C0 = same with
  psi (RoPE form) or delta (PoPE form) removed: matches the report's definition. PoPE dead zone: raw `pope_delta` is
  strictly > 0 on 98-100% of entries (checked on the tensors; clamp gradient is 0 there, 1 at exactly 0, and no entry is
  exactly 0), so "dead zone" is right.
- Checkpoints: correct arms/seeds, no duplicates (md5). `score_rank` seeds come from a directory listing (`seeds=None`);
  today that resolves to the registered seeds only, but it would silently pool any pilot dir added later. Pin them.
- Logit check: `verify` = max |rebuilt - model logits| = 0.0 on every run (re-run). Per-seed T=128 accuracies match
  `runs/paper2x2/eval.log` to 4.7e-5 (4-dp rounding) on all 32 runs.
- Re-run: `kernel_probe.py paper2x2` full (scratch copy, HERE patched): `_out.txt` and `.json` **identical** to committed.
  `SR_SEEDS=10,13 kernel_probe.py score_rank` (2 seeds x 4 arms, 40 walks): all 64 stratum rows and per-seed entries equal
  the committed rows for those seeds (max diff 0).
- Statistics: summary rows are medians over seeds in `_out.txt`, means in `probe_stats_out.txt` (the reports quote means).
  Columns mix head rules: massCell, C, C0, mAct, mObsX use the "best head" (max run-mean same-cell mass over all revisits);
  massRT and massOff1 use the max over heads.
- Post hoc choices: the best-head rule; walks = 40 of 100 at T=1024 (labelled).

Findings:

1. **BLOCKER - T=1024 C / C0 by stratum use the wrong head.** `summarise` picks `h = argmax(R["mass_cell"].mean(0))` over
   ALL revisits (83% short-gap) and reads every stratum on it. Per-head re-analysis (scratch `mp/perhead_all.py`, same
   stream, same 40 walks, all 40 models; its best-all-head column reproduces the committed C0 exactly): in the rank-4
   models the other head carries wrap-only revisits (e.g. MapPoPE r4 s10 wrap C0 head0 0.019 / head1 0.989; MapWM r4 s13
   -0.054 / 0.964). Re-read on the head with the most same-cell mass **in that stratum** (max-over-heads in brackets):

   | stratum | MapWM r2 C / C0 | PoPE score r2 C / C0 | MapWM r4 C / C0 | PoPE score r4 C / C0 |
   |---|---|---|---|---|
   | gap < 128 | 0.617 / 0.900 (unchanged) | 0.960 / 0.960 | 0.926 / 1.000 | 0.997 / 1.000 |
   | gap >= 128 | 0.524 / 0.814 -> **0.538 / 0.795** (0.834) | 0.840 / 0.837 -> **0.908 / 0.905** (0.915) | 0.923 / 0.993 | 0.991 / 0.993 |
   | wrap-only | 0.184 / 0.457 -> **0.144 / 0.399** (0.537) | 0.414 / 0.411 -> **0.485 / 0.482** (0.565) | 0.789 / 0.840 -> **0.904 / 0.967** (0.969) | 0.554 / 0.554 -> **0.969 / 0.970** (0.974) |

   Contrasts (unpaired permutation): r4 - r2 wrap C0, MapWM +0.383 (p 0.032) -> **+0.568 (p 0.0009)**; MapPoPE +0.143
   (p 0.51) -> **+0.488 (p 0.012)**. PoPE - MapWM r2: wrap C0 +0.083 (p 0.64), long C0 +0.110 (p 0.32), long C +0.316
   (p 0.029) -> +0.370 (p 0.005). The theory's verdict survives and strengthens (the score leaves rank-2 C0; rank 4 raises
   it to 0.97 under both scores); the numbers and the caveat change. A direct discrimination readout supports it: share of
   earlier different-cell observation keys whose code-only coherence is >= the target's, wrap-only 0.366 (MapWM r2) /
   0.264 (PoPE r2) vs 0.000 / 0.000 at r4; gap >= 128 0.076 / 0.049 vs 0.000.
   Fixes:
   - code: in `summarise` choose `h` per stratum (`argmax(R["mass_cell"][sel].mean(0))`); report max-over-heads as a check.
   - MAPPOPE.md Summary 4: "C0 is 0.41-0.46 under both scores, and only rank 4 raises it" -> "C0 is 0.40-0.48 under both
     scores; rank 4 raises it to 0.97 under either score".
   - MAPPOPE.md Theory 2 bullets: gap >= 128 "C rises from 0.524 to 0.840 (p 0.029) ... C0 stays at 0.81-0.84" -> "C rises
     from 0.538 to 0.908 (p 0.005) ... C0 0.80 / 0.91 vs 0.99 at r4"; wrap-only "C0 is 0.457 vs 0.411, p 0.79 ... Rank 4
     raises MapWM's C0 to 0.840 (p 0.03)" -> "0.399 vs 0.482, p 0.64 ... rank 4 raises C0 to 0.967 (MapWM, p 0.0009) and
     0.970 (PoPE score, p 0.012)".
   - MAPPOPE.md caveat "In MapPoPE r4, wrap-only C0 is 0.55 ... C0 is not a sufficient statistic": delete; replace with
     "wrap-only revisits in both rank-4 models are carried by the head that is not the short-gap head (C0 0.97 there)".
   - MAIN_APPENDIX2.md L2 table rows "plain gap >= 128" / "wrap-only" (lines 1669-1670) and the paragraph after it (C0
     "0.81-0.84 and 0.41-0.46 ... rank 4 raises MapWM's to 0.99 and 0.84 ... not a sufficient statistic") -> values above.
   - MAIN_DRAFT2.md line 612: "(0.81-0.84, against 0.99 at rank 4) and wrap-only revisits (0.41-0.46, against 0.84 at rank
     4)" -> "(0.80 / 0.91, against 0.99 at rank 4) and wrap-only revisits (0.40 / 0.48, against 0.97 at rank 4 under either
     score)".
   - Plan X7: "(C0 0.81-0.84 at gap >= 128, 0.41-0.46 wrap-only; analysis)" -> "(C0 0.80-0.91 at gap >= 128, 0.40-0.48
     wrap-only, 0.99 / 0.97 at rank 4; analysis)". Plan fig_kernel_peak (c): regenerate from the per-stratum head.

2. **MAJOR - "code term unchanged" on short gaps is head-dependent.** C0 0.900 vs 0.960 is p 0.16 on the best head but
   +0.058, p 0.013 on the max head. "The repair is entirely the readout term" -> "mostly the readout term (C +0.34 vs C0
   +0.06)". MAPPOPE.md Theory 2 first bullet; MAIN_DRAFT2.md line 611 "with the code term unchanged" -> "with the code term
   moving +0.06".

3. MINOR - argmax of the content-pair kernel runs over D = 1..16 including even offsets, where an observation key never
   sits (observations are at odd offsets from an action query). Restricted to odd D (scratch `mp/odd_argmax.py`): RoPE
   3: 569, 7: 342, 11: 156, 15: 21 (all 1088 pairs on a retrace offset 4j-1); PoPE unchanged (1: 1068, 7: 16, 11: 4). Fix
   MAPPOPE.md Cor. 1c bullet 2, MAIN_APPENDIX2.md line 1659 and fig_kernel_peak (a): "3 (414), 7 (277), 10-11 (147) and 2
   (168)" -> "3 (569), 7 (342), 11 (156), 15 (21): every pair peaks on a retrace offset".
4. MINOR - "MapWM's amplitude-weighted cos(beta) ... (+0.76 vs +0.75)" (Cor. 1b) are seed-0's values. Medians over seeds:
   observation keys +0.79, action keys +0.76 (ranges +0.45..+0.87 / +0.62..+0.79).
5. MINOR - Cor. 1b "softplus(k) 0.61-1.56 action vs 0.91-1.80 observation": the ranges overlap; say "lower on every seed
   (8/8; s7 1.24 vs 1.25)".
6. MINOR - "PoPE puts 0.307 of its attention on the newest observation" is massOff1 = max over heads; label "(its more
   recency-weighted head)".
7. MINOR - "Accuracy is +0.114 (p 0.036, replicating the registered +0.115)": same checkpoints and the first 40 of the
   registered 100 walks; "consistent with", not "replicating" (rule 6).
8. MINOR - "Removing the content phase alone would lift MapWM by +0.20" (Summary 2): a coherence statement on the best
   head (C0 - C), not an accuracy prediction; say "would raise its target coherence by +0.20".
9. MINOR - docstring names `kernel_probe_out.txt` / `kernel_probe.json`; actual files carry the batch tag.

Robustness check of the T=128 headline (scratch `mp/perhead_p2.py`): C on the max head is MapWM 0.784 vs MapPoPE 0.996,
p 0.0002, every MapPoPE seed above every MapWM seed: **"C 0.70 vs 0.995 (8/8)" survives any head rule** (claim 41's
first half, MAIN_DRAFT2 line 601). R3+ separation: PoPE max 0.648 < RoPE min 0.745 (complete, as stated). Decomposition
-0.006 / -0.005 / -0.110 / -0.004 and "~80% index side" (0.126 / 0.154 = 82%) reproduced.

### mappope/probe_stats.py

**Verdict: CORRECT WITH FIX.** Re-run on the committed JSONs (scratch copy): `probe_stats_out.txt` **identical**.

- Tests: `stats_core.perm2_p`, unpaired two-sided; 8 vs 8 exact (min p 1.6e-4 printed 0.0002), 12 vs 12 Monte Carlo
  (200k). Arms share seeds and evaluation walks; a paired test would be more powerful; unpaired is conservative.
- Multiplicity: ~60 contrasts, no Holm. Reports label every p post hoc (MAPPOPE caveats). Under Holm over the full
  family the p 0.0002 contrasts survive (0.012); the p 0.03-0.04 ones (MapPoPE - MapWM accuracy +0.028 p 0.040, short-gap
  accuracy p 0.036, long C p 0.029) do not. MINOR: say so where they are quoted.
- "seeds b > a max" counts are read correctly in the report ("8/8 seeds above") except as noted for R3+ (correct there).

Finding:

10. **MAJOR - action-key attention is a retrieval-head statement.** mAct is read on the best-mass head. MapPoPE's other
    head puts 0.25-0.31 on action keys on 6/8 seeds. Head-mean: MapWM 0.223 vs MapPoPE 0.147 (p 0.15); max-head 0.269 vs
    0.253 (p 0.79). Fix: MAPPOPE.md Summary 2 "MapPoPE also spends 4x less attention on action keys" and Cor. 1b -> "in the
    head that carries same-cell retrieval, 0.041 vs 0.179 (p 0.0002); over both heads 0.147 vs 0.223 (p 0.15, not
    separated)". MAIN_APPENDIX2.md L2 table row "attention on action keys (path arms)" (line 1653): add "(retrieval head)".
    Plan claim 41: same qualifier. (The same-cell mass row 0.747 vs 0.925 is also best-head; label it.)

### recurrence/common.py

**Verdict: REPRODUCED.** `step_info` is `eval_rank_strata.kinds_vec` line for line (start cell excluded, as in the
environment's revisit mask); `stratum` = wrap if unwrapped-new, else short if lag < 128, else long (the registered
SCORE_RANK / LOOP_RANK strata). `forward_passes` vs the model's own forward, spot-check (scratch `rc/spot_phases.py`,
A/L/L4/C x seeds 0, 5): max |dlogit| **0.0**; `phases()` equals the model's path-integrator cos/sin exactly (0.0); the
geometry step vector equals the per-action phase increment along a walk to <= 3.6e-4 (fp32 cumsum).

### recurrence/phase_reader_soft.py

**Verdict: CORRECT WITH FIX** (the code computes what it says; two claims built on it must change).

- Computes: per arm and seed, a reader `sum_c w_c cos(theta_c(i) + phq[a_i] - theta_c(j) - phk[o_j]) + b[o_j]`, softmax
  over earlier observation keys, soft vote; fitted 600 Adam steps (lr 0.03) on 24 walks (np seed 777+s), scored on 50 (np
  seed 1234+s). Matches RECURRENCE T1 / 5.3.
- Checkpoints: A = rank_matched_e900 s0-7 (registered A), L / L4 loop_rank s0-7, C rank_mi Vanilla_r4mi s0-7; no
  duplicates.
- Logit check: uses `phases()`, verified exact above.
- Re-run all 4 arms x 8 seeds (scratch): max |diff| vs committed 3.9e-6 (A), 1.9e-5 (L), 5.1e-6 (L4), 4.9e-9 (C):
  REPRODUCED (CPU nondeterminism only).
- Statistics: perm2_p 8 vs 8; post hoc, labelled.

Findings:

11. **MAJOR - "the loop and depth do not find a better code / leave the code unchanged" is an unmeasured null and partly
    contradicted.** Reader quality L - A +0.027 (perm CI [-0.025, +0.078]; exact-t MDE 0.080), L4 - A +0.031 (MDE 0.087);
    long-gap +0.050 / +0.095 with MDE 0.29 / 0.30 and CI upper ends +0.24 / +0.29, the same size as the compensation
    margins (+0.32 / +0.34). Rule 5: "unmeasured", not "no better". And geometry.py shows the learned code class does
    shift (1D cancelling 8/16 vs 0/8, Fisher p 0.022). What holds: same code FAMILY (architecture), far below rank 4's
    code (p 0.0002), compensation margin far above A's (p 0.0002-0.0045). Fixes:
    - RECURRENCE.md Sec. 0 item 1 "do not find a better code" and Sec. 4 -> "do not find rank 4's code; whether their code
      is more readable by one pass is unmeasured (+0.027 / +0.031, MDE 0.08)".
    - MAIN_DRAFT2.md line 73 "change how a code is read, never the code ... on an unchanged code (analysis)" -> "never
      change the code family; the learned codes stay far from rank 4's (analysis)"; line 590 "Their codes are no better
      than plain rank 2's (0.893 and 0.897 against 0.866, p 0.27-0.28)" -> "their codes are not measurably more readable
      (0.893 / 0.897 vs 0.866, below an MDE of 0.08)". Plan Sec. 2 item 4, 4.9 ("never change the code class" is true of
      the representable class only) and claim 35 likewise.
12. MINOR - reader under-fits on some runs. 5x the steps (3000) on seeds 0, 1, 3: L long +0.033 / +0.045 / +0.059, L4
    -0.001 / +0.005 / +0.045, A -0.019 / +0.004 / -0.046. The L long-gap margin (+0.32) is probably ~0.03-0.05 too large;
    verdicts unchanged. Fix: fit to convergence (or early-stop on a held-out fit walk) and report the sensitivity.
13. MINOR - wording: the reader shares one channel weight w_c across all token pairs, so it also lacks a head's
    token-dependent channel amplitudes |q_c(x_i)||k_c(x_j)|, not only "the second head and the MLP" (docstring, T1).
    "24 training-seed walks" (RECURRENCE 5, 5.3) are walks on the held-out evaluation map (env seed 10000) with another np
    seed; say "24 separate walks on the evaluation map".

### recurrence/geometry.py

**Verdict: REPRODUCED.** Re-run (scratch, reading the committed pass_strata / reader JSONs): `GEOMETRY.txt` and
`geometry.json` **identical**.

- Computes omega-weighted per-action phase steps from the model's own `action_to_lie` and `omega` (verified equal to the
  phase increment along a walk); dim2 = sigma2/sigma1 of the antisymmetric x/y steps, cancel = common mode / map step
  (action steps only; not wrap-aware, as stated), kappa; classes 2D (cancel < 0.1, dim2 > 0.3), 1D, drifting.
- Thresholds post hoc (labelled). Stable: any cancel cut in (0.08, 0.34) and any dim2 cut in (0.002, 0.568) give the same
  classes; closest call C s0 cancel 0.08. Fisher tests post hoc (labelled).

Findings:

14. **BLOCKER (text that uses the script) - "None of the 16 runs finds rank 4's two-dimensional code".** The script's own
    table classifies L4 s1 as a 2D map (dim2 0.749, cancel 0.06, tail loss 0.011; also in RECURRENCE 5.4 "1/8"). Fix:
    RECURRENCE.md 6.1 and MAIN_DRAFT2.md line 592 -> "Only 1 of the 16 runs (L4 s1) finds a two-dimensional code (rank 4:
    8/8); half end with x and y moving the phase along one direction." Plan claim 37 "never rank 4's 2D code (1/16 vs
    8/8)" -> "rarely (1/16 vs 8/8)".
15. MINOR - `wrapch` ignores observation-token steps (each action is followed by an observation step); unused in the
    reports, but its docstring should say so.
16. MINOR - RECURRENCE T4 "latent ratio 0.000-0.006" is computed by no script in scope; cite its source or drop.

### recurrence/pass_strata.py

**Verdict: CORRECT WITH FIX.** Re-run `SEEDS=0,1 pass_strata.py A 100` and `... L 100` (8 passes): per-seed JSON
**identical** to committed. `check_forward` assertion passes; L at pass j = model with n_loops = j (exact); L4
intermediate = logit lens (labelled). Same-cell observation / action masks and the argmax-hit definition match the text.

Findings:

17. **MAJOR - per-head columns are averaged by head index across seeds (rule 8: head labels are a symmetry).** At L pass 2
    the retrieval head is head 1 on seeds 4, 6, 7 (per seed h0/h1: 0.76/0.07, 0.30/0.01, 0.70/0.05, 0.57/0.01,
    0.00/0.64, 0.52/0.00, 0.03/0.27, 0.02/0.23). Best head per seed (max over heads, mean over seeds):

    | arm | pass | mass all | argmax all | mass long | mass wrap | argmax wrap |
    |---|---|---|---|---|---|---|
    | A | 1 | 0.52 | 0.75 | 0.25 | 0.13 | 0.19 |
    | L | 1 / 2 / 3 / 4 | 0.03 / 0.50 / 0.64 / 0.63 | 0.35 / 0.86 / 0.88 / 0.92 | 0.01 / 0.15 / 0.49 / 0.60 | 0.00 / 0.05 / 0.35 / 0.41 | 0.01 / 0.19 / 0.53 / 0.57 |
    | L4 | 1 / 2 / 3 / 4 | 0.01 / 0.24 / 0.75 / 0.88 | 0.10 / 0.40 / 0.91 / 0.96 | 0.00 / 0.09 / 0.53 / 0.80 | 0.00 / 0.03 / 0.20 / 0.63 | 0.00 / 0.07 / 0.29 / 0.81 |
    | C | 1 | 0.82 | 0.95 | 0.79 | 0.74 | 0.88 |

    Fixes: RECURRENCE.md Sec. 0 item 2 and MAIN_APPENDIX2.md M2 (line 1711) "same-cell attention mass rises 0.03 -> 0.36
    -> 0.46 -> 0.52 (head 0)" -> "0.03 -> 0.50 -> 0.64 -> 0.63 (best head per seed): retrieval forms at passes 2-3; pass 4
    sharpens it (argmax 0.88 -> 0.92)"; "L4's last layer finds the right cell 64% of the time" (RECURRENCE 5.2, APPENDIX2
    line 1712) -> "81% (its best head)" (and note that the reader's 12% is observation-key cell top-1, a different
    quantity); replace the head 0 / head 1 table in RECURRENCE 5.2 by the table above. "Mass on same-cell action tokens
    stays at 0.01-0.12" -> best-head means 0.02-0.15 over passes 1-4, single seed-heads up to 0.39 (L) / 0.42 (L4). Claim 36
    ("retrieval forms at passes 2-4") survives.
18. MINOR - "One pass of L's trained weights is below the always-blank floor (0.381 vs 0.51)" is the mean; per seed
    0.09-0.67, 7/8 below (s0 0.672 above). Add the range.

### recurrence/phase_code.py

**Verdict: REPRODUCED.** Re-run all 4 arms: JSON **identical**. Reader (i) (non-negative channel kernel, top-1 cell) and
the leak ratio |omega delta(obs)| / |omega delta(action)| compute what Sec. 5.3 says. The docstring's "upper bound on what
one attention pass can get from position alone" is not proved (RECURRENCE Sec. 8 already says so for reader (ii)).

### recurrence/summarize.py

**Verdict: REPRODUCED.** Re-run on scratch JSONs: `SUMMARY.txt` **identical**. Check vs stored LOOP_RANK / RANK_MI
evaluations: max |diff| 3.1e-4 (CPU vs stored eval, a few argmax flips), as stated.

19. MINOR - model accuracy comes from 100 walks, the reader from the first 50 of the same stream. Recomputed on the same
    50 walks (scratch `ps50/`): margins change by <= 0.004 (L long +0.319 -> +0.321, L4 wrap +0.435 -> +0.436); matched
    margin-vs-A p: L long 0.0034, wrap 0.0002; L4 0.0061, 0.0002. Say "model on 100 walks" or use the matched 50.

### recurrence/attn_offsets.py

**Verdict: REPRODUCED.** Full re-run (8 seeds, 10 walks): `ATTN_OFFSETS.txt` **identical**. Offset-bin arithmetic checked.

20. MINOR - the "entropy (nats)" column of RECURRENCE 5.2's attention table comes from this script (10 walks per seed,
    head-mean) while the other columns are pass_strata on 100 walks; label it.

### recurrence/training_curves.py, grad_at_init.py, index_loop_passes.py

**Verdict: REPRODUCED.** training_curves: output identical; every Sec. 5.5 range checked against it. grad_at_init:
`SEEDS=0,1` per-seed lines identical to committed (full run ~45 min not repeated); x2.9 / coherence 0.46 -> 0.41 are the
committed 8-seed means. index_loop_passes: full re-run identical; trained-depth accuracy equals FACTORIAL_{LOOP,L4}.json
to 1e-6 on all 8 models.

---

### Which of these numbers go into draft 2 (plan Sec. 3 claims table; MAIN_DRAFT2 / MAIN_APPENDIX2 as of 17:09)

| claim / place | script | status after audit |
|---|---|---|
| 4.9 theory, Sec. 2 item 4, draft line 73 ("never the code") | phase_reader_soft, geometry | MAJOR #11 (rephrase) |
| 4.10 dead-zone 98-100% | kernel_probe | REPRODUCED |
| 34 one-layer model ~ its reader (+0.028 / -0.002) | phase_reader_soft | REPRODUCED (+0.028 / -0.002 on matched walks too) |
| 35 code unchanged (+0.027 p 0.28) and read better (+0.32-0.44) | phase_reader_soft | first half MAJOR #11; second half holds (margin may be ~0.03-0.05 smaller, #12) |
| 36 one pass 0.381; retrieval at passes 2-4 | pass_strata | holds; head-0 numbers MAJOR #17 |
| 37 1D 8/16, 2D 1/16 vs 8/8 | geometry | numbers REPRODUCED; "never" / "None of the 16" BLOCKER #14 |
| 38 gradient x2.9, coherence | grad_at_init | REPRODUCED |
| 41 C 0.70 vs 0.995; action keys 0.179 vs 0.041 | kernel_probe, probe_stats | C robust; action keys MAJOR #10 |
| 42 K(3) > K(1) 0%; 0.110 of 0.126 on R3+ | kernel_probe | REPRODUCED; comb offsets MINOR #3 |
| 43 ~80% index side | probe_stats | REPRODUCED |
| 44 / draft lines 611-612 / App L2 T=1024 table | kernel_probe | BLOCKER #1, MAJOR #2 |
| fig_kernel_peak (a) / (b) / (c) | kernel_probe | (a) MINOR #3, (b) holds, (c) BLOCKER #1 |
| App M2 / M3 / M4 | pass_strata, phase_reader_soft, geometry | MAJOR #17; M3 holds (with #11 wording); M4 holds |

---

# Part 2. Section: rank

## Script audit: docs/paper/expansion/scripts/rank/ (RANK_THEORY.md)

Auditor: independent, results-blind, CPU only, 2026-10-21. No file in the repo was written. `git status` before and after
showed only the two `runs/srope_torus_pilot/*.md` files plus, after, `?? docs/paper/MAIN_DRAFT2.md`. That file was created
16:58 by another session, not by this audit. Scratch: `.../scratchpad/script_audit/rank/` (copies of the scripts
`rerun/`, and the auditor scripts `audit_geom.py`, `audit_decomp.py`, `audit_c1.py`, `audit_escape_abs.py`,
`audit_init.py`, `audit_init2.py`, `audit_r2clock.py`).

Draft status: `docs/paper/MAIN_DRAFT2.md` Sec. 7.2 "Why the extra rank helps" (l. 444-452) already carries these numbers:
off-plane share 0.96-1.00 in 9 of 10 sets, random 0.50-0.71, readgain 0.09-0.27, 57 of 62 runs against 28.7, 16 of 16
heads "each time by parking it off the plane", and rank-sum p 0.018. The plan (SYNTHESIS_DRAFT2_PLAN) quotes the
off-plane / random / lam numbers (X2, evidence row 30, Fig. 9 side panel). PAPER_LEVEL.md quotes none of these scripts'
numbers.

### Summary

| script | verdict | worst finding |
|---|---|---|
| rank_geom.py | REPRODUCED (definitions match the report; step tables exact) | MAJOR: the off-plane share holds only for c = mean move + mean observation step. Each of those is in-plane, and the two cancel in the latent. The move-pair clocks (SEARCH's definition) are in-plane |
| sweep_final.py | REPRODUCED (bit-identical txt and json) | covered under summarise_final |
| summarise_final.py | REPRODUCED (identical) | MAJOR: "parks the clock" is not shown by end states. MINOR: the random baseline is the RMS, not the median (0.50-0.71 -> 0.40-0.71) |
| c1_runlevel.py | REPRODUCED (identical) | MINOR: escape C (continuations of rank_mi's networks) is double-counted: 57/62 -> 50/55, p unchanged |
| escape_start.py | REPRODUCED (identical) | MAJOR: "each time by parking" is not shown (the latent clock shrank 14-86x). The p 0.018 depends on which statistic is chosen and is confounded by s1 |
| init_frame.py | REPRODUCED (identical; the init rebuild matches train_variant's model exactly) | MINOR: the pooled p 0.012 is pseudo-replicated (already disclosed) |
| grad_hiding.py | REPRODUCED (identical) | MINOR: labels (docstring says C9) |
| analyze_snap.py | REPRODUCED (pilot_t128 and pilot_5d outputs identical) | MAJOR (appendix only): R2Clock's "off-plane 1.00" and its P4 count are true by construction |
| train_snap.py / train_nd_cpu.py | NOT RE-RUN (training; reading the code and the R2Clock init check only) | MINOR: docstrings |

No BLOCKER: every number quoted in the draft is reproduced exactly, except the random-baseline range, which is a
labelling issue (MINOR).

---

### rank_geom.py -- REPRODUCED

**What it computes vs the report (Sec. 1, Sec. 3).**
- The phase table is A = omega * (Z F^T).
- The axis half-steps are U_d = (A[2d] - A[2d+1]) / 2.
- The clock is c = mean over the 2D action tokens + 0.5 blank + 0.5 mean of the 16 objects.
- The channel weight is w = mean_action |q pair| x mean_obs |k pair|.
- kappa = |wrap(c) sw| / mean |U_d sw|.
- lam = |c_lat| / mean |U_lat|, measured in the native latent restricted to the head's read subspace (an orthonormal
  basis, so the metric is preserved).
- lam_perp is the least-squares residual of c_lat on the rows of U_lat.
- readgain = (|c_raw sw| / nU) / lam.
- r_beta = bhat^T G bhat / (tr G / D), with G = (U sw)(U sw)^T.

These match the report's Sec. 1 definitions and the plan's X2 description: "the clock common to all four moves plus the
mean observation step, in the native latent against U". The vocabulary layouts were checked:
- torus: actions 0-3, objects 4-19, blank 20;
- GridWorldND: actions 0..2D-1, objects 2D.., blank 2D+16;
- ring (environment_cancel): actions 0-1, observations 2-18.
`p_empty` = 0.5 everywhere. Per-head models pair each head's slice of `w_in` with `w_out[h]`, as their forward does.
Block-diagonal models apply the mask.

**Logit / step check.** rank_geom rebuilds no forward. It reads the weights, so the check that matters is the step table:
the per-head Z F^T * omega from `latent_tables` against the model's own `action_to_lie(token_emb) * omega`.
- Every checkpoint of all 25 sets (200 checkpoints) gives max |diff| <= 1.9e-6.
- R2Clock with a random clock gives 5.1e-7.
- kappa / indep against `analyze_score_rank.head_stats` on all 116 torus checkpoints: max |diff| 4.8e-7. The report's
  claim of agreement "to 4 decimals" is verified on every torus checkpoint, not just the four it names.
- R2Clock at init gives the same logits as Vanilla at the same seed (max |diff| 0.0).

**Gauge (rule 8).** The off-plane share is measured in the native latent metric, which is not invariant to the GL(r)
gauge W_in -> A W_in, W_out -> W_out A^-1. Recomputed in the balanced gauge (W_in' W_in'^T = W_out'^T W_out', which is
what weight decay favours), solved rank > D heads give medians 0.94-1.00 (native 0.81-1.00). The result is robust to the
gauge. Suggest one sentence saying the metric is native and that the balanced gauge agrees.

**Findings.**

- **MAJOR (definition; RANK_THEORY Summary 1, C2, P3; MAIN_DRAFT2 l. 444-452; plan X2 / CPU-3).** In solved rank > D heads
  the "leftover clock" c = a + o has two parts, a the mean move latent and o the mean observation latent. Each is about 2%
  of |U|, each lies in the map plane, and they are anti-parallel, so they cancel each other in the latent (the
  collective / global route). Only the uncancelled residual is off-plane. Native latent, medians per set
  (`audit_decomp.py`):

  | set | abs(a)/abs(U) | abs(o)/abs(U) | abs(c)/abs(U) | off-plane share of a | off-plane share of o | off-plane share of c | cos(a, o) |
  |---|---|---|---|---|---|---|---|
  | torus r3 | 0.021 | 0.020 | 0.004 | 0.15 | 0.07 | 0.98 | -0.988 |
  | torus r4 (rank_mi) | 0.017 | 0.017 | 0.004 | 0.19 | 0.07 | 1.00 | -0.980 |
  | torus r4 (per-head) | 0.018 | 0.017 | 0.003 | 0.16 | 0.07 | 0.99 | -0.993 |
  | torus r4 (score_rank) | 0.019 | 0.019 | 0.003 | 0.14 | 0.05 | 1.00 | -0.984 |
  | ND D2 r3 | 0.017 | 0.017 | 0.003 | 0.21 | 0.03 | 0.99 | -0.988 |
  | ND D3 G10 r4 | 0.028 | 0.027 | 0.008 | 0.35 | 0.08 | 0.96 | -0.875 |
  | ND D3 G18 r4 | 0.021 | 0.022 | 0.0007 | 0.05 | 0.06 | 0.81 | -0.999 |
  | escape F | 0.026 | 0.026 | 0.004 | 0.13 | 0.02 | 0.98 | -0.989 |
  | T = 128 r4 | 0.034 | 0.029 | 0.021 | 0.48 | 0.40 | 0.99 | -0.897 |
  | ring R2 | 0.037 | 0.032 | 0.025 | 0.47 | 0.17 | 0.99 | -0.793 |

  SEARCH's move-pair clocks c_j = z(+e_j) + z(-e_j) are 0.034-0.058 of |U|, the same order as SEARCH's own 0.045-0.065.
  Their off-plane share in solved rank > D heads on the 2D sets is 0.14-0.20 (native; 0.33-0.47 balanced). That is
  BELOW the random-direction median of 0.50-0.71. With the observation steps included, c_j + 2o, the share is 0.85-1.00
  (escape F).

  So X2 resolves:
  - The magnitudes ARE comparable. SEARCH's per-pair clock leaves out the observation step, which cancels 60-97% of it
    in the latent on the T = 1024 sets.
  - On the move clock alone, SEARCH's "removed in the latent, in-span" is right.
  - RANK's off-plane share describes only the residual of that cancellation.

  **Fix.**
  - State the clock as "mean move step + mean observation step", as the draft already does in l. 445.
  - Report the decomposition beside the share: "a and o are each about 2% of a map step, in-plane and anti-parallel
    (cos -0.98); the residual |c| = 0.003-0.004 |U| lies off the plane".
  - For CPU-3: the per-pair recomputation gives in-plane per-pair clocks (0.14-0.20). Do not report a per-pair off-plane
    share as support for parking.

- **MAJOR (claim; Summary 1-2, C2, C5, Sec. 5.1; MAIN_DRAFT2 l. 444-451 "the stored checkpoints show it being used",
  "each time by parking it off the plane").** An off-plane residual with readgain much below 1 is also what plain removal
  through the latent predicts. The gradient on the latent steps flows through F^T, so read (in-plane) components are
  removed first and the least-read direction is left last. The random-direction baseline is therefore not a
  discriminating null.

  The absolute sizes argue for removal, not for holding:
  - Escape F: |c| fell 14-86x and |c_perp| fell 15-72x in 7/8 runs (1.7x in s1, which started at 0.005)
    (`audit_escape_abs.py`).
  - Escape E, solved runs: |c| fell 20-500x.
  - Pilot r4: lam fell 66x, 180x and 5.5x.

  A parked clock, in the report's own Prop. 1(a) sense, keeps c_perp and blinds the readout. W_in's decoupled decay alone
  over the 600 escape epochs is about 5x (lr 1e-3, wd 0.05, cosine), so at least part of the shrinkage is
  gradient-driven or comes from embedding decay. End states cannot separate the two.

  **Fix.**
  - MAIN_DRAFT2 l. 450: "removes the drift in 16 of 16 heads, each time by parking it off the plane" -> "removes the drift
    in 16 of 16 heads; the latent clock shrinks 14-86x and its residual (0.002-0.013 of a map step) lies off the plane
    (share 0.91-1.00)".
  - l. 444-445: "the stored checkpoints show it being used" -> "the stored checkpoints are consistent with it (analysis):
    the small residual clock of solved rank > D heads lies off the map plane, as both parking and removal through the
    latent would leave it".
  - RANK_THEORY C2 verdict "PASS" -> "consistent; not discriminating (like C1)".
  - Drop P3's "parked, not removed".
  - The decisive tests stay G1 and the r4 wd 0 arm in plan X2.

- **MINOR.** r_beta is fitted to the wrapped channel clock. The wrap-free latent beta (exact at rank D) gives the same
  run-level count (57/62) and a pooled median of 0.079 vs 0.087 (`audit_c1.py`). No change needed.

### sweep_final.py -- REPRODUCED

- Re-run: `cd /home/prashr; PYTHONPATH=/home/prashr python3 <scratch>/rerun/sweep_final.py` (under a minute).
- Diff: `sweep_final_out.txt` is identical and `sweep_final.json` has max |diff| 0 over every field.

**Checkpoints.**
- All 25 globs resolve to registered seeds:
  - 0-7, except score_rank 10-21 / 10-17 and the ring 20-27 (`analyze_score_rank.SEEDS`, `analyze_rank_tests.RING_SEEDS`);
  - no pilot seeds (score_rank's pilot was 100-101).
- The md5 sums of the 236 files have no duplicates.
- rank_nd D3 r4 vs rank_wrap N10 D3 r4, and rank_nd D2 r3 vs rank_wrap N32 D2 r3, are weight-identical (max |diff| 0).
  The report says so and counts them once in C2. `summarise_final_out.txt` still prints both.
- The escape `--init-from` paths in `run_rank_tests.sh` match escape_start's start globs.

**SOLVED.** `stats_core.classify_run(losses)["registered"]` (final-5% loss < 0.05). The counts match the registered
results: 0/8, 2/8, 2/8, 3/12, 0/8, 6/8, 8/8, 8/8, 8/8, 1/8, 8/8, 1/8, 4/8, 8/8, ring R1 7/8, E 3/8, C 1/8, F 2/8.

### summarise_final.py -- REPRODUCED (identical output)

Every number in RANK_THEORY's C1 and C2 tables, the C3 text and the MAIN_DRAFT2 sentence matches
`summarise_final_out.txt`.

**Findings.**

- **MAJOR.** Covered under rank_geom (the decomposition, and parking vs removal).

- **MINOR (random baseline; RANK_THEORY C2 table, P3, P9; MAIN_DRAFT2 l. 447 "0.50-0.71 for a random direction").** The
  code prints sqrt((r - D)/r) and labels it the "random-direction expectation". That value is the RMS. Under a uniformly
  random direction, off^2 ~ Beta((r - D)/2, D/2):

  | r, D | RMS (as printed) | median | mean |
  |---|---|---|---|
  | r3, D2 | 0.577 | 0.500 | 0.50 |
  | r4, D2 | 0.707 | 0.707 | 0.67 |
  | r4, D3 | 0.500 | 0.404 | 0.42 |
  | r2, D1 | 0.707 | 0.707 | 0.64 |

  The observed values are per-set medians, so compare them with medians.

  **Fix.** "0.50-0.71" -> "0.40-0.71 (median for a random direction)". In the C2 table: random 0.58 -> 0.50 (r3 sets),
  0.50 -> 0.40 (3D r4 sets). Relabel "expectation" -> "RMS" if the RMS is kept. Conclusions are unchanged. A single
  head at 0.96 or above has 1-18% probability under the null, so the claim rests on set medians, as stated.

- **MINOR (C1 text).** "Clean heads of rank > D sit at r_beta ~1 (0.93-1.05 medians)" -> 0.79-1.10. ND D3 G18 r4 is
  0.79 and escape F is 1.10.

- **MINOR (Summary 1, C2 last paragraph).** "In solved rank-D heads the clock is simply gone (lam <= 0.001)" is true only
  for the clean head of a solved rank-D run:
  - lam <= 0.0005 on the T = 1024 2D sets (0.027 in one block-diagonal head; <= 0.057 at T = 128; ring R1 <= 0.0013);
  - the other head often drifts (lam 0.15-1.75; per-head s1, block-diagonal s6, score_rank s19, ND s2, ND D3 s1).

  **Fix:** "in the clean head of solved rank-D runs" (plan X3).

- **MINOR (C4).** "14 clean heads end at lam <= 0.001; the 3 drifting heads (s20 h0, s22 h0, s26 h1)" should read:
  - 11 clean heads (kappa <= 0.01, lam <= 0.0013);
  - 5 drifting heads: s20 h0 0.314, s20 h1 0.040, s22 h0 0.502, s22 h1 0.058, s26 h1 0.240.

  The "14" is the head count of the solved runs. The |U| claim survives: drifting 2.8-5.0 vs clean 1.6-2.56.

- **MINOR (C2 table, escape F row).** The row is "solved heads", n = 4 (2 runs). The 12 unsolved F heads give the same
  values (0.985, readgain 0.19). Say so: the residual geometry does not track SOLVED within F.

### c1_runlevel.py -- REPRODUCED (identical: 62 runs, 57 observed, 28.7 expected, P < 5e-5)

**Statistics.**
- The null is a uniform drift direction relative to each head's own frame. The 2D formula is correct: s2 = indep^2 is
  the eigenvalue ratio. For the 3D set it is an approximation; the exact D-dim null from each head's G eigenvalues gives
  28.3 expected (`audit_c1.py`).
- The 0.5 cutoff and the "most aligned drifting head" readout are post hoc (labelled ANALYSIS).
- kappa > 0.01 is the T1 basin threshold, a post hoc threshold of SCORE_RANK.

**MINOR (pseudo-replication).** "Each checkpoint counted once" de-duplicates by path. But escape C is the rank_mi
`Vanilla` s0-7 networks continued for 600 epochs: the same 8 networks enter twice. Without escape C: 50/55 vs 24.7, P <
5e-5. Without escape C and loop_rank_e1800 (which has an identical init to rank_mi): 46/49 vs 22.1.

**Fix (RANK_THEORY C1; optional in MAIN_DRAFT2 l. 448):** "57 of 62 runs (50 of 55 counting each network once), against
28.7 (24.7) expected".

### escape_start.py -- REPRODUCED (identical)

Checkpoints are the registered `rank_tests/escape/{lifted,fresh}` starts and the E / F / C ends, s0-7. The C start is
`rank_mi/p0/Vanilla_s*`, matching `run_rank_tests.sh`. The SOLVED classes match RANK_TESTS (E 3/8, F 2/8, C 1/8).

Reproduced:
- E: 6/16 drift-free heads and 3/8 runs; end off-plane 1.00 in the solved runs, 0.01-0.31 in four unsolved, 0.66 in s4.
- F: 16/16 drift-free heads; start kappa 0.28-5.86; end off-plane 0.91-1.00.
- Fisher F vs E runs, 8/8 vs 3/8: p 0.0256.
- Exact one-sided rank-sum: 1/56 = 0.018.

**Findings.**

- **MAJOR (claim; MAIN_DRAFT2 l. 450-451, RANK_THEORY C5, Summary 2).** "Each time by parking": see rank_geom. F's latent
  clock was shrunk 14-86x.

- **MAJOR (statistic; MAIN_DRAFT2 l. 450-451 "the three that solve are the three whose clock started with the largest
  off-plane share (rank-sum p 0.018)").** The statistic, off-plane share of the full-latent clock at the start, was chosen
  post hoc, and the result depends on it:
  - The off-plane clock relative to the map, |c_perp| / |U|, ranks the solved runs 1st, 2nd and 8th (p 0.29). Share is
    |c_perp| / |c|.
  - E s1's source solves under plain continuation too (C s1 SOLVED). It starts nearly clock-free (kappa 0.06 / 0.05,
    |c| / |U| 0.05), so its "largest share" is the share of a near-zero clock. Without s1: p 1/21 = 0.048.

  **Fix:** move the sentence to App. G, labelled post hoc, with both caveats. Or drop it from the main text; G2 (seeded
  escape) tests it.

- **MINOR.** The report mentions the "descending at 900 epochs" confound but not that s1 also solves in arm C. Add it.

### init_frame.py -- REPRODUCED (identical)

**Init rebuild.** train_variant.main() was run with `train()` monkeypatched to capture the model at entry
(`audit_init.py`; output in scratch, then deleted). For Vanilla s10, Vanilla_r2ph s0 (torus), Vanilla_r4mibd s2,
Vanilla_r3ph s1 and Vanilla_r2ph s2 (nd, G32), max |rebuilt - actual| is 0.0 for every tensor. No arm in the sets used
`--init-from`.

**Statistics.**
- Mann-Whitney one-sided tests are post hoc and labelled so.
- The pooled p 0.012 mixes sets whose seeds 0-7 share a base: rank_mi per-head r2 and rank_nd D2 r2 initial kappas are
  near-identical (0.69 / 0.68, 1.25 / 1.20, ...). It is not a valid p. The report discloses this.
- The set choice leaves out rank_mi shared r2, which is 0/8, with initial min-kappa 0.44-2.03. Adding it gives p 0.006,
  so the omission does not favour the claim.

**MINOR fix (RANK_THEORY C7):** quote only the independent score_rank p 0.073 as the test. Give the pooled value
descriptively. Not in the draft.

### grad_hiding.py -- REPRODUCED (identical, 16 lines)

The code matches C8. The "drift 0.4-2.7 cells/move" and "|dkappa| < 0.4%" claims hold (max 0.25%). The hide index
ranges -1 to +1 in both ranks.

**MINOR.**
- The docstring says "check C9"; the report section is C8.
- The gradient is taken in train mode (dropout on), so it is one stochastic sample of 2 batches. Say so. The verdict
  (uninformative) is unchanged.

### analyze_snap.py (+ pilot_t128/, pilot_5d/) -- REPRODUCED (both analyze_snap_out.txt identical)

Re-run on copies of the stored JSONs. The C6 table values match: losses, accuracies, kappa / indep, r4 off-plane
0.62->0.96, 0.55->0.98 and 0.81->1.00, and P4 4/6 for r4 and for R2Clock.

**Findings.**

- **MAJOR (appendix only; RANK_THEORY C6, Summary 4, plan Fig. G2).** For R2Clock the latent clock includes the fixed
  slot coordinate, which steps by 1 per token (2 per move) and never shrinks. Its off-plane share is therefore about 1 by
  construction: it was 0.98-1.00 at step 0 in all three seeds, and 1.0 on an untrained net with a random clock
  (`audit_r2clock.py`). lam cannot fall 10x (3.7 -> 0.59, 4.9 -> 0.63, 2.6 -> 0.62; t_lam = None for every head), so
  R2Clock's "4/6 heads parking first" holds by construction.

  **Fix:**
  - "in the slot (off-plane 1.00)" -> "the slot holds a fixed unit clock by construction";
  - drop R2Clock from the P4 count;
  - drop or annotate R2Clock's off-plane panel in Fig. G2.

- **MINOR (C6 P4).** The rank-2 control also has visible drift falling before lam in 2/6 heads (s3000, one snapshot
  apart). Report 4/6 (r4) vs 2/6 (r2), n = 3. Also, rank-4 lam fell 66x / 180x / 5.5x within 40 epochs. Summary 4's
  "rank 4 parks" -> "rank 4's visible drift falls one or two snapshots before its latent clock; the latent clock is then
  removed too (5-180x)".

- **MINOR.** The docstring says "checks C5-C7"; the report uses C6 and C9.

### train_snap.py, train_nd_cpu.py -- NOT RE-RUN (training)

**train_snap.py.**
- The recipe matches train.py: AdamW wd 0.05, 5% warmup plus cosine to 10% (the same lambda), clip 1.0, loss on
  revisited observations.
- `MapFormerWM_r2Clock` equals Vanilla at init (logits identical).
- Snapshots draw no RNG.

**MINOR.**
- `held_out_acc` docstring says "argmax over the observation vocabulary", but the code takes argmax over the full
  vocabulary.
- train_nd_cpu's docstring says "check C7"; the report uses C9.
- Pilot seeds 3000-3002 and 4000-4002 lie outside every registered batch (rule 29 satisfied).
- C9's "uninformative" verdict is appropriate.

---

### Statistics and post hoc choices (all scripts)

- Every number is ANALYSIS, post hoc, and labelled so in RANK_THEORY. MAIN_DRAFT2 l. 444 says "(analysis; App. G)".
- No multiplicity correction across C1-C7. C1's P is extreme, so the correction would not matter there. The E rank-sum
  is one of several possible statistics, and choosing among them is itself a multiplicity (see escape_start).
- Thresholds chosen after seeing data: r_beta < 0.5 (C1); the 10x criteria for t_kappa / t_lam (P4); the off-plane share
  as the E start statistic (C5); the kappa <= 0.01 drift threshold (inherited from SCORE_RANK, where it is flagged
  "close to circular").
- The C1 result is insensitive to the beta method and to removing duplicates.

---

# Part 3. Section: sign

## Script audit: docs/paper/expansion/scripts/sign/ (results-blind, CPU, 2026-10-21)

Scope: the 10 scripts + run_basin.sh; reports SEARCH_THEORY.md, SYNTHESIS_DRAFT2_PLAN.md (Sec. 7.1 Table 5, claims 20/22/56),
PAPER_LEVEL.md (quotes none of these numbers), and the draft that now exists, docs/paper/MAIN_DRAFT2.md (Sec. 7.1 Table 4,
Sec. 7.3; it was committed and edited by someone else during this audit). Nothing in the repo was written. `git status` was
the same before and after (only the two srope pilot files plus papers/). Scratch: `.../scratchpad/script_audit/sign/`
(rerun/ holds copies of the scripts with their outputs; check_*.py, harmon.py, swap.py, parity*.py, prop1_split.py,
pishare.py are my checks).

### Summary

| script | verdict | worst finding |
|---|---|---|
| sign_clock.py | REPRODUCED (byte-identical output), CORRECT WITH FIX | MAJOR: map = wrap(a_y) depends on which move is called y (frozen-latent 1.39 -> 1.18 if the labels are swapped or made symmetric); "1.10-1.68" in SEARCH/plan is not any aggregate (arm means 1.22-1.68); solved runs from large starts end at 0.5-1.8 |
| rank_clock.py | REPRODUCED (byte-identical), CORRECT WITH FIX | BLOCKER for Table 4/5: it uses a different map (unwrapped half-difference, per head) from sign_clock's (wrap(a_y), pooled over heads), and the two sit in one column. Also MAJOR: 5/22 unsolved rank-2 runs have both heads clock-free; in-span = 1 by construction at r = 2 |
| path_profiles.py | REPRODUCED (byte-identical out + json) | MINOR: "monotone in 5/8 seeds" should be 4/8 |
| latent_geometry.py | REPRODUCED (byte-identical out + json) | MINOR: the quoted init values (0.056; 15.6 / 5.0) are not printed by the script. Latent res/cos are not invariant under GL(r) |
| latent_reach.py | REPRODUCED | none. Signed 1.000 is by construction, and the report says so |
| aobal_geometry.py | REPRODUCED | MINOR: the same-sign share counts unused and blind channels |
| prop_checks.py | REPRODUCED (byte-identical) | MINOR: the "exact only for a 4/4 split" claim is not printed by the script. Patched run: confirmed (1130/1130 4/4 splits exact, 0 of the others) |
| lattice_p.py | REPRODUCED (both tolerances) | MINOR: the docstring says 0.3 rad but the report uses the 0.05 file; the summary says the signed unsolved runs "have the relation in place", but s1300 has 0.68 |
| basin_summary.py | REPRODUCED for the script part | MINOR: the 3 ad hoc p-values at the end of basin_summary_out.txt are not produced by the script (exact perm reproduces them: 0.0118 / 0.0016 / 0.0131). No Holm |
| step_basin.py (+run_basin.sh) | NOT DETERMINISTIC | MAJOR: per-seed end accuracies do not reproduce. S_abs s801 stored 0.944, two reruns 0.509 and 0.939; A_abs s803 0.548 -> 0.450; A_high s804 0.921 -> 0.839; S_def s800 0.8875 -> 0.888 |

Shared checks:
- Angles: I compared the scripts' rebuilt angles with the models' own step code (`action_angles`, `action_to_lie(...)*omega`) on all 136 sign checkpoints and all 48 rank checkpoints. Max error 4.5e-6 and 9.9e-7.
- Rotary pairing (interleaved 0::2 / 1::2), head-major channel order and torus pairs ((0,1), (2,3)) = (-x,+x), (-y,+y) all match model.py and environment.py.
- SOLVED: the scripts' rule (mean of the last len//20 losses < 0.05) matches `stats_core.classify_run` on every run (0 mismatches). Per-arm counts match the registered results.
- Checkpoints: no duplicated files (md5 over 208 checkpoints). The scripts pool no pilot seeds: MECH 800-807, T2 1000-1007, GAUGE 1300-1307 (pilot 1390 excluded), TORUS_START 1200-1207, WRAP_EXIST 600-607 (pilot 690 excluded), RANK 0-7. RANK_MI per-head r2 s0-1 are the registered reuse of the pilot.
- Usage proxy: the scripts weight each vocabulary token equally. Weighting by token frequency on real streams (`model_wrap_exist.usage_weights`) gives r = 0.94-0.997 with the scripts' weights and changes every arm-mean clock ratio by <= 0.04. "Up to token frequencies" holds.

### sign_clock.py -- REPRODUCED; CORRECT WITH FIX

What it computes: per inverse pair (x, y), clock s = wrap(a_x + a_y) and map m = wrap(a_y). The ratio is
sqrt(sum_c w_c s_c^2 / sum_c w_c m_c^2), with usage w normalised over all 64 channels (both heads pooled), one ratio per run, then the
arm mean. The clock is wrap-aware. Angles include omega (frequency-scaled, signed omega kept). For T2 it uses `frozen_act_emb` for the steps and `token_emb` for the content usage, which is correct (`freeze_move_latents`).

Re-run: `python3 sign_clock.py`. Output identical. JSON identical to 5.6e-17.

Findings:
1. MAJOR (rule 8, move-label symmetry). The map uses only a_y. For |.| one-sided channels it then matters which move is called y. Recomputed with map = a_x, or with a symmetric map (rms of both):
   - MECH AO 1.683 -> 1.394 / 1.471
   - AO_absmag 1.632 -> 1.601 / 1.571
   - GAUGE AO_m 1.549 -> 1.653 / 1.559
   - torus AO4 1.219 -> 1.341 / 1.248
   - **T2 SO_absmag_fz 1.394 -> 1.175 / 1.181**
   - Signed arms and AO_high: unchanged.
   Fix: use the symmetric denominator. Then the draft's "clock / map 1.39" (MAIN_DRAFT2 Sec. 7.3 and Table 4; SEARCH S5 and 5.1; plan Table 5) becomes 1.18, and "|.| from small starts" becomes 1.25-1.57 (arm means). Alternatively state the convention (y = the second move id) and give both.
2. BLOCKER for SEARCH and the plan, already fixed in the draft. "1.10-1.68 per arm mean" (SEARCH S1 Check), "1.1-1.9" (SEARCH Sec. 0 and S4 summary) and plan Table 5 "|.| from small starts: 1.10-1.68" and claim 22 "1.1-1.9 |.|". The arm means are 1.683 / 1.632 / 1.549 / 1.219, so "1.10" and "1.9" are no aggregate of the output. Fix: 1.22-1.68 (or 1.25-1.57 symmetric). MAIN_DRAFT2 already says 1.22-1.68.
3. MAJOR (scope). "Every model that succeeds ends with clock-free heads" (MAIN_DRAFT2 7.1, first sentence) and SEARCH 5.1 "every success ends clock-free (<= 0.065)" are false without "from a small start":
   - Solved SO4_high runs end at 1.81 and 1.78.
   - Solved AO_mid s801 ends at 0.71.
   - Solved AO_high runs end at 0.11-0.76 and SO_high at 0.00-0.47.
   - Solved T2 SO_same_fz runs end at 0.04-0.16.
   Fix: "every model that succeeds from a small start". The plan's Table 5 label "Signed solutions: 0.001-0.018" should become "signed, small starts (SO, SO_absmag, SO_m, SO4)".
4. MAJOR (the signature does not apply to a clock of pi). SO4_high s1206 is solved (final loss 0.049) with 0.99 of its usage on channels whose per-pair clock is |k| ~ pi; s1201 has 0.43. Revisit accuracy on 16 fresh walks at T = 1024 is 1.000 in all four parity strata of (n_-x, n_-y), 3135 revisits. In channel 22 of s1201 the phase at the correct key flips sign with the parity (cos 0.985 vs -0.92). Summed over channels, the score at the correct key is -9.3 vs +10.1 by parity, yet the attention mass on same-cell keys is unchanged. So "any code with clock rate k serves revisits with gap < pi / k and loses the rest" (SEARCH Sec. 0 and 5.1; MAIN_DRAFT2 7.1 "By Proposition 2 the failing codes therefore ... lose long-gap") is not a consequence of S5: S5 bounds the position factor only for n_x <= pi/K. A clock of ~pi in the used channels is compatible with perfect retrieval here (mechanism not identified). Fix: state S5 as a bound on the position factor, not a signature theorem; drop "loses the rest"; or exclude |k| ~ pi; or test the signature on the strata directly.
5. MINOR. The cone check (0 violations in 4,059 channels) is an identity for |.| (|a_x + a_y| >= |a_x - a_y| with equal signs). The report says so, but still lists it as a Check.
6. MINOR. "Blind" is |wrap(a_y)| <= 0.05 alone. That counts one-sided channels (a_y ~ 0, a_x != 0) as blind, against the S1 definition (both angles 0). Not quoted in the draft.
7. MINOR (framing). For unwrapped |.| channels clock/map >= 1 holds by S1, so "|.| ends with a clock as large as its map" is largely by construction for the class. The informative content is that the channels did not wrap. Label it so in Table 4 (only frozen-latent signed and rank-2 are informative rows).

Statistics: none (descriptive, n = 8 per arm). Labelled ANALYSIS (post hoc) throughout. The thresholds 0.1 / 0.05 rad are set by the script, not tuned.

### rank_clock.py -- REPRODUCED; CORRECT WITH FIX

What it computes, per head:
- latents Z = e W_in^h (shared: the latent dims head h reads; block-diagonal: masked)
- clock c_x = z_0 + z_1, c_y = z_2 + z_3; map d = half-differences
- ch-clock = sqrt(sum w (wrap(om W c_x)^2 + wrap(om W c_y)^2)) / sqrt(sum w ((om W d_x)^2 + (om W d_y)^2)), with the map UNWRAPPED and w per head
- lat-clock = (|c_x| + |c_y|) / (|d_x| + |d_y|) in the raw latent
- in-span = share of the weighted, wrapped clock inside span{map_x, map_y}

Re-run: `python3 rank_clock.py`. Output byte-identical.

Checks:
- Rebuilt per-head angles equal the model's `action_to_lie(token_emb) * omega` for every token (max 9.9e-7), for the shared, per-head (w_in rows h*r:(h+1)*r) and block-diagonal (mask) variants.
- SOLVED matches classify_run on all 48 runs.

Findings:
1. BLOCKER (Table 4 in MAIN_DRAFT2; Table 5 in the plan). The rank rows (0.51-1.91, 0.043-0.051) come from rank_clock's definition: per head, map = unwrapped half-difference. The sign rows come from sign_clock's: pooled over heads, map = wrap(a_y). The draft states one definition for the whole table. The two differ by a factor of about 2 for failing codes:
   - sign arms under rank_clock's definition: |.| small starts 3.14-3.55 (AO 3.42, AO_absmag 3.50, AO_m 3.55, AO4 3.14); frozen-latent 2.27; signed 0.001-0.017.
   - rank arms under sign_clock's definition (per run, mean over unsolved runs): shared r2 0.664, per-head r2 0.717, block-diagonal 0.512; solved r3/r4 0.043-0.050; solved r2 0.217 / 0.348.
   Fix: one definition for every row. With sign_clock's (made symmetric, finding 1 above): |.| 1.25-1.57, frozen 1.18, unsolved rank-2 runs 0.51-0.72. With rank_clock's: |.| 3.1-3.6, frozen 2.3, unsolved rank-2 heads 0.51-1.91. The ordering survives either way. "A clock about as large as its map" fits rank 2 only under rank_clock's definition. Successes are <= 0.05 under both.
2. MAJOR (claim). "Every model that fails from a small start ... ends with a clock about as large as its map" (MAIN_DRAFT2 7.1; SEARCH S7 "Every failure ends clock-coupled") holds for arm means only.
   - Unsolved runs with BOTH heads' ch-clock < 0.15: shared s1 (0.050 / 0.061; loss 0.051), per-head s5 (0.021 / 0.104) and s6 (0.038 / 0.130), block-diagonal s1 (0.018 / 0.094) and s7 (0.010 / 0.075). That is 5 of 22 unsolved runs.
   - Heads below 0.15 among unsolved runs: shared 4/16, per-head 4/12, block-diagonal 7/12. The block-diagonal median head is 0.084 against a mean of 0.509.
   - The rank-3 unsolved runs are clock-coupled (no exceptions). On the sign side, the unsolved SO_absmag run ends at 0.016.
   Fix: say "failing runs end clock-coupled or clock-free with a lost map dimension (the collapsed class, plan X4)" and give counts, not only arm means.
3. MAJOR (robustness). The per-head r2 unsolved mean 1.910 is dominated by one head (s0 h1 12.54, a head with a near-zero map). The median is 1.130 and the mean without that head 0.94. Fix: report medians (shared 0.661, per-head 1.130, block-diagonal 0.084), or the clock and the map separately.
4. MAJOR (by construction). At r = 2 = D the unwrapped in-span is exactly 1.0000 for every head (minimum 1.0000 over all 40 r = 2 heads). The reported 0.91-0.98 ("with the clock pattern inside the map span", SEARCH S7 consistency paragraph) measures only the wrapping of the clock: it is S6(a), not evidence. Fix: drop it as a measurement at r = 2, or mark it "by construction". At r = 3/4 it is informative (unwrapped 0.9987-0.9997 solved; 0.78 unsolved r3).
5. MINOR (rule 8). lat-clock is computed in the raw latent, which is not invariant under W_in -> A W_in, W_out -> W_out A^-1. The docstring promises a W_out^h version too ("with W = I and W = W_out^h"), but the code computes only W = I. In the W_out image the unsolved values are 2.00 / 2.58 / 1.38 (raw 1.72 / 3.19 / 1.63), so "latent clock 1.6-3.2" becomes 1.4-2.6. The solved r3/r4 values are unchanged (0.043-0.054).
6. MINOR. "Obs" uses the uniform mean of all observation-token latents (blank included). Frequency weighting does not change ch-clock (checked); obs was not checked.

Draft numbers from this script (MAIN_DRAFT2 Table 4):
- 0.51-1.91 (head means): correct as computed; finding 1 applies.
- Solved rank-2 clean heads 0.029-0.072 / drifting 0.94, 1.43: REPRODUCED.
- Rank 3/4 0.043-0.051: REPRODUCED.
- "Successes from small starts at 0.001-0.072" mixes arm means (0.001-0.051) with the largest clean rank-2 head. Individual solved rank-4 heads reach 0.134 / 0.138 (per-head r4 s0) and 0.102 (shared r4 s7). Fix: "arm means 0.001-0.051; individual solved heads <= 0.14".

### path_profiles.py -- REPRODUCED

It rebuilds no forward: it uses the real ring model, with the content of solved MECH SO s800-807, built by wrap_construct balanced / signed_balanced. Angles are set with `set_move_angles`; the max angle error along both straight paths is 2.4e-7 / 9.5e-7 (seed 800). Endpoint loss 0.000, so the constructions reproduce the solution. Same 64 walks for all paths.

Re-run (~15 min, 4 threads): output and json byte-identical.

Checks:
- Complete separation of the max rise (|.| min 0.60 > signed max 0.48): REPRODUCED.
- p = 2 / C(16, 8) = 1.6e-4: correct for "all above or all below".
- 0.30 -> 0.72 and 0.30 -> 0.30: REPRODUCED.

Findings:
- MINOR. "Monotone in 5/8 seeds" (SEARCH 3.3): the max rise is < 0.03 in 4/8 (s800, 801, 802, 807); s806 rises 0.21. 5/8 is the count that "descends at the first step". Fix: "monotone (rise < 0.03 nats) in 4/8; descends from the start in 5/8".
- MINOR. The rise statistic is post hoc (exploratory label present). The |.| path is longer in angle space (wrapped targets in [0, 2 pi)), so part of the "ruggedness" is path length. Say so.

Draft numbers (MAIN_DRAFT2 7.3): 0.6-1.9 vs 0.04-0.48, complete separation, 0.30 -> 0.72 / 0.30: REPRODUCED.

### latent_geometry.py -- REPRODUCED

- The init rebuild matches the stored default angles bit-exactly (MECH 800-807). The GAUGE init order (seed, env, model) matches train_gauge.py.
- Re-run: output and json byte-identical.
- res 0.044-0.053 and cos -0.994 / -0.995 (SEARCH S2): REPRODUCED.
- +1/+2 latent relation |.| 5/8, signed 7/8: REPRODUCED.

Findings (all MINOR):
- The quoted initialisation references "0.056" (cone-tight share) and "15.6 / 5.0" (1- / 2-collapsed) are not printed. The script computes inits only for GAUGE: 0.078 and 19.8 / 3.1. Recomputing the MECH inits with the script's functions gives 0.0559 and 15.6 / 5.0, so the numbers are right but unsourced. The docstring says MECH inits are rebuilt; the code does not do it. Fix: print the MECH init row.
- Range wording. "0.30-0.36" (SEARCH Sec. 0) omits AO_m 0.25; MAIN_DRAFT2 has since been corrected to 0.25-0.36. "19-22 one-sided and 14-16 blind" omits AO_m (18.8 / 13.2): use 18.8-22.2 / 13.2-16.0. For signed, "2-4 and 8" should be 2.1-4.1 and 8.1-8.9.
- Counts, cone_tight and collapse are unweighted over all 64 channels. The ACCOUNTS lesson applies: unused channels are counted. Label them so.
- res and cos are computed in the raw latent (GL(r)-dependent). Only res ~ 0 is gauge-meaningful.

### latent_reach.py -- REPRODUCED

- Gradients of the angle sums w.r.t. the latents: signed om*W; |.| om*sign(pre)*W stacked for the two latents. Correct.
- The signed share 1.000 holds by construction, and the report says "by S2(a)".
- Re-run identical. 0.339 / 0.561 and 0.285 / 0.463 (MAIN_DRAFT2 7.3 "0.34 ... 0.56"): REPRODUCED.
- MINOR: the 0.125 random baseline applies to the unweighted share only.

### aobal_geometry.py -- REPRODUCED

- Re-run identical. 0.3-11% target share, 9-13 cancelling channels carrying 92-98% of usage, same-sign share 0.77-0.86: REPRODUCED.
- MINOR: the same-sign share counts all 64 channels (blind included).

### prop_checks.py -- REPRODUCED

- Re-run byte-identical (~3 min).
- The claim "8 channels cancel exactly only when they split 4/4" (SEARCH S3 Check) is not in the output. A patched re-run on the same RNG stream: 4/4 splits exact in 1130/1130 trials; 5/3, 6/2, 7/1 and 8/0 exact in 0 (minimum error 0.0014 rad); 9 channels never (minimum 0.027). Claim CONFIRMED. MINOR: print it.
- Checks (2) and (3) quoted correctly (0.12-0.50; 80/80; 30/80 with 16 via the mirror).

### lattice_p.py -- REPRODUCED

- Both tolerance files are identical on re-run. Wrap-aware: wrap(a2 - 2 a1); lattice wrap(32 a1).
- Quoted values (relation 0.98-1.00 in 8/8 |.| runs; signed 7/8 >= 0.99; lattice in solved |.| runs 0.15 / 0.38; four unsolved runs >= 0.75) are from the 0.05 file: REPRODUCED.

Findings (all MINOR):
- The docstring says "|error| < 0.3 rad". State the 0.05 tolerance where it is quoted. Both thresholds are post hoc, and the lattice shares change a lot between them (s1303 0.99 -> 0.75; s1305 0.94 -> 0.38).
- SEARCH Sec. 0 point 3: "The runs that stay unsolved (|.| 6/8, signed 3/8) have the relation in place". Signed s1300 is unsolved with relation 0.68 (and latent res 0.751). S4 lists the signed runs correctly (s1304, s1306). Fix: "signed 2 of its 3 unsolved runs".

### basin_summary.py -- REPRODUCED (script part)

- Re-run identical except for 3 trailing lines in the committed out.txt ("0.75 vs 1 0.0122", "0.5 vs 1 0.00175", "S_def vs A_seg0.5 0.0131") that the script does not produce. Exact two-sided permutation p (C(16,8) enumerations): 0.0118 / 0.0016 / 0.0131 for those, 0.0053 / 0.0006 / 0.3175 / 0.5630 / 0.6103 for the script's contrasts. Reported values REPRODUCED.
- MINOR: add the A_seg contrasts to the script.
- MINOR: `perm_p` has a mutable default RNG, so p depends on call order.
- MINOR: the tests are unpaired although arms share the seed's content (a sign-flip paired test fits better).
- MINOR: 8 exploratory tests and no Holm. Under Holm, tau 0.75 vs 1 (0.012 -> 0.059) and S_def vs tau 0.5 (0.013 -> 0.052) no longer pass 0.05.
- The skipped A_seg_t0.25 / S_seg_t0.75 placeholders are {"skipped": true}, as documented.

### step_basin.py / run_basin.sh -- CORRECT CODE, NOT DETERMINISTIC

- Code: content frozen (requires_grad only on w_in, w_out, omega), eval mode (no dropout), the same walk stream per seed. S_abs_fz freezes W_in. Starts are set in the balanced gauge (init_angle_err 1.2e-7 stored). The A_sol pilot holds 1.000 (seed 800, 300 steps).
- Smoke test: acc0 reproduces exactly (S_def s800 0.364).
- The trajectories do not reproduce. S_def s800 step-200 loss 6.541 vs stored 6.642. End accuracy (2000 steps, 2 threads, my rerun vs stored):

  | run | stored | rerun |
  |---|---|---|
  | S_def s800 | 0.8875 | 0.888 |
  | S_abs s801 | 0.944 | 0.509, then 0.939 (two reruns, the same environment) |
  | A_abs s803 | 0.548 | 0.450 |
  | A_high s804 | 0.921 | 0.839 |

  The step-map search is bistable, and run-to-run nondeterminism on CPU picks the branch.
- MAJOR: the per-seed values and the "acc >= 0.9" counts in SEARCH 3.4 do not reproduce: S_abs 1/8 (its one >= 0.9 seed went 0.94 -> 0.51 on one rerun), A_high 1/8 (0.92 -> 0.84). Neither does the sentence "2/8 signed runs move off the start (0.94, 0.75) against 1/8 frozen-latent and 1/8 |.|". (Note: the off-start seed is s803 in all three arms.)
- Large contrasts (S_def vs S_abs +0.365; A_high vs A_abs +0.28) are much bigger than the observed swings and are likely robust. That is unverified at n = 8 per arm; I did not rerun beyond these 5 runs.
- Fix: make it deterministic (torch.use_deterministic_algorithms(True), 1 thread), or rerun each cell 2-3 times and report the spread. Drop per-seed claims. Keep only the qualitative "with the content frozen, the step map alone rarely takes the collective route from the same-direction start" (MAIN_DRAFT2 7.3), labelled exploratory. Plan claim 56 ("basin does not reach halfway") rests on 0/8 >= 0.9 at tau 0.5 (end accuracies 0.42-0.82). A rerun could change it, though no run there is near 0.9.

### Numbers from these scripts in the draft (MAIN_DRAFT2.md, as of 17:09)

- Sec. 7.1, Table 4:
  - 0.015-0.018 / 0.001: OK.
  - 1.55-1.68 / 1.22: label-dependent (finding 1).
  - 1.39: becomes 1.18 under a symmetric definition.
  - 0.31 / 0.82: OK.
  - 0.51-1.91: different definition (BLOCKER 1, rank).
  - 0.029-0.072 / 0.94, 1.43: OK.
  - 0.043-0.051: OK.
- Sec. 7.1 text: "Every model ... fails ... clock about as large", "every model that succeeds ends with clock-free heads", "0.001-0.072" and "By Proposition 2 ... lose long-gap" (MAJOR findings above).
- Sec. 7.3 (all REPRODUCED):
  - share 1.000 / 0.34 / 0.56
  - 0.6-1.9 vs 0.04-0.48
  - 0.30 -> 0.72 / 0.30
  - 0.25-0.36 vs 0.056
  - clock / map 1.39 (see above)
- "Signed steps cancel ... in 70-93% of channels and |.| in 2-10% (App. H6)" is NOT from these scripts. sign_clock's usage-weighted cancelling shares are 0.77-0.96 (signed) and 0.02-0.16 (|.|, AO4 0.16). Source unverified.

---

# Part 4. Section: depth

## Audit: docs/paper/expansion/scripts/depth/ (results-blind code audit, rule 29, post hoc analyses)

Auditor: independent agent, 2026-10-21. CPU only; scripts copied to `scratchpad/script_audit/depth/code/` with
`sys.path` and `OUTDIR` patched to scratch; nothing in the repo was written (git status before and after: only the two
`runs/srope_torus_pilot/*.md` untracked; `docs/paper/MAIN_DRAFT2.md` and `MAIN_APPENDIX2.md` appeared during the audit,
written by another session, not by this one). Per-revisit `.npz` intermediates were regenerated in scratch and deleted.

Outputs: all 15 files in `scripts/depth/out/` are committed (`git ls-files`). The per-revisit rows that
`summarize_torus.py` / `summarize_ring.py` read (`torus_depth_*.npz`, `ring_depth.npz`) are NOT committed (they went
to `$DEPTH_SCRATCH`, the author's scratchpad); they regenerate in ~2 min.

### Summary

| script | verdict | worst finding |
|---|---|---|
| depthlib.py | REPRODUCED | none (forward re-implementation logit-exact, max diff 0.0 on 16 torus + 128 ring runs) |
| torus_depth.py (priority) | REPRODUCED | MAJOR: the fixed-gap segment table and the 98.5% / median-37 figures have no committed generator; table mislabelled "pooled" (values are seed means) |
| ring_depth.py (priority) | CORRECT WITH FIX | MAJOR: the ring "global count" (P_t / t R^2 0.69) is a <= 64-move window code on T = 128 walks; the displacement before the last 64 moves has R^2 0.01 |
| torus_probe.py (priority) | REPRODUCED | none (byte-identical); MINOR on d_w target at t < w |
| torus_runlen.py | REPRODUCED | MINOR (gap bins differ from 4.1's; distribution shift already caveated) |
| walk_stats.py | REPRODUCED | MINOR (window sds from t >= w, probes include t < w) |
| summarize_torus.py | REPRODUCED | MINOR: "accounts exactly" is a definitional identity |
| summarize_ring.py | CORRECT WITH FIX | MAJOR: its own p-invariant mixture test FAILS (0.678 / 0.683 / 0.749 vs 0.710 / 0.677 / 0.824); the report quotes the identity instead. MAJOR: "floor on everything else" false at p_plus 0.9 (0.592 vs floor 0.502). MINOR: ring segment rows mix in wrap-only revisits |
| dyck_distance.py | REPRODUCED | none (byte-identical; per-seed A2f = DYCK_MDEPTH_RESULTS.json to 2.6e-7, 96 runs) |
| dyck_crosstab.py | REPRODUCED | none (byte-identical) |

No BLOCKER: every number I re-derived matches its committed output to the printed digits. The MAJOR findings change
claims, not digits.

### depthlib.py -- REPRODUCED

- `forward_capture` re-implements `WMTransformerLayer` (pre-norm, manual softmax attention, eval mode, dropout off) for
  index RoPE (plain and looped: the looped block is repeated `n_loops` times, as `MapFormerWM_RoPE_Looped.forward`) and
  for the path model (`get_position_state`). `check_logits` asserts max |diff| < 2e-4 against `m(tokens)`.
- Measured: 0.0 for all 16 torus runs (walk 0) and all 128 ring runs (20 walks).
- `runs_info`: verified by brute force on 20 torus walks (6,381 revisits): the revisit flag equals "cell seen at an
  earlier post-move step" (start cell not a visit, as the env); `lag` = moves since the LAST earlier visit; `wrap` =
  every earlier visit at a different unwrapped position (= `eval_rank_strata.kinds_vec`); `nruns` = number of maximal
  single-action runs among moves s+1..t (0 mismatches); every retrace has lag == 2 x current run length.
- Retrace definition is identical to `docs/paper/figs/torus_strata_floors.py` / `sign_floors.py` (one-level: current
  run reverses the previous maximal run, `cur <= prev`; predicts o[t - 2 cur]).

### torus_depth.py -- REPRODUCED (priority)

- Computes: per-revisit correctness, stratum (wrap-only / short < 128 / long >= 128 by `kinds_vec`, identical to the
  registered strata code), retrace, runs spanned, attention masses per pass/head, ridge probes. Matches the docstring.
- Checkpoints: `runs/factorial/{L1,L4,LOOP}/p0/{RoPE,RoPELooped,Vanilla_r4}_s{500..503}`; the FACTORIAL batch, n = 4,
  no pilot seeds; 16 distinct md5s (no shared files).
- Stored accuracy: the FACTORIAL eval stream (env seed 10000, np seed 1234 + s, 100 walks) re-generated; overall
  accuracy equals `FACTORIAL_*_STRATA.json` "all" exactly for all 16 runs (assert < 2e-3 fired for none).
- Re-run (all four arms, 100 walks) -> `summarize_torus.py` output `torus_depth_out.txt` byte-identical; JSON probes
  differ by <= 1.8e-7 (float noise), checks identical.
- Strata shares and floors consistent with `fig_factorial_strata_out.txt` (short n 105,644 = 86,736 + 18,908; wrap n
  10,558).
- Eval mode: index 1L rows are eval-mode (as the registered FACTORIAL numbers). Re-scored with `rescore_hook` (x1/0.9):
  index 1L all 0.719 -> 0.738, short 0.769 -> 0.792, retrace 0.833 -> 0.862, non-retrace 0.475 -> 0.470, long 0.480 ->
  0.474, wrap 0.482 -> 0.476. 4L / loop correctly not rescaled.

Findings:
- MAJOR (rule 7, commit the script behind every number). The "Segments at a fixed gap" table of DEPTH 4.1 (gap 32-127:
  n 240 / 503 / 4565 / 6569 / 1202; 4L 0.874 / 0.832 / 0.688 / 0.560 / 0.493; loop 0.996 / 0.911 / 0.575 / 0.499 /
  0.502; gap 8-31 4L 0.993 / 0.940 / 0.818, loop 1.000 / 0.987 / 0.768) and the "98.5% of long-gap revisits span 17+
  (median 37, min 11); wrap-only 99% (median 65)" are not produced by any committed script. I recomputed them from the
  regenerated rows: all REPRODUCE (0.9852 / 37 / 11; 0.9905 / 65). Fix: add both blocks to `summarize_torus.py` and
  re-emit `torus_depth_out.txt`. These numbers are in MAIN_DRAFT2 Sec. 6 ("0.874 at 2 runs and 0.493 at 17 or more";
  "98.5%; median 37") and plan claim 5.
- MINOR (label). That table says "pooled over seeds"; its values are means of per-seed accuracies. Pooled values differ
  for the small cells: 4L 2 runs 0.896 (not 0.874), 3-4 runs 0.785 (not 0.832). Fix DEPTH 4.1 and MAIN_APPENDIX2 F
  (line ~902): "mean over seeds" (or switch to pooled and change 0.874 -> 0.896, 0.832 -> 0.785). The 2-run cells at
  gap 32-127 are all retraces of merged runs (retrace share 1.000); say so.
- MINOR (Fig. 8 source). The plan's Fig. 8 cites `out/torus_depth_*.json` / `ring_depth.json`; those JSONs hold only
  checks and probes. The per-class accuracies live in `torus_depth_out.txt` / `ring_depth_out.txt`, built from the
  uncommitted `.npz`. Fix: have `summarize_*` write a class-accuracy JSON and point `fig_segments` at it.
- MINOR (Fig. 8 index 1L curve) is eval-mode; label it, or plot the re-scored 1L (retrace 0.862).

### ring_depth.py -- CORRECT WITH FIX (priority)

- Computes what it states. Labels: revisit / gap / wrap-only identical to `docs/paper/figs/ring_strata.py`; the revisit
  flag is asserted equal to the env mask on every walk. Retrace as torus (opposite = the other action). `lap` =
  (u_t - u_{t-32}) mod 32 == 0 (includes zero net displacement over 32 moves at p_plus 0.5, not only true laps).
- Checkpoints: `runs/cancel/p0/{RoPE_L1,L2,L3,Vanilla_L1}_p{0.5,0.75,0.9,1.0}_s{0..7}`, 128 runs, 128 distinct md5s.
  `ring_depth.py` processes seeds 0-7; every table/probe in `summarize_ring.py` filters to seeds 2-7 (checked every code
  path: `acc`, shares, mixture, attention, probes). No pilot pooling.
- Stored accuracy: overall accuracy equals each run's `eval.json` "128" for all 128 runs; logits max diff 0.0.
- Re-run (all 128 runs) -> `ring_depth_out.txt` byte-identical; JSON probes within 5.7e-7.
- Eval mode: 1L re-scored (x1/0.9), fresh seeds: all 0.710 / 0.677 / 0.824 -> 0.719 / 0.688 / 0.817; retrace 0.797 /
  0.692 / 0.677 -> 0.811 / 0.715 / 0.674; lap and neither move <= 0.022. The 0.75 dip survives rescaling.

Findings:
- MAJOR (claim). "Three index layers ... their residual stream carries a global count (P_t / t R^2 0.69, P_t 0.60);
  route A is found on the ring" (DEPTH Summary 5, 4.3 table, 4.5; MAIN_DRAFT2 Sec. 6 "carry a global count in the
  residual stream (linear-probe R^2 0.69 for P_t / t, against 0.18 at one layer)"; MAIN_APPENDIX2 F lines ~976-977;
  plan claim 8). On T = 128 walks the probe positions t >= 8 put half of P_t / t's variance at t <= 32, and for t < 64
  the d64 target equals P_t, so the probe cannot separate a count from a recent-window code. Check (my script
  `ring_probe_late.py` / `ring_probe_old.py`, p_plus 0.5, fresh seeds 2-7, best pass, same ridge):
  - restricted to t >= 64: 3L P 0.591, P/t 0.590; 2L 0.332 / 0.332 (vs 0.60 for P/t at t >= 8); 1L 0.066 / 0.061.
  - at t >= 96, targets that a window code cannot supply: displacement before the last 32 moves P_{t-32}: 3L 0.313,
    2L -0.033; before the last 64 moves P_{t-64}: 3L 0.014, 2L -0.098, 1L -0.116. Window codes at t >= 96: 3L d32 0.845,
    d64 0.720.
  - oracle: the true d64 alone gives R^2 0.508 for P_t at t >= 96, so 3L's P_t R^2 0.534 is what its 64-move window code
    predicts.
  The 3L ring model carries a displacement code reaching ~32-64 moves back, not a global prefix count. Fix: replace "carry
  a global count in the residual stream (R^2 0.69 for P_t / t)" by "carry a displacement code over the last ~64 moves
  (d64 R^2 0.72 at t >= 96; displacement older than 64 moves R^2 0.01): no global count is detectable on the ring either,
  but on T = 128 walks a 64-move window covers most revisits". The torus/ring contrast "count found on the ring, not on the
  torus" must become a contrast in window reach (torus 4L d64 0.77, d256 0.37 at T = 1024). The route-A / route-B
  dichotomy then has no positive instance of route A. Section 3.5's table row "count found at 2-3" and summary item 5
  change accordingly. (The 1/t "normaliser" R^2 0.89 is likewise dominated by small t; MINOR.)
- MINOR (definition). Ring segment rows ("neither, 3-4 / 5-8 / 9-16 runs") include wrap-only revisits (29% / 47% / 56% of
  each bin at p_plus 0.5); the torus segment rows are plain short-gap only. Plain-only ring values, p_plus 0.5: 2L 0.910 /
  0.828 / 0.742 (reported 0.891 / 0.802 / 0.678), 3L 0.999 / 0.996 / 0.988 (reported 0.997 / 0.993 / 0.972). The
  conclusions (2L decays, 3L flat) hold. Fix: state the population, or split by plain / wrap-only, before Fig. 8 puts ring
  and torus curves side by side.

### torus_probe.py -- REPRODUCED (priority)

- Ridge probes (standardised, lambda 0.01 n, train walks 0-29, test 30-39, action positions t >= 16) on L1 / L4 / LOOP
  seeds 500-503, FACTORIAL eval stream. Uses `forward_capture` (logit-exact). Re-run output byte-identical.
- Every number in DEPTH 4.3's table matches the best pass of `torus_probe_out.txt` / `torus_depth_out.txt`.
- The negative ("no global count on the torus", P/t <= 0.21, P <= 0.10) is not touched by the confound above (a window
  confound can only inflate a count probe).
- MINOR: d_w targets at t < w are the displacement since the start (d256 at 24% of positions), while `walk_stats.py`'s sds
  use t >= w, so the rmse-in-cells conversion (0.5 / 1.8 / 7.1 / 25 cells) mixes populations; at w <= 16 irrelevant, at
  w = 256 approximate. Fix: restrict d_w probes to t >= w, or say so.

### torus_runlen.py -- REPRODUCED

- Walks with direction uniform over 4 and run length U(1..10) (= the env's generator) or fixed 2 / 5 / 10, the fixed
  held-out map (env.obs_map, seed 10000, as the eval stream), np seed 7000 + s; plain revisits (not wrap-only);
  segments / gap from `torus_depth.labels`. Re-ran all four arms at the stated walk counts (L1 60, L4 / LOOP 30, PATH 10):
  all four outputs byte-identical. Every number in DEPTH 4.4 / summary 8 checks against them.
- MINOR (post hoc bins): gap bins here are 2-15 / 16-63 / 64-127 / 128+, in 4.1 2-7 / 8-31 / 32-127 / 128-511; cell
  filter >= 50. Doubling bins, plausibly a priori, but chosen after the 4.1 tables were read; state it.
- Fixed-run walks also change maximal-run statistics (equal-direction consecutive runs merge); already caveated as a
  distribution shift ("suggestive").

### walk_stats.py -- REPRODUCED

- Output byte-identical. Window sds and excursion sds as stated. See the MINOR under torus_probe.

### summarize_torus.py -- REPRODUCED

- Byte-identical. Class means are means of per-seed accuracies; shares pooled.
- MINOR (claim wording). "This accounts exactly for 'just above retrace': 0.943 = (0.680 x 0.996 + 0.148 x 0.699) /
  0.828" (DEPTH Summary 4, 4.1; MAIN_DRAFT2 Sec. 6 "exactly the measured 0.943"; plan claim 6) is a definitional
  identity (short = retrace + not-retrace), so "exactly" carries no evidence. The informative part holds: attributing
  4L's margin over the retrace floor (0.943 - 0.912 = 0.031) by class against each class's blank floor gives 3-4 runs
  +0.0153, 5-8 +0.0152, 9-16 +0.0034, 17+ -0.0002, retrace -0.0030 (total 0.0306), i.e. the 3-8-run returns carry the
  margin. Fix: "decomposes as" in place of "accounts exactly" / "exactly the measured"; optionally cite the attribution.

### summarize_ring.py -- CORRECT WITH FIX

- Byte-identical. Fresh seeds only.
- MAJOR (claim). The script's own test of a p-invariant account (per-class accuracies pooled over p) FAILS: predicted
  a1 0.678 / 0.683 / 0.749 / 0.870 vs observed 0.710 / 0.677 / 0.824 / 1.000 (`ring_depth_out.txt`). DEPTH 4.5 and
  MAIN_APPENDIX2 F (lines ~979-980) instead quote the per-p share x accuracy sum (0.709 / 0.678 / 0.823), an identity, and
  DEPTH summary 6 calls it "reproduced from class shares and class accuracies". Class accuracies depend on p (retrace
  0.797 / 0.692 / 0.677; lap 0.497 / 0.987 / 0.994; neither 0.487 / 0.487 / 0.592). From 0.5 to 0.75 the dip
  (-0.033) = share shift at 0.5 accuracies (-0.067) + accuracy change at 0.75 shares (+0.036: retrace -0.052, clock
  learned +0.088, neither 0.000). Fix: drop "reproduced"; report that a p-invariant class mixture does not reproduce a1;
  and add that retrace accuracy itself falls 0.797 -> 0.692 (as large as the net dip). The registered "no mechanism
  sentence" stands; App. F only, descriptive.
- MAJOR (claim). "One index layer learns exactly retrace and the clock ... and the floor on everything else (0.49-0.59)"
  (DEPTH summary 5, 4.5; MAIN_DRAFT2 Sec. 6). At p_plus 0.9 the 'neither' class (25.6% of revisits) scores 0.592 (re-scored
  0.570) against its best-constant floor 0.502 (measured here; argmax blank), +0.09; runs 3-4 0.619. A 64-move offset
  does not explain it (3.7% of the class, accuracy 0.485 there). Fix: "and the floor on everything else at p_plus 0.5 /
  0.75 (0.487 vs floor 0.491-0.493); at 0.9 an unidentified +0.09 above the floor (0.592 vs 0.502)".

### dyck_distance.py -- REPRODUCED

- Scores the registered L32 D12 cell (512 sequences, `cell_with_mask`, closer-feasible prefixes) with pc / (pc + pw), as
  A2f. Checkpoints `runs/dyck_mdepth/T12/{arm}_s0..7` (all 8 seeds are the registered batch; no pilot), distinct md5s.
- Per-seed A2f equals `DYCK_MDEPTH_RESULTS.json` per_seed for all 96 runs (max diff 2.6e-7), which is stronger than the
  docstring's by-hand check of arm means.
- Definitions: distance = t - (input index of the top opener) (0 = opener is the previous token; distance bins start at
  2 because nonzero distances are even); sibling blocks = closed child blocks under the top opener (0 <=> distance 0);
  stack depth before token t. Correct.
- Re-run: `dyck_distance_out.txt` and `dyck_distance.json` byte-identical. DEPTH 4.6 and summary 7 numbers match.

### dyck_crosstab.py -- REPRODUCED

- Byte-identical. Cells with < 10 prefixes set to NaN per seed; n are per-sequence counts (same 512 sequences per seed).
  DEPTH 4.6 cross-tab numbers match. Descriptive; acknowledged as not separating the routes.

### Statistics and labelling (all scripts)

- No tests anywhere; all tables are descriptive means over n = 4 (torus), 6 fresh (ring) or 8 (Dyck) seeds, with ranges.
  Every number is labelled ANALYSIS (post hoc) in DEPTH_THEORY and "(analysis)" in the plan's claims table; MAIN_DRAFT2
  Sec. 6 says "(analysis; Table 3, App. F)". Adequate. The torus n = 4 class contrasts (e.g. 4L vs loop at 3-4 runs) carry
  no test and should not be read as differences (4L 3-4 runs range 0.74-0.99).

### Numbers from these scripts that are in draft 2 (MAIN_DRAFT2 Sec. 6, MAIN_APPENDIX2 App. F, plan claims 4-11)

Correct as printed: Table 3 (all cells); retrace 0.996 / 1.000; runs decay 0.93 / 0.98, 0.74 / 0.65, 0.56 / 0.50;
0.874 / 0.493 at gap 32-127 (seed means; see label fix); attention 0.55-0.58 / 0.00-0.01 / path 0.81; 98.5%, median
37; retrace floor 0.911 and the 0.943 decomposition (wording fix); torus probes <= 0.21; ring 3L flat 0.997 / 0.993 /
0.972 (population fix); ring probe R^2 0.69 / 0.18 (digits right, interpretation wrong); ring 0.75 decomposition 0.709 /
0.678 / 0.823 (identity; wording fix); Dyck blocks table and cross-tab.

---

# Part 5. Section: language

## Script audit: docs/paper/expansion/scripts/language/ (results-blind, rule 29, 2026-10-21)

Scope: the 11 scripts in `docs/paper/expansion/scripts/language/`, the report `docs/paper/expansion/LANGUAGE.md`, the
numbers quoted in `PAPER_LEVEL.md`, `SYNTHESIS_DRAFT2_PLAN.md` (Sec. 3, X9, C-1, C-2, C-5, rows 46-53), and
`docs/paper/MAIN_DRAFT2.md` Sec. 8 / Sec. 9. That file appeared, untracked, while this audit ran. This audit did not write it.

Method: every script was copied to the scratch dir, with output paths and `sys.path` pointed at scratch and threads capped
at 4. Everything ran on CPU, from `/home/prashr` with `PYTHONPATH=/home/prashr`. Nothing was written to the repo. The
check `git status --short | grep -v '^?? papers'`, before and after, shows only the two srope pilot files plus
`docs/paper/MAIN_DRAFT2.md` / `MAIN_APPENDIX2.md`, which another session created. Scratch:
`.../scratchpad/script_audit/language/{code,out}`.

### Summary

| script | verdict | worst finding |
|---|---|---|
| tw_lib.py | REPRODUCED | none. `check_gen` passes (token / obs / rev identical). `check_forward` max abs diff 0.0 on 5 checkpoints x 2 streams |
| tw_retrace_floor.py | CORRECT WITH FIX | the third output line (NEEDS-MAP n 179, best constant 0.1117) comes from no committed code. It was reproduced here (MINOR) |
| tw_strata.py | CORRECT WITH FIX | the NEEDS-MAP stratum is "no retrace AND the target is not blank". The report and draft gloss it as all places reached another way (12.3% / 13.4% of targets, best constant 0.52) (**BLOCKER** for the draft sentence). The stratum uses the first 100 walks, but the 0.934 floor uses all 200 (MAJOR) |
| tw_step_account.py | REPRODUCED (bit-exact, txt and json) | "aside net" and "aside noun offset" are the same number by construction (MINOR) |
| tw_attention_asides.py | REPRODUCED (subset: tw s0-s2 in order, bit-exact incl. random draws; MapWM s13, NormStep s16 deterministic parts exact) | "0-13 errors, none of that kind" is false for MapWM s13 (1 of 4) (MINOR) |
| tw_attention_asides_dironly.py | REPRODUCED (s10, s16 bit-exact) | the quarter-move lift is a post hoc variant (MINOR) |
| tw_clock_unit.py | REPRODUCED (identical) | none |
| tw_common_used.py | REPRODUCED (identical) | no p for Spearman +0.905. Exact permutation p 0.0046 (n 8). Uses the post hoc "0 mod pi" rule (MINOR) |
| tw_alpha_drift.py | REPRODUCED (identical) | "within 0.01-0.07" should read 0.003-0.074 (MINOR) |
| usage_tick.py | REPRODUCED (identical) | the text world uses retrieval-word usage and the other tasks use all-token usage. Checked: all-token gives the same 0.056-0.232 range, so this is not a finding. "Converged seeds" mislabels the range (MINOR) |
| crossover_usage.py | REPRODUCED (identical) | none |

No BLOCKER is a computation error. The one BLOCKER is a wrong definition attached to correct numbers.

---

### tw_lib.py: REPRODUCED

- `gen()` re-implements `TextWorld.generate_trajectory` (`environment_textworld.py`) and adds roles, clause index,
  directions and cells. `check_gen(3)` is True: tok, obs and rev are all identical.
- `forward()` is the one-layer MapFormerWM forward. Rotation pairs are interleaved (`_apply_rope` uses `0::2` / `1::2`),
  and `qk_mask.repeat_interleave(2)` and `view(..., dh//2, 2)` match that layout. `scale` multiplies `softmax @ V` before
  `o_proj`, the same place `rescore_hook.py` scales (DROPOUT_RESCORE.md).
- Logit check, re-run on tw s0, tw s3, NormStep s16, DirOnly s10 and MapWM s14 (np seeds 0 and 7): max abs diff 0.0 on all.
- Checkpoints: `runs/textworld/p0/Vanilla_r4_L1_s{0..7}`, `runs/tw_normstep/p0/{arm}_s{10..17}`. All 24 textworld and
  32 tw_normstep `.pt` files have distinct md5 sums. So do the TinyStories, code2048, recency and sign_matched files used
  by usage_tick / crossover.

### tw_retrace_floor.py: CORRECT WITH FIX

- It computes what it says. The retrace rule is a faithful port of `docs/audits/2026-10-16/sign_floors.py::floors_one`
  (same run / prev_run logic, copy o[t-2j]). Action ids are 0/1 = N/S and 2/3 = W/E, so `a ^ 1` is the reverse.
- Stream: held-out map 10000, np seed 0, 200 walks. This is the registered S5 stream: `analyze_textworld_secondary.floors`
  uses the same setup. The constant (0.5054) and the registered reversal-copy rule (0.5965) reproduce the registered values,
  which confirms the stream.
- Re-run (`out/retrace.log`): identical to the first two lines of `tw_retrace_floor_out.txt`. TEXTWORLD: n 6381, retrace
  0.9337. TW_NORMSTEP: n 6109, retrace 0.9365. Walk length is 132.1 full clauses per 1024 words (124-141): "~132 moves" holds.
- MINOR (rule 7). Line 3 of the committed output ("NEEDS-MAP stratum ... n 179; best constant 0.1117 ('cat'); share
  'nothing' 0.0000") comes from no committed script. It was reproduced exactly by the audit extension
  (`code/tw_strata_x.py`). Fix: add the computation to `tw_retrace_floor.py`.

### tw_strata.py: CORRECT WITH FIX

What it computes. RETRACE-OK means retrace's prediction equals the target, where retrace copies o[t-2j] if the rule applies
and otherwise predicts 'nothing'. NEEDS-MAP is the complement. Re-run of the committed per-seed and mean lines:
**identical** (path re-scored and eval mode, RoPE 1L and 2L, all 8 seeds). "Fresh seeds 2-7" is `v[2:]`, which correctly
excludes pilot seeds 0-1. Path is re-scored (x1.111) and RoPE is in eval mode, as CLAUDE.md requires for multi-layer models.

The audit extension (`code/tw_strata_x.py`, `out/strata_x.log`; 200 walks in `out/strata_x200.log`) splits revisits by
whether the rule applies and whether the target is blank:

| stream | targets | rule applies | applies but wrong | not applicable (NON-RETRACE) | NON-RETRACE blank share | NEEDS-MAP = NON-RETRACE and non-blank |
|---|---|---|---|---|---|---|
| first 100 walks (used) | 3055 | 2679 (0.877) | **0** | 376 (0.123) | 0.524 | 179 (0.059) |
| all 200 walks (registered stream) | 6381 | 5525 (0.866) | **0** | 856 (0.134) | 0.506 | 423 (0.066) |

**BLOCKER (wrong definition attached to a number in the draft).** The rule is never wrong when it applies, so NEEDS-MAP is
exactly "the rule does not apply AND the target is an object". On the blank half of the non-retrace revisits, retrace's
default 'nothing' is right by chance, so those revisits are filed under RETRACE-OK. Three consequences:

- "Places reached by a route other than undoing the last run, 5.9% of targets" is wrong. Those places are 12.3% of targets
  on 100 walks and 13.4% on 200, and their best constant is 0.524 / 0.506 ('nothing').
- "Best constant 0.112; 'nothing' never" holds by construction.
- RoPE 1L's "0.00-0.01" on NEEDS-MAP only shows that it always predicts 'nothing'. On the full non-retrace stratum it
  scores 0.477 (100 walks), at its blank share.

The sentence sits in `MAIN_DRAFT2.md` Sec. 8 ("The decisive revisits are the 5.9% that retrace gets wrong, places reached by
a route other than undoing the last run"), `LANGUAGE.md` 2.6, 6.1 and summary item 6, and `SYNTHESIS_DRAFT2_PLAN.md`
X9 / C-1 / Sec. 2 item 5 / row 46. Fix: either option.

(a) Keep the stratum and change the gloss: "the revisits retrace gets wrong: places reached by another route that hold an
object (on blank cells retrace's default is right); best constant there 0.112, which excludes 'nothing' by construction."
Report RoPE 1L's 0.002 as "it predicts 'nothing'".

(b) Report the full non-retrace stratum, which needs no outcome-defined split:

| stream | NON-RETRACE share | best constant | path re-scored (all 8) | path, fresh seeds 2-7 | path eval mode (all 8) | RoPE 2L | RoPE 1L |
|---|---|---|---|---|---|---|---|
| first 100 walks | 12.3% | 0.524 ('nothing') | 0.953 | 0.974 | 0.935 | 0.430 (0.35-0.56) | 0.477 |
| all 200 walks | 13.4% | 0.506 ('nothing') | 0.956 | 0.978 | 0.934 | 0.411 (0.34-0.55) | 0.454 |

On this stratum two index layers score BELOW the constant (0.41-0.43 against 0.51-0.52). That makes "almost nothing that
needs the map" stronger, not weaker. Path on its blank half: 0.982 / 0.985.

**MAJOR (subset mismatch).** The strata use the first 100 of the 200 registered walks, where retrace scores 0.9414, not
0.9337. So "retrace scores 0.934" and "5.9% that retrace gets wrong" describe different sets (1 - 0.934 = 6.6%). The
comparison "two RoPE 2L seeds reach 0.93, just under the 0.934 floor" mixes the sets too: on the 100 walks the floor is
0.941. Fix: re-run on all 200 walks. It costs under 1 min of CPU, and the extension already does it.

| quantity (stratum as defined) | old (100 walks) | new (200 walks) |
|---|---|---|
| NEEDS-MAP share | 5.9% (179 / 3055) | 6.6% (423 / 6381) |
| best constant | 0.112 ('cat') | 0.088 ('cat') |
| path re-scored, all 8 | 0.920 | 0.925 |
| path re-scored, fresh seeds 2-7 | 0.961 | 0.967 |
| path re-scored, seeds at ceiling | 0.944-1.000 | 0.936-1.000 |
| path re-scored, s0 / s7 | 0.603 / 0.832 | 0.617 / 0.872 |
| path eval mode, all 8 / fresh | 0.885 / 0.930 | 0.884 / 0.927 |
| RoPE 2L (eval mode) | 0.138 (0.10-0.17) | 0.129 (0.09-0.15) |
| RoPE 1L | 0.002 | 0.004 |
| "all" column: path / RoPE 2L / RoPE 1L | 0.994 / 0.779 / 0.507 | 0.993 / 0.772 / 0.505 |
| RoPE 2L s2 / s4 against the floor | 0.932 / 0.930 vs 0.941 | 0.927 / 0.926 vs 0.934 |

Alternatively, keep the 100 walks and say "first 100 walks; retrace 0.941 there" wherever the share appears. Table 7 in
MAIN_DRAFT2 already says "first 100 evaluation walks", but the text beside it quotes 0.934.

**MINOR.** The docstring says the re-score moves RoPE by <= 0.001. RoPE 1L moves up to +0.004 on "all" (s0 0.5064 -> 0.5100,
s1 0.5061 -> 0.5100). Re-scored RoPE 2L NEEDS-MAP would be 0.129, but rescaling is not valid for 2 layers. Fix: "<= 0.004
(1 layer)".

Statistics: per-seed NEEDS-MAP n is 179, so the binomial SE is about 0.015 at 0.96. All models share the same items. No test
is reported. The separation is complete: path > RoPE 2L on 8/8 seeds, exact two-sample permutation p 0.0002 at n 8 / 8.
Correctly labelled "(analysis)".

Post hoc choices: the first-100-walk subset, and the blank-excluding stratum (it follows from the rule's blank default and
is not declared).

### tw_step_account.py: REPRODUCED

- Re-run: `tw_step_account_out.txt` is identical, and the json max abs diff is 0.
- What it computes:
  - Step table: omega-scaled, per word class, divided by dN, the mean norm of the clause-level direction differentials.
  - Clause regression: the tick tau is the mean over the 4 direction means. Opposition is not wrap-aware here; the
    wrap-aware version is `oppW` in tw_attention_asides, and both give 0.001-0.013 on the 24 learned-step models.
  - Drift: consecutive visits, mean |wrap dtheta| at object slots, 1 rad threshold.
  - Usage: mean ||q_c|| over the 4 query words x mean ||k_c|| over the 17 object words. This is exact for one pre-norm
    layer with no positional embedding.
- Every range quoted in LANGUAGE 2.1-2.2 (class table, tick, 26-38 / 0-3 drift channels, agreement 0.98-1.00, r 0.96-1.00)
  was checked against the output and matches.
- MINOR. `aside_net_over_dN_usage` equals `aside_noun_offset_over_dN_usage` in every row, by construction. The noun's offset
  from the object slot is d['.'] + d[aside words] + d[noun]. The aside's net is d[aside words] + d[noun] + d['.']. The step
  is context-free, so the clause's '.' and the aside's '.' have the same step. Fix: LANGUAGE 2.1 / 5 / 6.1 should present
  "aside net 0.04-0.16" and "noun 0.04-0.16 off the place" as one measurement. It is also the persistent displacement the
  aside leaves on every later token.
- MINOR. "The tick alone predicts which channels drift (0.98-1.00, 32 models)" is close to an identity once D cancels over
  loops. On the 22 map / DirOnly models it is 0 vs 0. The informative evidence is the exact count match on the 10 clock
  models (26/26 ... 28/28). Say so.

### tw_attention_asides.py: REPRODUCED (subset)

- Re-run of tw_s0, tw_s1, tw_s2 in the committed order (`out/asides_sub2.log`). All three lines are identical, including the
  shared-rng off-plane lift vectors and the 5 random-channel draws (tw s0: 0.4269 [0.2400, 0.5807]; tw s2: 0.2514).
- Re-run of MapWM s13 and NormStep s16 out of order (`out/asides_sub.log`). The deterministic parts are identical: intact,
  x1, attention, collapse, clock removed, only clock kept. The random parts differ, as expected, because the rng stream
  depends on model order.
- Code checks:
  - Collapse puts every aside token at the object-slot phase. It sets d[a0] = C[o] - C[a0-1], zero for the rest of the
    sentence, and d[adot] = C[adot] - C[o], so the phase from adot onward is unchanged.
  - Lift adds v at a0 and subtracts it at adot.
  - The aside-error test is: pred in (aside nouns told at earlier visits to the cell) minus {target}.
  - Ablation zeros q and k in both components of each block.
- Every number in LANGUAGE 2.3-2.5 and MAIN_DRAFT2 Sec. 8 matches the committed output:
  - 19/5 split; collapse 0.952-0.975; aside errors 71-100%.
  - Clock removal <= 0.009 on 7 models, and 0.09-0.26 on 3.
  - Random controls: means 0.03-0.65.
  - Attention: 0.55-0.76 / 0.07-0.14 / 0.004-0.035 (mean 0.015) / 0.07-0.30 / 0.01-0.06.
- MINOR. LANGUAGE 2.4 says "before the manipulation they make 0-13 errors, none of that kind". MapWM s13 has 1 aside error
  of 4 (share 0.25). Fix: "0-13 errors, one of that kind (MapWM s13)".
- MINOR. The 19 / 5 split at eval-mode accuracy >= 0.98 is a post hoc threshold. It falls in a natural gap (0.960 vs 0.989)
  but is missing from LANGUAGE 8's threshold list. Add it.
- MINOR (provenance). The run was stopped, no JSON was written, and the last line of the out file was added by hand. Each
  random control has only 5 draws with a wide spread (e.g. [0.03, 0.72]).

### tw_attention_asides_dironly.py: REPRODUCED (s10, s16)

- Both lines are identical to the committed output (the rng is re-seeded per model, 100 + s).
- Claims checked against all 8 lines in the committed file:
  - intact (re-scored) 0.962-0.981, with 100% aside errors on 8/8;
  - in-plane half-move lift 0.999-1.000 on 8/8;
  - off-plane one-move lift 1.000 on 7/8 (s16 0.972);
  - off-plane quarter-move lift 0.963-0.996;
  - aside-noun attention 0.041-0.077, mean 0.047.
- MINOR. The quarter-move variant was added after the main run ("added" in the docstring), so it is post hoc. The magnitudes
  (0.5 in-plane, 1 and 0.25 off-plane) are free choices. Declare them.

### tw_clock_unit.py: REPRODUCED

- Output identical.
- The clause regression is on [4 direction one-hots, aside present, filler count, adverb present], over drift channels, on
  unwrapped omega-scaled sums. It supports "aside <= 0.062 of a tick, filler / adverb word <= 0.007".

### tw_common_used.py: REPRODUCED

- Output identical.
- The S3 costs come from the registered `TEXTWORLD_SECONDARY.json` and are eval mode. Seeds s0 and s7 are below ceiling in
  eval mode, so their costs are partly the dropout under-report.
- MINOR. Report the exact permutation p for Spearman +0.905 at n 8: two-sided 0.0046 (computed here). It uses the post hoc
  "0 mod pi" used-channel rule, which LANGUAGE 8 already flags.

### tw_alpha_drift.py: REPRODUCED

- Output identical.
- Its definitions match `docs/audits/2026-10-12/alpha_language.py`: raw pre-omega cumsum, mean range, log-log slope.
- All of LANGUAGE 3.4's table checks out:
  - map seeds: alpha 0.611-0.738, pooled rho 0.121-0.863;
  - clock seeds: alpha 0.978-1.016, rho 9.3-13.9;
  - DirOnly: alpha 0.60-0.65;
  - TinyStories: 1.00-1.01 / 11.9-12.5;
  - code: 0.99-1.01 / 2.9-5.0.
- MINOR. "alpha_pred within 0.01-0.07 of alpha; over-predicts by 0.01-0.03 on map-type": tw s6 differs by 0.003. Fix:
  "within 0.003-0.074; over-predicts by 0.003-0.03".

### usage_tick.py: REPRODUCED

- Output identical (text world, TinyStories, code).
- Every LANGUAGE 2.3 / 3.3 range checks out:
  - 0 mod pi usage 0.79-0.99;
  - clock-dominated usage 0.13-0.30 on clock models and 0.01-0.15 on map models;
  - pi channels 2-6, up to 0.242 of usage;
  - used rho 0.056-0.232 (tw s0 2.546);
  - TinyStories 9.865-12.326; code 4.483-6.642.
  MapWM s11's pi ticks are -3.140 and -3.141, as stated.
- Usage definitions differ: the text world uses retrieval-word usage, the other tasks all-token usage. Audit check
  (`code/alltok_median.py`): all-token usage on the text world gives the same usage-weighted median rho on every model
  (range excluding tw s0: 0.056-0.232). The cross-task table holds under either definition.
- MINOR. "Text world 0.06-0.23 (converged seeds)" includes 4 models below eval-mode ceiling (tw s7, MapWM s12, MapWM s14,
  NormStep s16); only tw s0 is excluded. Fix: "23 of 24 learned-step models (tw s0, unconverged: 2.5)". The same applies
  to MAIN_DRAFT2 Sec. 9.

### crossover_usage.py: REPRODUCED

- Output identical.
- Checkpoints:
  - `runs/sign_matched/p0/Signed_r4_s0-7`: config n_steps 1024, 900 ep, cosine; L = 2048 tokens is matched length.
  - `runs/recency/Signed_r4_s0-7`: k_max 64, n_symbols 16, min_gap 64, the same as the script's RecencyWorld.
  Both use the same `Signed_r4` class.
- torus 0.064-0.079 (0/64 clock channels); recency 21.8-22.1, usage 0.93-1.00.

### Draft-2 numbers that come from these scripts

| number (SYNTHESIS_DRAFT2_PLAN / MAIN_DRAFT2) | script | status |
|---|---|---|
| retrace 0.934 (0.937 TW_NORMSTEP) | tw_retrace_floor | REPRODUCED |
| 5.9% "places reached by a route other than undoing the last run" | tw_strata | **wrong gloss** (BLOCKER). With the gloss kept: 13.4% (200 walks). With the stratum kept: relabel, or use 6.6% (MAJOR) |
| NEEDS-MAP path 0.961 fresh / 0.920 all 8 / RoPE 2L 0.138 / best constant 0.112 / RoPE 1L 0.002 | tw_strata (+ uncommitted line) | reproduced on 100 walks. On the 200-walk stream: 0.967 / 0.925 / 0.129 / 0.088 / 0.004 (MAJOR) |
| Table 7 (0.994 / 0.998 / 0.920; 0.779 / 0.819 / 0.138; 0.507 / 0.538 / 0.002) | tw_strata | REPRODUCED. The column "retrace wrong" is correctly named |
| DirOnly 0.970-0.974 | TW_NORMSTEP registered (not these scripts). On stream 424242 the scripts give 0.962-0.981 re-scored | consistent |
| step table 0.004-0.03, aside 0.04-0.16 | tw_step_account | REPRODUCED (net = noun offset by construction) |
| collapse 0.952-0.975, 71-100%; lift 0.999-1.000 8/8 | tw_attention_asides(_dironly) | REPRODUCED |
| clock removal <= 0.009 (7), 0.09-0.26 (3); map removal 0.03-0.65 | tw_attention_asides | REPRODUCED |
| tick counts moves: aside <= 0.06, filler <= 0.007 | tw_clock_unit | REPRODUCED |
| Spearman +0.905 | tw_common_used | REPRODUCED (add p 0.0046) |
| agreement 0.98-1.00; 26-38 channels; 79-99% usage at 0 mod pi | tw_step_account, usage_tick | REPRODUCED (0 mod pi is post hoc) |
| alpha 0.98-1.02 (C-2) | tw_alpha_drift | REPRODUCED |
| used-channel rho 0.064-0.079 / 0.06-0.23 / 4.5-6.6 / 9.9-12.3 / 21.8-22.1 | crossover_usage, usage_tick | REPRODUCED (relabel "converged") |

Seeds: no pilot seeds are pooled in any "fresh" number. TEXTWORLD seeds 0-1 are excluded via `v[2:]`. The TW_NORMSTEP
batch uses seeds 10-17 and its pilot seed 100, which is outside the batch. No checkpoint is duplicated.

---

# Part 6. Section: floors_causal

## Script audit: floors and the causal what/where test (results-blind, rule 29 applied post hoc; 2026-10-21)

Scratch: `script_audit/floors_causal/`. Every re-run used a scratch copy with its output paths patched, CPU only, 4 threads.
Repo state before and after: only `?? runs/srope_torus_pilot/PILOT_SRoPE{,_NoSiLU}.md` plus `?? docs/paper/MAIN_DRAFT2.md`.
That third file appeared during the audit (mtime 2026-10-21 16:58). Another session wrote it, not this audit.

| script | verdict | worst finding |
|---|---|---|
| `docs/audits/2026-10-20/causal_whatwhere_T1024.py` | REPRODUCED (2 checkpoints, JSON rows bit-identical) | MINOR: the write-up's and MAIN_DRAFT2's wording ("position alone explains 92.5% of the variance", "failing ones collapse to 0.03-0.07", "\|.\| models depend") needs tightening. The Spearman has no committed script. The stats are uncorrected |
| `docs/audits/2026-10-16/sign_floors.py` | REPRODUCED (byte-identical output) | MINOR: stale docstring. The n-gram is fit on maps 10-25, not SIGN_MATCHED's training maps 0-7; refit on 0-7 gives 0.5747 vs 0.5749 |
| `docs/audits/2026-10-16/ring_floors.py` | REPRODUCED (identical) | MINOR: another stream than the arms (labelled as such). The overall retrace on the registered eval stream is 0.8636 vs 0.8596 |
| `docs/paper/figs/torus_strata_floors.py` | REPRODUCED (s0-7 and s500-503 JSON identical; asserts vs registered STRATA pass) | none |
| `docs/paper/figs/ring_strata_floors.py` | REPRODUCED (json and txt byte-identical) | none |

No BLOCKER. No MAJOR.

---

### 1. causal_whatwhere_T1024.py -- REPRODUCED

**What it computes.** It is a port of `docs/audits/2026-10-03/causal_whatwhere.py`, diffed line by line. The only changes are:
- the arms (SIGN_MATCHED Signed_r4 / Abs_r4 / Pos_r4 / RoPE, `runs/sign_matched/p0/{arm}_s{0-7}`) and T = 1024;
- the evaluation runs in chunks of 10, with integer hits pooled over targets, which is exact;
- the calibration set is 20 walks instead of 100. That is 20.5k tokens, more than T = 128's 12.8k;
- the phase-lesion and lam_add grids are dropped;
- the floors are not recomputed (`floors()` is still defined but is never called).

The intervention code is identical to the T = 128 version, and I read it against its claims:
- **pos_class**: per-type (action / observation) frequency-weighted mean of q and k from the calibration set. The class means are assigned by broadcast, which is correct.
- **pos_tmatch**: beta is the ratio of mean within-row sd over action rows, excluding t = 0.
- **additive**: P + C(v_t, v_s), with C the per-pair mean of S - P.
- **gain**: a per-pair OLS fit S ~ alpha + gamma P (gamma = cov/var), falling back to gamma 1 when a pair has n < 50.
- **content**: c = 1, s = 0.

The pair tables are gathered as `tab[:, inp[:, :, None], inp[:, None, :]].permute(1, 0, 2, 3)`, which gives (B, H, T, T) correctly. Phases are never touched. Every intervention goes through the same `score()` / `finish()` as the intact rebuild.

**Checkpoints.** 32 distinct checkpoints: md5 has no duplicates. Seeds 0-7 are the registered SIGN_MATCHED batch, with no pilot seeds. Eval stream: env seed 10000, np seed 1234 + s, 100 walks. Per-seed `n_targets` (31923, 31836, ...) equal `sign_floors_out.txt`'s n exactly, so the quoted floors (blank 0.506, n-gram 0.575, retrace 0.843) are on the same walks.

**Logit check.** The rebuilt forward matches `model(tokens)` with max |dlogit| 0.0 on all 32 (log) and on my two re-runs. Intact accuracy equals `acc_model` exactly. The stored registered eval differs by at most 0.0007 (Abs_r4 s1: 0.9158 vs 0.9165). I re-ran that checkpoint with an independent batch-1 CPU forward (`b1check.py`) and got 0.9158 (29156 / 31836). The gap is therefore GPU vs CPU numerics in the stored GPU eval (22 targets), not the rebuild.

**Re-run.** `TAG=a ARMS=Signed_r4 SEEDS=5` and `TAG=b ARMS=Abs_r4 SEEDS=1`, run on `cw.py`: a copy with OUT moved to scratch, ARMS and SEEDS taken from the environment, and 4 threads. All 25 fields of both JSON rows are bit-identical to the committed `causal_whatwhere_T1024.json`.

**Numbers checked against the JSON.**
- Claim 54: signed pos_class runs 0.9255-0.9994 (mean 0.966) and gain 0.9897-1.0000 (costs <= 0.0006). Correct.
- Plan line 303: |.| pos_class costs 0.020-0.613 (mean 0.277) and gain 0.483-0.897. Correct.
- Write-up: the Spearman 0.84 (p 5e-5) reproduces, at 0.838 and p 5.0e-5, from (intact, pos_class / intact) over the 16 Abs + Pos runs.
- Against the per-seed retrace floor: below on |.| 3/8, softplus 4/8, RoPE 8/8, signed 0/8. Correct.

**Dropout re-score.** None is applied. Every intervention is compared within eval mode, so this is consistent. DROPOUT_RESCORE moves these arms by at most +0.0011 (Abs / Pos) and +0.004 on average for RoPE (+0.028 on its largest seed). No run crosses retrace.

**Statistics.**
- The paired t-tests are uncorrected over 24 contrasts (4 arms x 6 conditions) and labelled post hoc.
- Signed identity-removed: -0.032, MDE 0.031, t p 0.0116, exact sign-flip p 0.0078 (8/8 lower, which is the design's minimum p). Holm(t, 24) gives 0.081.
- Softplus gain -0.106: t p 0.0495, Holm 0.15.

**Post hoc choices.**
- N_CAL = 20 and CHUNK = 10 were set for memory, not chosen from the data.
- The gain-fit cutoff (n >= 50) is inherited from T = 128.
- The Spearman (arms pooled, ratio measure) was chosen after the numbers were seen. Its script is not committed (rule 7).

**Findings.**
- **MINOR-1 (wording: "variance explained")**
  - Where: WHAT_WHERE_T1024.md, table row "share of score variance explained by position alone 0.925". The same sentence is in `docs/paper/MAIN_DRAFT2.md` sec. "What and where" ("position alone explains 92.5% of the score's variance").
  - Problem: the quantity is SS(P)/SS(S), a ratio of within-row sums of squares over action-query rows of the calibration set. It is not an R^2.
  - Check: I computed R^2 = 1 - SS(S - P)/SS(S) with the script's own functions (`r2check.py`) on all 8 signed runs: mean **0.923** (per head 0.761-0.968), against the ratio 0.925. The number survives (0.925 -> 0.923); only the definition is loose.
  - Fix: write "the content-averaged score accounts for 92% of the within-row score variance (action queries, calibration walks; R^2 = 1 - SS(S - P)/SS(S))", or keep 0.925 but call it SS(P)/SS(S).
- **MINOR-2 (wording: the collapse sentence)**
  - Where: WHAT_WHERE_T1024.md, "the failing ones collapse to 0.03-0.07".
  - Problem: the claim is true only for the 4 failing softplus runs (s0, s1, s2, s4), and only as absolute accuracy (0.033-0.065). As a share of their accuracy they keep 0.05-0.12. The failing |.| runs keep 0.22-0.78 of theirs (absolute 0.155-0.69).
  - Fix: "the four failing softplus runs fall to 0.03-0.07 (5-12% of their accuracy); |.| runs keep 22-98%".
- **MINOR-3 (claim 54 wording)**
  - Where: SYNTHESIS_DRAFT2_PLAN claim 54, "|.| models depend on content-position interactions".
  - Problem: this holds for the arm mean (-0.277 identity removed, gain -0.237; 8/8 lower on both) but not uniformly. Abs_r4 s0 (0.931) loses only 0.020 and 0.034. The two solving softplus runs separate (they keep 0.97 / 0.99).
  - Fix: "|.| models lose 0.28 on average (7/8 runs >= 0.15) ...; separation tracks success (Spearman 0.84, analysis)".
- **MINOR-4 (plan is out of date)**
  - Where: SYNTHESIS_DRAFT2_PLAN lines 299-306 and 523.
  - Problem: they still say "Pos n = 5, RoPE pending, .txt pending". The run is complete (`.txt` exists).
  - Final values for the plan: softplus intact 0.780, identity removed 0.476 (-0.305), gain 0.675 (-0.106). RoPE intact 0.731, identity removed 0.479 (-0.252), gain 0.439 (-0.293).
- **MINOR-5 (wording: "object identity removed")**
  - Where: WHAT_WHERE_T1024.md (inherited from T = 128).
  - Problem: pos_class also removes action identity within the action class and blank-vs-object within the observation class. "Object identity removed (type kept)" understates this.
  - Fix: "token identity within type removed". This changes no number.
- **MINOR-6 (wording: "shared kernel x content gain")**
  - Where: WHAT_WHERE_T1024.md, MAIN_DRAFT2.
  - Problem: the gain condition also has a per-pair additive offset alpha(v_t, v_s): 2 x 441 parameters per head, fit on the calibration walks. It therefore nests the additive condition.
  - Fix: "shared kernel times a per-pair content gain plus offset".
- **MINOR-7 (statistics)**
  - Where: WHAT_WHERE_T1024.md and MAIN_DRAFT2, "-0.032, p 0.012".
  - Problem: the p is an uncorrected paired t over 24 tests, Holm 0.08, and the effect sits at the MDE (0.031).
  - Fix: "8/8 seeds lower (sign-flip p 0.008, uncorrected; analysis)".
- **MINOR-8 (stale docstring, mismatched log)**
  - The docstring is stale. It describes the in-script floors, one-batch evaluation, `causal_whatwhere.json` outputs and the phase / lam_add grids, none of which this script runs. `floors()`, `SIGMAS` and the `koff` path are dead code.
  - The committed `.log` ends in `NameError: summarize`. The `.txt` was written afterwards via `--report`, by a script edited after the run; the JSON is unaffected.
  - Fix: trim the docstring, and note in WHAT_WHERE_T1024.md that `.txt` came from `--report`.
- **MINOR-9 (rule 7)**
  - Where: WHAT_WHERE_T1024.md and MAIN_DRAFT2 ("Spearman 0.84 over 16 runs").
  - Problem: no committed script computes the Spearman.
  - Fix: add it to `report()`.

### 2. sign_floors.py -- REPRODUCED

**What it computes.**
- Floors: always-blank, which equals the best constant, and the best backoff token n-gram of orders 0-5. The n-gram is fit on revisit targets of 50 walks of T = 1024 on each of maps 10-25.
- One-level retrace.
- The wrap-only share.

All are computed on the eval stream: env 10000, np 1234 + s, 100 walks, T = 1024.

**Checks.**
- Retrace is correct. Validated empirically on 30 walks: it fires only on revisits (6461 fires, 0 on non-revisits), always at the same cell as `env.visited_locations`, and its precision is 1.000. The reversal test `a ^ 1` matches DELTA (0 <-> 1, 2 <-> 3).
- The n-gram's index arithmetic is right: the target is at token 2t + 1, and order 4 is the one-step reversal copy, which is why it jumps to 0.58.
- The stream equals the arms' stream: n per seed matches the causal JSON and `SIGN_MATCHED_STRATA.json`.

**Re-run.** `FLOOR_OUT=<scratch> python3 sign_floors.py` produced output byte-identical to `sign_floors_out.txt`.

**Findings.**
- **MINOR-10 (n-gram fit maps)**
  - Where: APPENDIX_draft6 sec. B (l. 156-157), "n-gram still fit on the 16 training maps 10-25".
  - Problem: maps 10-25 are SCORE_RANK's training maps. SIGN_MATCHED trains on maps 0-7 (`GridWorld(seed=args.seed)`).
  - Check: refit on maps 0-7 (100 walks each, `sf_fit07.py`) gives n-gram 0.5747 vs 0.5749, and blank and retrace are unchanged. No number changes.
  - Fix: "fit on 16 other maps (seeds 10-25), none the eval map".
- **MINOR-11 (stale docstring)**
  - Problem: the docstring still describes "Per seed s in 10..25 and T in {512, 1024, 2048}" and gives the output file as `score_rank_floors_out.txt`.
  - Fix: update the docstring.

### 3. ring_floors.py -- REPRODUCED

**What it computes.** Blank and one-level retrace on GridWorldCancel(32, p_plus), with the map redrawn per walk: 2000 walks of T = 128 at np seed 4242, scored on revisit targets as in `train_cancel.evaluate`. The environment defaults (16 obs, p_empty 0.5) match `train_cancel`. Retrace is "any change of move reverses", which is correct for +1 / -1.

**Re-run.** The output is identical to `ring_floors_out.txt` (0.8596 / 0.7582 / 0.6118).

**Findings.**
- **MINOR-12 (stream)**
  - Problem: the stream is not any arm's eval stream. The drafts say "another stream of the same task", which is correctly labelled.
  - Check: on the registered ring eval stream (np 10**6, 400 walks, p_plus 0.5; `ring_strata_floors`) retrace is 0.8636 vs 0.8596. No arm is within 0.02 of either value: ring |.| 0.95-0.97; index 0.710 / 0.677 below 0.860 / 0.758 and 0.824 above 0.612.
  - No fix needed. Optionally quote "0.860 (0.864 on the registered eval stream)".
- No n-gram is computed here. The registered ring n-gram (0.554) is quoted beside it, so rule 1 is covered.

### 4. torus_strata_floors.py -- REPRODUCED

**What it computes.** Per seed and stratum: the best constant, blank and retrace. The strata come from the registered `eval_rank_strata.kinds_vec` (imported, not re-implemented) with the same assignment: wrap = the unwrapped position is new; otherwise plain, split by the lag since the last visit to the wrapped cell at < 128 / >= 128. The start cell is unobserved in both `kinds_vec` and `environment.revisit_mask`, so they are consistent.

**Seed sets.**
- s0-7 is the eval stream of SIGN_MATCHED / RANK_MI / RANK3. The checker also asserts n and the constant against those STRATA JSONs.
- s500-503 is FACTORIAL's registered n = 4 seeds, asserted against FACTORIAL_{L1, L4, LOOP}_STRATA.
- The two sets are never pooled, and no pilot seed is included.

**Re-run.** Both `--check` sets were run from a scratch copy. Every assert passes, and `per_seed` and `checked_against` are identical to the committed JSONs.

**Quoted numbers.** These are correct:
- s0-7: constant 0.506 / 0.500 / 0.507, retrace 0.911 on short gaps.
- s500-503: constant 0.508 / 0.515 / 0.517, retrace 0.912; overall 0.509 / 0.844.

Retrace equals the constant in the long-gap and wrap strata by construction, because runs are <= 10 moves.

**Note.** No per-stratum n-gram is computed. In the long-gap and wrap-only strata the n-gram cannot beat the constant: its only gain is the order-4 reversal copy at gap 2. In the short-gap stratum retrace dominates it.

### 5. ring_strata_floors.py -- REPRODUCED

**What it computes.** It reuses `ring_strata.stream()`: GridWorldCancel(32, seed 10000, p_plus 0.5), np 10**6, 400 walks, the registered ring eval stream. Its strata function is unit-tested against environment_cancel's revisit mask. It applies the same one-level retrace as `ring_floors`.

**Re-run.** Copied to scratch with `figs/` on PYTHONPATH and no bytecode written. The output txt and json are byte-identical.

**Note.** The bin label "32-128" (GAP_BINS hi = 129) is quoted as "32-127" in FIGS_NOTES and MAIN_APPENDIX1. The two are equivalent, because the maximum gap at T = 128 with the start cell unobserved is 127.

---
