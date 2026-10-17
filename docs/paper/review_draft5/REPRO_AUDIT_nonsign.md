# Reproduction audit of the non-sign batches behind OUTLINE_overall.md

2026-10-16, HEAD 00372e1, torch 2.10.0+cu128, CPU only (`CUDA_VISIBLE_DEVICES=""`; GPU 0 was held by another
user's job, GPU 1 left free for the lr-sweep). Same method as `REPRO_AUDIT.md` (the sign line). Nothing in `runs/` or
any committed analysis output was written; re-run outputs live in the session scratchpad (`.../scratchpad/repro2/`,
ephemeral; the commands at the end regenerate them).

Terms. **Byte-identical**: the re-run's stdout (or written file) equals the committed file byte for byte.
**Generated body identical**: the script writes a results `.md` that was later extended by hand (CORRECTED block,
"Reading" prose); every line the script writes is identical, the hand-written lines are absent from the re-run.
**IC**: an independent results-blind code check amended into the pre-registration before any result was read
(rule 29). **Pilot in batch**: a pilot run's checkpoint is bit-identical to (or is) a batch run.
**Spot-check**: `stats_core.classify_run` on the stored losses vs the class the analysis prints, plus an independent
CPU re-evaluation with the trainer's or the batch's evaluation protocol vs the stored accuracy.

## Verdict per batch

| batch | analysis script | committed output vs re-run | run dirs (complete?) | spot-check (2 runs) | prereg | IC | pilot in batch | verdict |
|---|---|---|---|---|---|---|---|---|
| PAPER2X2 | analyze_paper2x2.py (reads `_PAPER2X2_RAW.json`) | RESULTS.md generated body (95 lines) identical; RAW_CONTRASTS.json byte-identical | 6 arms x 8, s0-7, 300 ep, 48/48 | Vanilla s3 SOLVED, 0.999640 exact; RoPE s5 STALLED, 0.795047 exact (T=128) | yes (3 h before first run) | no | no | **REPRODUCED** |
| FACTORIAL | analyze_factorial.py | ANALYSIS.txt and VERDICTS.json byte-identical | 4 cells x 4, s500-503, 900 ep, 16/16 | L4 RoPE s501 STALLED, 0.869687 exact; L1 Vanilla_r4 s502 SOLVED, 1.000 exact | yes | **yes** (Amendment 1) | no | **REPRODUCED** |
| DYCK_MDEPTH | analyze_dyck_mdepth.py (device lines edited to CPU; it hard-codes `.cuda()`) | RESULTS.md generated body identical (all tables and verdicts); RESULTS.json max abs diff 6.4e-6 (CPU vs GPU); in-batch repro weights_differing 0 | T12 16 x 8, T12x3 6 x 8, Tmix 4 x 8 = 208/208 + 2 repro; ladder 128 re-scored | the analysis re-evaluates all 336 checkpoints on CPU (full re-eval, not a sample) | yes (1 min before first run) | no | no | **REPRODUCED** |
| DYCK_LADDER (not cited in the outline) | analyze_dyck_ladder.py (CPU edit as above) | generated body identical (68-line CORRECTED/Reading prefix is hand-written); JSON max abs diff 2.6e-4 | 16 x 8, s0-7, 128/128 | full CPU re-eval inside the analysis | yes | no | no | **REPRODUCED** |
| CANCEL | analyze_cancel.py | ANALYSIS.txt byte-identical | 16 cells x 8, s0-7, 300 ep, 128/128, eval.json 128 | Vanilla L1 p0.9 s2 SOLVED, 1.000 exact; RoPE L3 p0.75 s5 SOLVED (tail 0.024), 0.998230 exact | yes | no | **yes**: `runs/cancel_pilot` L1 p0.5/p1.0 s0-1 (8 ckpts) bit-identical to batch | **REPRODUCED** (pilot pooled; eval caveat below) |
| TEXTWORLD | analyze_textworld.py; analyze_textworld_secondary.py | ANALYSIS.txt byte-identical; SECONDARY.txt byte-identical; SECONDARY.json one value differs 1.6e-4 (CPU re-eval) | 3 arms x 8, s0-7, 900 ep, 24/24 | Vanilla_r4 L1 s3 SOLVED, 0.994515 exact; RoPE L2 s6 STALLED, 0.759599 exact | yes | **yes** (Amendment 1, committed while the batch trained, before results were read) | **yes**: `runs/textworld_pilot` s0-1 (4 ckpts) bit-identical to batch | **REPRODUCED**; fresh s2-7 recomputed: +0.4742, perm p 0.0022, 6/6 vs 0/6 (matches) |
| TW_NORMSTEP (not cited in the outline) | analyze_tw_normstep.py --readouts (re-evaluates every model) | ANALYSIS.txt and TW_NORMSTEP.json byte-identical | 4 arms x 8, s10-17, 32/32 | NormStep s12 SOLVED, DirOnly s15 SOLVED (accuracies re-evaluated by the analysis) | yes | **yes** | no (pilot s100) | **REPRODUCED** |
| RANK_MI (arms A, B, C of the five-arm table) | analyze_rank_mi.py | ANALYSIS.txt byte-identical; repro s3 loss diff 0 | A = 8 symlinks to `rank_matched_e900` (s0-7); B s0-1 symlinks to `rank_perhead_pilot`, s2-7 new; C 8 new; 900 ep | B s3 STALLED, 0.722630 vs 0.722661; A s6 STALLED, 0.773291 vs 0.773260 | yes | no | **yes, disclosed**: B s0-1 are the per-head pilot (outcomes known in advance); A's s0-1 are RANK_MATCHED's own pilot | **REPRODUCED** |
| RANK_SEP | analyze_rank_sep.py | ANALYSIS.txt byte-identical; repro C s0 loss diff 0 | 2 arms x 8, s0-7, 900 ep + A/B/C from rank_mi | C_bd s1 DESCENDING (0.302), 0.936299 exact; D s4 SOLVED, 0.997605 exact | yes | no | via B (above) | **REPRODUCED** |
| RANK_PROJ | analyze_rank_proj.py (changed after its results commit: scipy fisher_exact -> stats_core.fisher_solved; output unchanged) | ANALYSIS.txt byte-identical | projected inits 8 (`runs/rank_proj`), trained 8 (`rank_proj_train`), control 16 (`rank_matched_e900c`) | trained s0 SOLVED, 1.000 exact; s5 SOLVED, 0.998503 exact | yes (inits built 2 min before the commit; training 6 h after) | no | no | **REPRODUCED** |
| RANK3 | analyze_rank3.py | ANALYSIS.txt byte-identical; repro (rank3_repro = rank_sep r4ph s0) bit-identical | 1 arm x 8, s0-7, 900 ep | s2, s6 SOLVED, 0.999937 / 0.999969 exact | yes | no | no | **REPRODUCED** |
| RANK_TESTS | analyze_rank_tests.py | ANALYSIS.txt and VERDICTS.json byte-identical | ring 5 x 8 (s20-27, 300 ep); escape C/E/F x 8 (s0-7, 600 ep); lifted/fresh = 0-epoch builds | ring R1 s21, R2 s26 SOLVED, 1.000 exact; escape E s3 SOLVED, 0.999597 exact | yes | **yes** (Amendment 1, before the pilot was read) | no (pilots s100/120; lifted s0 build is deterministic) | **REPRODUCED** |
| LOOP_RANK | analyze_loop_rank.py | ANALYSIS.txt byte-identical; repro A s0 loss diff 0 | 2 arms x 8, s0-7, 900 ep + A, C reused | Looped s1 DESCENDING (tail 0.0528), 0.985111 vs 0.985143; Vanilla_L4 s5 DESCENDING (0.0538), 0.994563 exact | yes | no | via A | **REPRODUCED** |
| LOOP_RANK_E1800_P1 | analyze_loop_rank_e1800_budget.py | ANALYSIS.txt byte-identical | 2 arms x 8, s0-7, 1800 ep, 16/16 (L/L4 launches of part 2 were stopped; none kept) | A s2 DESCENDING (0.391), 0.893927 vs 0.894021; C s5 STALLED (0.180), 0.962703 exact | yes (+ Amendment 1 splitting the batch, mid-run, before results were read) | no | no | **REPRODUCED** (part-2 script `analyze_loop_rank_e1800.py` cannot run: `LOOP_RANK_E1800.json` never produced, as designed) |
| RANK_ND | analyze_rank_nd.py; analyze_rank_nd_secondary.py | ANALYSIS.txt byte-identical; SECONDARY.txt one rounding (0.767 -> 0.768); SECONDARY.json max 8.1e-4 (CPU re-eval) | 4 cells x 8, s0-7, 900 ep, 32/32 | D3 r3ph s0 STALLED, 0.704152 vs 0.704132; D2 r2ph s5 DESCENDING, 0.722581 vs 0.722283 | yes | **yes** | no | **REPRODUCED** |
| RANK_WRAP | analyze_rank_wrap.py (re-scores strata on CPU) | ANALYSIS.txt byte-identical; RANK_WRAP.json max 1.7e-4 | 4 cells x 8, s0-7, 32/32; cells 2L and 3H bit-identical to RANK_ND B2/B3 (registered determinism check) | 3L s2 SOLVED, 1.000 exact; 2H s6 STALLED (0.064), 0.238088 exact | yes | **yes** | no | **REPRODUCED** (only 2H and 3L are new evidence) |
| LEAK | analyze_leak.py (re-evaluates all 24 models; device arg set to CPU); analyze_leak_secondary.py | ANALYSIS.txt byte-identical (full CPU re-eval of 24 models, 31 min); LEAK.json 73 values differ, max 8.5e-5; LEAK_SECONDARY.txt byte-identical | 3 arms x 8, s0-7, 900 ep, 24/24 | MapWM s4 DESCENDING (0.070); NormStep s1 SOLVED (0.020); accuracies re-evaluated by the analysis | yes | **yes** (Amendment 1, mid-run, before results were read) | no (pilot s100) | **REPRODUCED** |
| GENERALITY CARoPE arms | analyze_generality.py | byte-identical (already in `REPRO_AUDIT.md`; ring arms CPU re-scored inside the analysis) | torus CARoPEF 8, ring CARoPEF 8, s700-707 | torus CARoPEF s703 STALLED, 0.647791 exact (CPU); ring s701 STALLED | yes | **yes** | no | **REPRODUCED** |
| LOOP_HEADROOM | agg_loop_headroom.py | generated body (51 lines) identical; the "Final, all five arms at n=8" section is hand-written (its per-seed numbers match the JSONs) | 5 arms x 8, s0-7, 300 ep, 40/40 (s0-2 first, s3-7 a top-up) | PI_loop s2 0.782895 vs 0.771930 (+0.011); PI_flat s6 0.118834 vs 0.109865 (+0.009) | no | no | no | **REPRODUCED** (eval note below) |
| REFINE | agg_refine.py | generated body identical except one line: committed "(gate not recoverable from the checkpoints)", re-run recovers the per-seed gates, equal to the hand-added gate table; the "Corrected analysis" section is hand-written | 2 arms x 8, s0-7, 16/16 | Looped s2 0.127193 vs 0.122807 (+0.004) | no | no | no | **REPRODUCED**; pooled loop vs no loop recomputed from the 24 JSONs: +0.346, se 0.092, t 3.75 (Welch p 0.0025, Mann-Whitney p 0.002); drift 0.185; loop vs 3 layers +0.032 (t 0.30) |
| MQ_RANK_2X2 | agg_mq_rank.py | byte-identical | 4 arms x 8, s0-7, 32/32 | -- (same trainer and eval as above) | no | no | no | **REPRODUCED** |
| MATCH_QUERY_SCALE / _RESULTS (necessity n=5, destruction n=3) | inline heredocs in `run_match_scale.sh` / `run_match_query.sh`; destruction ablation: no committed script | -- | **`runs/match_query` does not exist** (never tracked; no `*_matchquery.pt` of these runs anywhere under /home/prashr) | -- | no | no | -- | **COULD NOT RUN** (arithmetic of the committed per-seed table checked: 0.7304 +/- 0.2473 vs PlainFlat 0.1542 +/- 0.0178, matches) |

No batch DIFFERS. Every byte-identical re-run exited 0 with empty stderr; total CPU wall per analysis 2 s (rank,
cancel, textworld, factorial) to 31 min (LEAK, 24 models re-evaluated). Analysis script at its results commit equals
HEAD for every batch except RANK_PROJ (hygiene change, output-identical); each script's last change is an ancestor of
its results commit.

## What was done

1. **Re-run.** Pure-stdout analyses run as `python3 -m mapformer.X` from `/home/prashr`; scripts that write into the
   repo copied to the scratchpad with only the write path changed (`sed` on the `"w")` lines), or given their own
   `--out` / `--out-stem`; the Dyck scripts and `analyze_leak.py` additionally had their hard-coded CUDA device
   replaced by CPU. Outputs diffed against the committed files; JSON outputs compared numerically.
2. **Spot-check, 2 runs per batch** (36 runs; 31 also re-evaluated). Classes from `classify_run` equal the per-seed classes each ANALYSIS
   prints in every case. CPU re-evaluation (torus: `eval_noise_refine`, 100 trials, env seed 10000, seed 1234 + s;
   3D torus: `eval_nd`; ring/text: the trainers' `evaluate`; Match-Query: `train_match_query.evaluate`, 8 x 8 episodes,
   seed 5000 + s): 22 runs exact, 6 within 3.0e-4 (CPU vs GPU argmax ties on unsolved runs), 3 Match-Query runs within
   0.011 (see note 7). Dyck, LEAK, TW_NORMSTEP, RANK_WRAP strata and the ring arms re-evaluate every model inside the
   analysis, so those are full CPU re-evaluations.
3. **Integrity scan** of 1001 checkpoints (991 unique files; symlinks counted once) under 29 batch run roots, plus 61 under 7 pilot / control roots: loaded,
   seed (dir vs checkpoint), `len(losses) == epochs`, `output_dir` == own dir, one recipe per cell (args identical up
   to seed), SHA-1 of every state dict and loss curve compared across all. No load error, seed mismatch, epoch
   mismatch, foreign `output_dir` or mixed-recipe cell (Dyck per-run JSONs differ only in output fields such as
   wall time). Driver logs: no traceback / FAILED / ABORT line; launch counts equal run counts plus the declared
   reproduction runs (rank_sep 17, rank3 9, loop_rank 17, rank_mi 15); every `.X_done` marker present. All analysis
   inputs (eval JSONs, per-run JSONs, eval.json files) are tracked in git and unmodified.
4. **Duplicates.** Every bit-identical pair is a declared determinism check or a declared reuse: dyck_mdepth/repro =
   dyck_ladder s0 (2); rank_sep_repro = rank_mi C s0; rank_mi_repro = rank_matched_e900 s3; loop_rank_repro =
   rank_matched_e900 s0 = rank_perhead_pilot Vanilla s0; rank3_repro = rank_sep r4ph s0; rank_wrap 2L/3H = rank_nd
   B2/B3 (16); rank_mi A and B s0-1 are symlinks. The pilots that ARE batch runs: cancel (8) and textworld (4). The
   `rank_proj` inits carry their source r=4 runs' loss curves (built by projection). None of these counts as
   replication (rule 6).
5. **Code versions.** Every md5 in the run roots' `code_md5.txt` resolves to a committed version. Changed since their
   batch: `train_variant.py` (VARIANT_MAP additions), `train_cancel.py` / `train_textworld.py` (docstring only),
   `model_rank_perhead.py` (+6 lines, new class), `rescore_hook.py`; `rank_matched_e900` predates the 2026-09-24
   bit-identical refactors of `model.py` / `train.py` / `environment.py` (training identity verified then, rule 19;
   here only the evaluation side is re-confirmed, A s6 within 3.2e-5). No `code_md5.txt` for PAPER2X2, DYCK_LADDER, LOOP_HEADROOM, REFINE, MQ_RANK,
   RANK_PROJ_TRAIN (the guard postdates them).
6. `python3 -m mapformer.test_guards` (CPU): 13/13 passed.

## Integrity notes

1. **Pilot seeds inside the registered batch, confirmed bitwise.** CANCEL: `runs/cancel_pilot` L1 p0.5 / p1.0 s0-1
   = batch s0-1. TEXTWORLD: `runs/textworld_pilot` s0-1 = batch s0-1. Both disclosed (CORRECTED blocks); the
   TEXTWORLD fresh-seed contrast reproduces. Rank: the five-arm table's B (per-head r=2, "2/8") contains the per-head
   pilot's s0 (STALLED) and s1 (SOLVED), known before registration -- on fresh seeds s2-7 B is **1/6**; A ("0/8")
   contains RANK_MATCHED's own 900-epoch pilot s0-1. The RANK_SEP verdict (D - C_bd) and the rank-3 contrast do not
   use B; the rank-2 "0-2/8" range in CLAUDE.md does.
2. **Arm A of every rank contrast is one stored batch.** `rank_matched_e900` (2026-09-23) is reused as the r=2
   reference by RANK_MI, RANK_SEP, LOOP_RANK, RANK3 and E1800's 900-epoch column. Under its own registration it
   landed UNREADABLE (four r=2 runs descending; Amendment 3 of `RANK_MATCHED_PREREG.md`); RANK_MI re-registered the
   reuse with the Amendment-2 classifier and a bitwise reproduction run. Legitimate and disclosed, but the r=2 0/8 is
   one set of 8 runs, not five.
3. **RANK_WRAP is half determinism.** Cells 2L and 3H repeat RANK_ND bit for bit (registered as a determinism void
   check). Only 2H (wrap-heavy 2D, 1/8) and 3L (grid 18, 8/8 -- the "rank D+1 suffices in 3D" number) are new.
4. **CANCEL evaluation stream** (disclosed in `train_cancel.py`, 2026-09-30): the ring map is redrawn per trajectory,
   so env seed 10000 is not a held-out map, and the eval stream (np seed 0) equals training batch 0 of seed-0 runs
   (1 of 29,400 batches). `train_textworld.py` has the same stream overlap but a different map. Reproduces as stored.
5. **Knife-edge SOLVED in the loop/depth aids.** Both LOOP_RANK spot runs sit just above the 0.05 cutoff (tails
   0.0528, 0.0538); the batch's solved counts (2/8, 5/8) move with the cutoff, as `LOOP_RANK_RESULTS.md` already
   says (verdict UNMEASURED).
6. **Dyck scripts hard-code CUDA** (`.cuda()`, `map_location="cuda"`): a CPU reproduction needs a device-only edit.
   On CPU every table and verdict line is identical; numbers move <= 6.4e-6.
7. **Match-Query loop evaluations used TF32.** `--fast-attn` in `train_match_query.py` also enables TF32 matmuls; the
   stored per-run accuracies differ from an fp32 CPU re-evaluation by up to 0.011 (5 of 456 answers on PI_loop s2).
   Far below the contrasts (+0.346) and below the batch-to-batch drift (0.185). The pooled +0.346 is asymmetric by
   construction: the loop arm pools two batches (16 runs), the no-loop arm is batch 1 only (8 runs), so batch drift
   enters one side.
8. **Match-Query necessity and context destruction cannot be reproduced.** `runs/match_query` is gone (never
   tracked); the destruction ablation's script was never committed (commit 33626d3 adds only the md). The committed
   per-seed TQ=256 table is arithmetically consistent, but `MATCH_QUERY_LONGQ.json` re-evaluated seeds 0-2 at
   0.803 / 1.000 / 0.910 against the table's 0.731 / 1.000 / 0.934 for the same seeds -- two evaluations of the same
   checkpoints differ by up to 0.072 per seed (evaluation sampling at 64 episodes).
9. **PAPER2X2 has no code record.** No `code_md5.txt`, and its checkpoints store no training args. The two exact
   re-evaluations show the current eval code reproduces the stored `_PAPER2X2_RAW.json`; the training code version
   is inferred from the commit date only.
10. Registration timing: TEXTWORLD, LEAK and E1800 amendments were committed while the batch was training (before any
    result was read, as each states); DYCK_MDEPTH and DYCK_LADDER preregs were committed one minute before the first
    checkpoint. Nothing found contradicts the "before results" statements.

## What this means for the outline's strength ladder (Sec. 5)

- **Tier 3 can carry "repro-audited" for every member**: PAPER2X2, DYCK_MDEPTH, RANK_SEP, RANK_PROJ, RANK3, E1800 and
  LOOP_RANK reproduce byte-identically (Dyck to 6e-6 on a full CPU re-evaluation) and spot-check exactly. What they
  still lack is IC; it cannot be retrofitted. A post hoc blind code read (as G5 proposes for CANCEL) is the most that
  can be added, labelled post hoc.
- **Tier 2 also reproduces** (TEXTWORLD, RANK_ND, RANK_WRAP, RANK_TESTS, LEAK) and **Tier 4** (FACTORIAL, CANCEL).
  G5's reproduction half is done for all of G5's list; only the blind code check of `train_cancel.py` /
  `analyze_cancel.py` remains.
- **The rank table needs two flags in the text**: "per-head r=2 2/8" is 1/6 on fresh seeds (two pilot seeds in the
  arm), and "our shared r=2 0/8" is a single stored batch (rank_matched_e900, itself containing its pilot seeds)
  reused as the reference for five contrasts. Neither changes a registered verdict (per-head rank fires on D - C_bd,
  which uses neither arm).
- **RANK_WRAP's 3D claim rests on one new cell** (3L, 8/8); 2L / 3H are determinism repeats of RANK_ND, not support.
- **TEXTWORLD and CANCEL pilot pooling is confirmed bitwise**; lead with the fresh-seed numbers (TEXTWORLD n = 6
  reproduces: +0.474, 6/6 vs 0/6, p 0.0022).
- **Tier 5 Match-Query is now weaker than "NR"**: its checkpoints and the destruction script are missing, so
  "necessary for in-context maps" and "0.918 -> 0.074" are unverifiable from the record. This strengthens G7: register
  and re-run, or demote to an appendix sentence citing the committed table only. The loop line (+0.346) reproduces
  from committed JSONs, but is cross-batch and asymmetric (note 7).
- **Nothing found moves a registered verdict.**

## Reproduce

```
cd /home/prashr; export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr
# pure-stdout analyses
for m in cancel textworld rank_sep rank3 loop_rank loop_rank_e1800_budget rank_nd rank_proj rank_mi; do
  python3 -m mapformer.analyze_$m | diff mapformer/$(echo $m | tr a-z A-Z | sed 's/LOOP_RANK_E1800_BUDGET/LOOP_RANK_E1800_P1/')_ANALYSIS.txt -; done
# writers: copy, redirect the '"w")' paths (and .cuda() -> CPU for dyck / leak), then diff
python3 -m mapformer.analyze_tw_normstep --readouts --out /tmp/x/TW_NORMSTEP.json | diff mapformer/TW_NORMSTEP_ANALYSIS.txt -
python3 -m mapformer.agg_mq_rank --runs-dir mapformer/runs/mq_rank --tq 256 --out /tmp/x/MQ.md && diff mapformer/MQ_RANK_2X2.md /tmp/x/MQ.md
python3 -m mapformer.eval_noise_refine --runs-dir mapformer/runs/rank_sep --variants Vanilla_r4ph --noises 0.0 --seeds 4 \
  --lengths 1024 --n-trials 100 --device cpu --title x --out /tmp/x/r.md
python3 -m mapformer.test_guards
```
