# FIXES: disposition of every VERIFY.md finding

Applied 2026-09-13 to `report.tex` and `refs.bib`. Each finding was checked against the primary file(s) named
in the last column before editing. `L1`-`L11` are the lead's confirmed findings; each maps onto a VERIFY.md id
and is recorded on that row. Sec. 1b bullets are numbered N1-N33 in order of appearance; Sec. 3 objections are
R<claim>.<n>; Sec. 5 items are B1-B6. `X1` is an extra error found while checking O17.

Dispositions: FIXED / REJECTED / PARTIAL / NEEDS-NUMBER.

## Section 1: errors

| id | disposition | change / reason | source checked |
|---|---|---|---|
| E1 (L4) | FIXED | "-0.388 of the -0.438 swing" replaced: rotation alone takes the effect from +0.438 to +0.050 (reduction 0.388); all five knobs combined take it to -0.076 (change -0.514). | KNOB_SWEEP_n8.md; BASELINE_TABLE.md sec. I |
| E2 (L3) | FIXED | Retracted "every looped seed >= 0.77" removed; replaced by pooled 0.803 +/- 0.200, 1/16 failures, and an explicit retraction sentence. | REFINE_RESULTS.md "Corrected analysis" |
| E3 (L3) | FIXED | Match-Query loop batch said to be pooled from three launches (seeds 0-2; PI one-layer arms 3-7; other arms 3-7); contrasts read unpaired: loop +0.346 (se 0.092, t 3.75); loop vs 3 layers +0.032 (t 0.30). Paired +0.414/+0.099/+0.315/+0.348 and MQ_RANK +0.154/+0.149 kept only as paired estimates with the pairing caveat (Claim 1 opening, Sec. 6.2, Sec. 4 reproducibility paragraph, Claim 5 opening, Tab. loop caption and rows, rank scope bullet, contributions row, Limitations). | runs/loop_headroom/*.log mtimes; run_loop_headroom.sh, run_loop_topup.sh, run_loop_topup2.sh; REFINE_RESULTS.md; LOOP_HEADROOM.md; MQ_RANK_2X2.md, runs/mq_rank logs (one launch) |
| E4 (L5) | FIXED | "firing the pre-registered outcome B" replaced: 400-ep levels meet B's numeric condition but the scripted verdict was withheld for non-convergence; falsification rests on converged cells (+0.178 n=5; +0.305 at 800 ep, n=3, exploratory), no test of their difference, and the post-hoc pooled reading. Claim 2 opening and contributions row rescoped accordingly. | ALIASING_CONTROLLED.md verdict; MINIWORLD_ENDPOINTS.md; VISITS_TEST.md; inventory D20 |
| E5 (L1) | FIXED | "monotone beats index nowhere" scoped to matched loss in abstract, Claim 3 opening, Result paragraph, contributions row; raw Abs-RoPE +0.146/+0.226/+0.213 (MDE 0.055/0.059/0.097) added to text and Tab. sign-contrasts. | SIGN_ABLATION.md contrast tables and sec. 4 |
| E6 | FIXED | PoPE-on-path-integration: "the last two detectable, MDE 0.032 and 0.065" (raw, grid 64). | POPE_WRAPPING.md T=512 and T=1024 tables |
| E7 | FIXED | Selective RoPE full generator: not better on parity (-0.009, MDE 0.017, unmeasured); marginally better on torus at T=512 (+0.031, MDE 0.030); unmeasured at T=1024. | SELECTIVE_ROPE.md |
| E8 | FIXED | Registered interaction given as +0.058 (MDE 0.182, expectation zero by the gauge); +0.113 (MDE 0.195) labelled post hoc. | N5_RESULTS.md P2; AUDIT_2026-09-10.md #5 |
| E9 | FIXED | "magnitude-locked control" -> "phase-locked control with magnitude free (EMDoF_alignlock)". | D5_RESULTS.md; train_variant.py |
| E10 | FIXED | One exposure-matched pair ceiling/uninformative; the other unmeasured (-0.068, MDE 0.132, 1/8). | SPREAD_RESULTS.md contrast table |
| E11 | FIXED | "the weakest arm" -> "a weak arm (... base 0.827)". | BASELINE_TABLE.md sec. H |
| E12 | FIXED | Both never-moved values marked "pre-fix scorer, not re-measured" (no corrected value exists; nothing re-run). | MATCH_GATES_128_16.md; MATCH_GATES_64_4.md; MATCH_QUERY_GATES.md correction |
| E13 | FIXED | "path integration over index" -> "single-origin (commutative) MapEM minus Plain-Flat grows from +0.115 to +0.180 (no MDE recorded)". | FAMILY_TREE_RESULTS.md; FAMILY_TREE_D7_RESULTS.md; run_family_tree_d7.sh |
| E14 (L10) | FIXED | TEM-t noise values now state evaluation-time noise and give Vanilla 0.757/0.638 under that protocol; tab:april noise columns flagged as not comparable; faithful TEM stated as scored on one fixed map (env seed 0). Intro "every evaluation redraws the map" given an appendix exception. | TEM_T_MULTISEED.md; run_tem_t_multiseed.sh (eval_noise=0.10); eval_single_env.py; TEM_BACKGROUND_BASELINES.md |
| E15 | FIXED | Sec. 4 uses fresh-map 0.901 +/- 0.102; App. B gives 0.898 (training map) and 0.901 (fresh map), identical maps only for MapWM and single-origin EM. | PAPER_TASK_ACCURACY.md |
| E16 | FIXED | "all token magnitudes equalised" -> "filler increments set to the mean content increment". | ablate_recency_gate.py docstring; RECENCY_RESULTS.md |
| E17 | FIXED | "32 lowest-frequency of its 64 channels (2 heads x 32)" at both places; "64 frequency channels per head". | LOCALISATION.md; probe_localisation.py; MAPPOPE_R4_RESULTS.md |
| E18 | FIXED | CARoPE - Signed at T=1024 marked detectable; T=128 cells filled (see N17). | SIGN_ABLATION.md T=1024 table |
| E19 | FIXED | Vocab sweep: full contrast +0.0597 (MDE 0.169), all from one collapsed seed; worst-dropped +0.0000 given without an MDE. | VOCAB_EM.md |
| E20 | FIXED | n-gram gates compared to clean 128^2 values; never-moved 0.1042 stated as above chance with no corrected clean 128^2 value (the audit's clean 0.089 is the 64^2 grid, so its suggested comparison was not used). | MATCH_QUERY_GATES_P010.md; MATCH_GATES_128_16.md; MATCH_QUERY_GATES.md |
| E21 | FIXED | Tab. knob caption: all arms 16 epochs; rotate/allocentric 392 batches (matched supervised events), baseline/combined 98. | run_seeds_n8.sh phase 2 |
| E22 | FIXED | BLiMP "(0.79 against 0.78)" (MapWM first). | papers/txt/mapformer.txt Fig. 7 and l.1994; LANGUAGE_LANDSCAPE.md |
| E23 | FIXED | Flip-Flop: "all arms but CARoPE_r4 (0.03%) are at 0.00%". | FLIPFLOP_RESULTS.md table |
| E24 | FIXED | Commanded best 0.955 at T=1024. | MINIGRID_FULL_2X2X2.md |
| E25 | PARTIAL | Detectable cell named (r=D+2 minus r=D at D=5, +0.073, t 3.35, 8/8) with mapformer_math.tex and paper App. D.1 in src. No MDE exists for +0.073 in any file, so none given (NEEDS-NUMBER for the MDE). | DXR_RANK_THRESHOLD.md; mapformer_math.tex table and l.2366; papers/txt/mapformer.txt App. D.1 |
| E26 | PARTIAL | Wording fix: oracle 0.854 and best single count 0.847 (k=3) stated as unweighted means over five fifths, while the per-count list pools tokens. The audit's alternative list of strata means is in no file and was not used (would be a new statistic). | LOOP_DEPTH_STRATA.md; loop_depth_strata.py |

## Section 1b: numeric imprecisions

| id | disposition | change / reason | source checked |
|---|---|---|---|
| N1 (l.51) | FIXED | Abstract now states T=1024 (eight times training length), raw 0.363 and matched-loss 0.280. | SIGN_ABLATION.md |
| N2 (l.58) | FIXED | Abstract names single shared origin and gives 0.137 for separate origins. | RECENCY_EM_RESULTS.md |
| N3 (l.390) | FIXED | "at least four times in this line"; +0.215 marked pooled at n=48. | PAIRSPLIT_RESULTS.md ("Fourth instance") |
| N4 (l.418) | FIXED | Match-Query change stated with epochs doubled to 600; compositional change "the recipe change above", sd 0.070 to 0.126. | run_mq_noise_2x2.sh, run_mq_noise_c2.sh; COMP_HEADROOM.md |
| N5 (l.456) | FIXED | "Our EM is MapEM-os with a single shared origin p0 (VanillaEM_P0), the ablation of the paper's separate q0,k0"; table labels carry (p0). | PAPER_OOD_EXTENDED_n8.md; INDEX_BASELINE_PAPER_TASK_n8.md; paper App. A.7 |
| N6 (l.476) | FIXED | Caption notes Appendix B l=512 vs v4 caption l=256 and gives our l=256 values (0.978/0.984; 0.988/0.988; floor 0.803). | PAPER_OOD_RERUN.md; PAPER_OOD_EXTENDED_n8.md; PAPER_TASK_FLOORS.md |
| N7 (l.152) | NEEDS-NUMBER | "of 40 sources" removed; 42 is only a directory count stated in no results file. | ls papers/txt; papers/INDEX.md |
| N8 (l.571) | PARTIAL | No .md file states sd/MDE for +0.461; wording changed to "No results file reports an sd or MDE". Auditor's recomputed sd/MDE not used (new statistic). | INDEX_BASELINE_PAPER_TASK_n8.md; BASELINE_TABLE.md |
| N9 (l.612) | FIXED | "about 50 times the per-arm sd" replaced by the action-stream drops 0.604-0.811 against the largest per-arm sd 0.0449 (auditor's 14-68x not used). | PAPER_TASK_ABLATION.md |
| N10 (l.639) | FIXED | Context destruction labelled n=3. | MATCH_QUERY_RESULTS.md |
| N11 (l.655) | FIXED | "peaks at L=32 and shrinks beyond it". | ALGORITHMIC_RESULTS.md |
| N12 (l.703) | FIXED | Match-Query index arm stated at warmup+cosine, lr 3e-4, 300 ep; recipes table row gains lr. | run_loop_headroom.sh; train_match_query.py default lr |
| N13 (l.756, l.2096) | FIXED | Order-3 gate caveat (0.634 vs 0.536) added to Tab. knob caption and App. knob paragraph. | KNOB_SWEEP.md gate section |
| N14 (l.813) | PARTIAL | Parameter range replaced by training-log counts across all five arms (613,576-614,664); the pre-registration counts in the old caption do not match the logs. Auditor's 614,090 (other batch) not used; spread percentage dropped (NEEDS-NUMBER). | runs/minigrid_em/logs/*_s0.log; MINIGRID_EM.md; MINIGRID_EM_PREREG.md |
| N15 (l.925) | FIXED | -0.287 attributed to Selective RoPE's generator (its raw torus cost minus its recency cost). | MONOTONE_RESULTS.md Q3 |
| N16 (l.937) | FIXED | "constrained checkpoints probed, seeds 0-2 of 12 per arm"; opposition scores labelled seeds 0-2. | SIGN_PROBE.json; probe_sign.py; run_sign.sh |
| N17 (l.973) | FIXED | Pos/CARoPE T=128 cells filled: -0.004 (MDE 0.010), -0.017 (MDE 0.023), unmeasured. | SIGN_ABLATION.md T=128 table |
| N18 (l.985) | FIXED | Signed - RoPE at T=128 loss-matched (-0.021, detectable negative) stated in text and table. | SIGN_ABLATION.md |
| N19 (l.1086) | FIXED | "(App. D.1)" and paper added to src. | papers/txt/mapformer.txt App. D.1 |
| N20 (l.1120) | FIXED | T=2048 recency values labelled not registered (exploratory). | MONOTONE_RESULTS.md exploratory table |
| N21 (l.1203) | FIXED | "Every recency contrast ... between -0.936 and -0.987". | SPREAD2_RESULTS.md; D5_RESULTS.md; MONOTONE_RESULTS.md; PAPERTASK_RESULTS.md |
| N22 (l.1403) | FIXED | 0.578 to 0.928 labelled primary readout; 0.609 to 0.930 over all trained offsets; App. H value labelled primary. | SPREAD_RESULTS.md |
| N23 (l.1466) | FIXED | Rule 9 caption: over the 96 EMPair and EMPairConst runs. | PAIRSPLIT_RESULTS.md; analyze_pairsplit.py |
| N24 (l.1390) | FIXED | Curriculum epochs cell "11 (8/8)*", footnote "not comparable". | SEARCH_RESULTS.md S3 |
| N25 (l.1549) | FIXED | Claim 5 opening, abstract, intro and conclusion acknowledge hierarchy's ceiling-level +0.012 at L=16 (MDE 0.002, 16/16); "hierarchy cannot operate" at L=16 corrected. | HIER_PARITY.md |
| N26 (l.1593) | FIXED | "the best arm in its batch". | MQ_RANK_2X2.md |
| N27 (l.1722) | FIXED | n=12 raw +0.129 marked detectable; loss-matched at its MDE; "only after loss-matching" scoped to n=5. | L15_LOOP_2X2.md; L15_ABLATION.md |
| N28 (l.1967) | FIXED | LSTM/Vanilla T=2048 stated as evaluated on the training map. | LONG_SEQ_clean.md; long_sequence_eval.py |
| N29 (l.2014) | FIXED | "MapEM-os (commutative)" -> "single-origin MapEM (commutative)" in table and contrast. | ABLATE_FAMILY_TREE.md; FAMILY_TREE_RESULTS.md |
| N30 (l.2029) | FIXED | 14.4x attributed to MapEM-NC-L (also in Sec. 4); 3.9x for single-origin MapEM; NC-NL not timed. | TIMING_BENCHMARK.md |
| N31 (l.2115) | FIXED | Parenthetical: +0.263/+0.230 from one evaluation; budget-curve evaluation of same runs gives +0.264. | CONTINUOUS_ALLOC.md; H12_BUDGET_CURVE.md |
| N32 (l.2125) | FIXED | "about 0.15" replaced by per-arm values at T=512 (Vanilla 0.653->0.801, RoPE 0.620->0.798). Auditor's 0.148-0.178 range not used (not in a file). | MINIWORLD_FIXED_FINDINGS.md sec. 3 |
| N33 (l.2143, l.2172) | FIXED | Lengths and raw/loss-matched stated. | SELECTIVE_ROPE.md; MAPPOPE_R4_RESULTS.md |

## Section 2: overclaims and language

| id | disposition | change / reason | source checked |
|---|---|---|---|
| O1 (L6) | FIXED | Abstract: per-pair origins "add 0.215 over single-origin MapEM (n=48): 0.124 pathway capacity with kernel still shared, 0.091 per-pair freedom, both detectable", in "a separate batch" from the n=8 0.375 gap; no fraction of 0.375 implied. Claim 4 opening "recover" -> "add"; intro "recovers part of that cost" -> "detectably raises accuracy". Auditor's n=48 mean 0.683 not used. | PAIRSPLIT_RESULTS.md; PAIRORIGIN_RESULTS.md; RECENCY_EM_RESULTS.md |
| O2 (L2) | FIXED | Abstract, Claim 3 opening and Sec. 7.3 crossover sentence scoped to the two WM-type generators (WM -0.060 unmeasured, SRoPE -0.068 detectable, both raw); MapEM -0.198 (MDE 0.155) detectable, -0.043 at matched loss unmeasured; EM-WM difference -0.138 (MDE 0.181) unmeasured; torus cost given raw (0.363) and matched-loss (0.280) so scales are explicit; "cheap there" scoped to MapWM. | MONOTONE_RESULTS.md P1-P3, Q2, Q3; SIGN_ABLATION.md |
| O3 | FIXED | Intro: "pays under three conditions, each supported by an intervention on its own task; that rank acts through conditioning is descriptive" ("only" dropped). Conclusion: "supported by", conditioning descriptive. | ACTION_GEOMETRY.md; RANK_SWEEP.md; report Sec. 7.2 |
| O4 (L9) | FIXED | Claim 1 title drops "not the encoding"; intro and contributions row scoped to paper recipe at training length, contributions row notes converged replication pending; Sec. 6.1 adds floor/ceiling caveat, PoPE detectable at T=512 (+0.037, MDE 0.032) and T=1024 (+0.101, MDE 0.065), MiniGrid encoding +0.076, and that PAPER2X2 will settle it. Nothing presupposes PAPER2X2. | BASELINE_TABLE.md; POPE_WRAPPING.md; MINIGRID_FULL_2X2X2.md; PAPER2X2_PREREG.md |
| O5 (L7) | PARTIAL | "ties" replaced by "no detectable difference" (claim opening, paragraph heading, fixed-offset sentence, capacity sentence) with the MDEs that exist (vocab 0.169). Not scoped to "at training length" as the lead asked: the MiniGrid contrasts are at T=512/1024 (training T=128), so that scope would be false; text says so instead. | MINIGRID_EM_FIX.md; VOCAB_EM.md; PAPERTASK_RESULTS.md; tab:tasks (MiniGrid T=128) |
| O6 | FIXED | "learns faster" -> median 54 vs 99 epochs to loss < 0.5, descriptive (both places). | SEARCH_RESULTS.md S3 |
| O7 | FIXED | "not simply a loss gap"; Sec. 10 MapEM-length statement labelled exploratory, gate failed. | PAPERTASK_RESULTS.md |
| O8 | FIXED | "failures are consistent with search rather than with the absence of a solution". | THEORY_SEARCH_AND_LENGTH.md T1 |
| O9 (L8) | FIXED | Section title "a powered negative" -> "no measurable benefit"; "excluded" -> "would likely have been detected, in arms that are not converged"; slope stays unmeasured; organisation paragraph "negative" -> "result". Auditor's one-sided bound 0.084 not used. | MQ_NOISE_2X2.md; MQ_NOISE_2X2_C2.md |
| O10 | FIXED | "with no evidence that it works by inference"; landmark gain "not evidence for the filter's measurement mechanism (n=3, exploratory)". | LM200_ABLATION.md; L15_ABLATION.md |
| O11 | FIXED | "Hierarchy's gain is generic compression..." -> "Room-aligned pooling did not help (n=3, exploratory)". | COMPOSITIONAL_EXPERIMENT.md; HIER_RECHECK.md |
| O12 | FIXED | Rank paragraph: "gate arm (sigmoid gate plus a readout swap) ... cannot be attributed to the gate"; App. G: "where that arm gains / loses". | SELECTIVE_ROPE.md CONFOUND block |
| O13 | FIXED | "suggests"; cites the traceable separation measurement (C1 ratio 25.0 +/- 22.4 at r=2, 35.9 +/- 19.0 at r=4, Tab. fig4, separate batch). GATED_RESULTS.md's "about five times" traces to no results file and was not used. | GATED_RESULTS.md; PAPER_FIG4_REPRO.md |
| O14 | FIXED | "restore the effect to above its baseline" -> "+0.488 against the baseline's +0.438 (no MDE recorded; trained at matched supervised events, four times the baseline's batches)". | KNOB_SWEEP_n8.md; run_seeds_n8.sh |
| O15 | FIXED | Claim 2 opening: converged cells +0.178 (n=5, 400 ep) and +0.305 (n=3, 800 ep, exploratory); no settled "sign inverted" direction. | ALIASING_CONTROLLED.md; MINIWORLD_ENDPOINTS.md |
| O16 | FIXED | "(3 seeds, exploratory)"; "appear to be at a capability limit". | RECENCY_RESULTS.md |
| O17 | FIXED | Exploratory labels on frequency control, recursion horizon, fixed-map MiniWorld, refine-theta; blanket sentence for App. F n=3 results. See X1. | FREQ_CONTROL.md; LOOPED_PILOT.md; MINIWORLD_FIXED_RESULTS.md; NOISE_REFINE.md |
| O18 (L3) | FIXED | Sec. 8.1 "Both contrasts are detectable and the interaction is super-additive" replaced by unpaired loop result and paired-only interaction; parity sentence and contributions row reworded. Abstract recursion sentence kept (supported unpaired) and labelled "in an unpaired analysis". | REFINE_RESULTS.md; LOOP_HEADROOM.md |
| O19 | FIXED | "The raw sign cost ... not specific"; abstract gives raw and matched-loss torus costs explicitly. | MONOTONE_RESULTS.md Q1 and exploratory rows; SIGN_ABLATION.md |
| O20 | FIXED | "loses to CoPE and PaTH" -> perplexity improves on RoPE at all lengths but degrades more sharply than the numbers reported for CoPE and PaTH. | papers/txt/mapformer.txt App. B.5 |
| O21 | FIXED | 64^2 n=5 "seeds pooled across two runs, compared unpaired". | run_match_scale.sh; MATCH_QUERY_SCALE.md |
| O22 | PARTIAL | Kept with caveat, labelled exploratory: the first scored stitch task was dropped (negative control 0.617); the reported numbers are an attention probe on revisit-trained models whose episodes were never gated; "correct room" -> "correct look-alike cell"; approach-observation confound stated. Not dropped because the numbers do not come from the voided scored task. | CSCG_TASK_GATES.md; STITCH_ATTENTION.md; train_stitch.py docstring; RESULTS_INDEX.md |
| O23 | FIXED | "Closing the content leak ... removes the residual at 8x by the registered seed-count readout". | NOLEAK_RESULTS.md; UNFREEZE_RESULTS.md |
| O24 | FIXED | Heading "Mechanism, by intervention, on descriptive n=8 readouts". | PAIRCONST_RESULTS.md |
| O25 | FIXED | "the paper is right about what is expressible" -> "the learned codes are two-dimensional, as the paper's argument expects (r=1 was not run)". | ACTION_GEOMETRY.md |
| O26 | FIXED | "refuted twice" -> "not supported in either of two recipes". | MQ_NOISE_2X2.md; MQ_NOISE_2X2_C2.md |
| O27 | FIXED | Contributions row: "shows no detectable scaling with drift"; gate "buys no measurable accuracy". | same |

## Section 3: referee objections

| id | disposition | change / reason | source checked |
|---|---|---|---|
| R1.1 | FIXED | Contributions row scoped to paper recipe, training length, "converged replication pending". | pending.tex; PAPER2X2_PREREG.md |
| R1.2 | FIXED | Covered by O4. | POPE_WRAPPING.md |
| R1.3 | NEEDS-NUMBER | Covered by E3. No unpaired Q1 (path-int minus index, 128^2) exists and per-seed ranges overlap (0.110 vs 0.148), so text reports arm means and the paired estimate with the pairing caveat and states that no unpaired test was run. | LOOP_HEADROOM.md per-seed; REFINE_RESULTS.md |
| R1.4 | REJECTED | Already answered in the report (intro, Limitations); no change. | report.tex |
| R2.1 | FIXED | Added: recoding removes the heading-to-displacement computation; the result shows a fixed per-token increment cannot perform it, not that a map can do without it. | report Sec. 5.1 |
| R2.2 | FIXED | Covered by E21. | run_seeds_n8.sh |
| R2.3 | REJECTED | Already answered (effect size and sd caveats present); no change. | MINIGRID_EM.md |
| R2.4 | FIXED | Covered by E4 and O15. | ALIASING_CONTROLLED.md |
| R3.1 | FIXED | Covered by E5. | SIGN_ABLATION.md |
| R3.2 | FIXED | Covered by O2. | MONOTONE_RESULTS.md |
| R3.3 | FIXED | Covered by O3. | ACTION_GEOMETRY.md |
| R3.4 | REJECTED | Already answered (mediator caveat; Pos/CARoPE init confound; Abs clean); no change. | SIGN_ABLATION_PREREG.md |
| R4.1 | FIXED | Abstract: "at a shared budget"; "On that task, constructions place the gap in search rather than representation". | report Sec. 6.3.3, 6.3.8 |
| R4.2 | REJECTED | Already answered (Scope paragraph, Limitations); no change. | report.tex |
| R4.3 | FIXED | Covered by O1. | PAIRSPLIT_RESULTS.md |
| R4.4 | FIXED | Covered by O5 and O7. | PAPERTASK_RESULTS.md |
| R5.1 | FIXED | Covered by E2, E3, O18. | REFINE_RESULTS.md |
| R5.2 | FIXED | Claim 5 opening and conclusion: Match-Query loop gain "associated with, not separated from, more reliable training" (no loss-matched analysis exists); torus gain vanishes at matched loss. | LOOP_HEADROOM.md (no loss analysis); L15_LOOP_2X2.md |
| R5.3 | FIXED | Covered by O9 and O26. | MQ_NOISE_2X2*.md |
| R5.4 | FIXED | Covered by E7. | SELECTIVE_ROPE.md |

## Section 5: bibliography

| id | disposition | change / reason | source checked |
|---|---|---|---|
| B1 (L11) | FIXED | whittington2022temt: authors Whittington, Warren, Behrens; unverified ICLR venue removed (now @misc); not-in-corpus note added. | papers/txt/mapformer.txt ref. [17] |
| B2 | FIXED | whittington2020tem: not-in-corpus note citing the local sources for journal/volume/pages; unverifiable issue number removed. | papers/txt/tale_two_algorithms.txt; papers/txt/kv_brain.txt |
| B3 | FIXED | Intro ordering: GRAPE and Puranik (who gives the Jordan-form reason it is exhaustive); Vetcha "posted between the two". Audit's "Jordan-form terms are Puranik's addition" softened because GRAPE also has Jordan structure for its additive lifts. | papers/INDEX.md; survey_infext.txt (3 Jan 2026); puranik_janestreet.txt (22 Apr 2026); grape.txt |
| B4 | PARTIAL | Zoology: eprint 2312.04927 and full authors added (confirmed in local corpus). MiniGrid/Miniworld (2306.13831), Habitat (1904.01201) and MoR venue not confirmable from local files; entries left with their existing "not verified" notes. | papers/txt/dape.txt, hgrn2.txt, gla.txt |
| B5 | REJECTED | Key gu2024mamba with year 2023 is cosmetic; the entry itself is correct. No change. | refs.bib; papers/txt/mapformer.txt |
| B6 | REJECTED | Audit confirms no novelty claim contradicts prior art; nothing to fix. | papers/INDEX.md |

## Extra

| id | disposition | change / reason | source checked |
|---|---|---|---|
| X1 | FIXED | App. F frequency control said "Freezing omega changes accuracy by +0.004 / -0.008"; the source difference is learned minus frozen, so the sign was inverted. Now "learned minus frozen omega is +0.004 / -0.008". | FREQ_CONTROL.md |

## Build

`pdflatex report && bibtex report && pdflatex report && pdflatex report`: 40 pages, zero errors, zero undefined
references or citations (one float-specifier warning).
