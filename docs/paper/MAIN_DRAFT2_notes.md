# Notes on MAIN_DRAFT2 (2026-10-21; G-1 and G-3 folded in 2026-10-23; SIGNSEP, G-4 and RANKRECIPE 2026-10-24)

Drafted from `docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md`, draft 1 (`MAIN_DRAFT1.md`, `MAIN_APPENDIX1.md`), the theory
appendix v3, the seven expansion reports, and the results that landed after the plan (DELTA_FREE, SROPE_TORUS, SIGNROUTE,
`docs/audits/2026-10-20/WHAT_WHERE_T1024.md`, `figs/fig_summary.png`). Files: `MAIN_DRAFT2.md` (main),
`MAIN_APPENDIX2.md` (A-O). Nothing committed; no results file, code, run directory, figure or existing draft was edited
(the appendix reuses draft-1 appendix blocks by copy, with section references renumbered).

## Script-audit status

The independent results-blind audit of every analysis script (`docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md`,
CPU-9, commit cb13feb; line references to 45abed7) is **applied** (2026-10-21; list below). Every number formerly labelled
[A-pending] has been cleared by the audit (re-derived exactly from its script) or replaced by a corrected number with a
committed generator; the label is now **[A-audited]**. The one script that does not reproduce, `sign/step_basin.py`
(not deterministic), is labelled exploratory wherever it is used (Sec. 7.3, App. H7). The what/where T = 1024 script
(`docs/audits/2026-10-20/causal_whatwhere_T1024.py`) was in the audit's scope (Part 6: REPRODUCED, MINORs only), so K1 is
[A-audited] too.

### Changes applied (old -> new; audit item)

Counts: 4 BLOCKERs, 20 MAJORs, 46 MINORs touching draft text (MINORs that touch only reports or code docstrings are left
to the reports' CORRECTED banners, except where noted).

BLOCKERs.
- B1 / M-LG1 (Sec. 1 contribution 5, Sec. 8 paragraph 2, Table 7, App. B, App. J6). Headline map evidence is now the whole
  non-retrace stratum on all 200 registered walks: "the 5.9% that retrace gets wrong, places reached by another route ...
  path 0.961 (fresh), RoPE 2L 0.138, constant 0.112" -> "the 13.4% that retrace cannot reach ...: path 0.956 all 8 seeds
  (0.978 fresh; 0.934 eval mode), RoPE 2L 0.411 (seeds 0.34-0.55) and RoPE 1L 0.454, both below the constant 0.506
  ('nothing')". The object-target sub-stratum is kept with its correct label ("the revisits retrace gets wrong: ... whose
  cell holds an object; on blank cells retrace's default is right"): 6.6% (was 5.9%), path 0.925 (was 0.920) / 0.967 fresh
  (was 0.961), RoPE 2L 0.129 (0.138), RoPE 1L 0.004 (0.002; "it predicts 'nothing'"), best constant 0.088 (0.112,
  'nothing' excluded by construction); seeds at ceiling 0.936-1.000 (0.944-1.000). Table 7 "all": 0.993 / 0.772 / 0.505
  (was 0.994 / 0.779 / 0.507), with "retrace applies" (0.999 / 0.828 / 0.513) replacing "retrace correct". RoPE 2L best
  seeds 0.927 / 0.926 against 0.934 (were 0.932 / 0.930 against the 100-walk 0.941). Generator:
  `scripts/language/tw_strata_200.py` -> `tw_strata_200walks_out.txt` (new; `N_WALKS=100` reproduces `tw_strata_out.txt`).
- B2 / M-MP1 (Sec. 7.4 last paragraph, App. L2). C0 by stratum re-read on the per-stratum head: gap >= 128 "0.81-0.84"
  -> "0.80 / 0.91", wrap-only "0.41-0.46, against 0.84 at rank 4" -> "0.40 / 0.48, against 0.97 at rank 4 under either
  score"; PoPE-score long-gap C 0.840 -> 0.908 (p 0.029 -> 0.005); r4 - r2 wrap C0 +0.568 (p 0.0009) and +0.488 (p 0.012);
  "C0 is not a sufficient statistic" deleted (artefact of the head rule); short gaps "code term unchanged" -> "moving +0.06".
  Generator: `kernel_probe.py` with `HEAD_RULE=stratum` -> `kernel_probe_score_rank_hstratum_{out.txt,json}`,
  `probe_stats.py` with `SUFFIX=_hstratum` -> `probe_stats_hstratum_out.txt` (code fix; default output unchanged).
- B3 / M-SG1 / M-SG4 / M-SG7 (Sec. 4 definition, Sec. 7.1 text and Table 4, Sec. 7.3, Sec. 1, Sec. 9, App. E7, G3, H7). One
  definition, stated once in Sec. 4 (and E7.1'): sign_clock's ratio with a symmetric map, usage pooled over heads (per head:
  within the head). Table 4: |.| small starts "1.55-1.68" -> 1.47-1.57 (ring; with torus 1.25); frozen latents 1.39 -> 1.18
  (Sec. 7.3 too); past pi 0.31 -> 0.32; torus |.| 1.22 -> 1.25; unsolved rank 2 "0.51-1.91 (head means)" -> 0.54-0.69 run
  means (medians 0.58-0.79); solved rank-2 heads "0.029-0.072 / 0.94, 1.43" -> "0.03-0.07 / 0.54, 0.68"; rank 3/4
  0.043-0.051 (unchanged). "Every failure ends with a clock about as large as its map" -> "a clock of the order of the map,
  or (4 of 20 unsolved rank-2 runs) collapsed, clock-free heads; one more just above the SOLVED threshold". "Failing
  non-negative arms end at 1.22-1.68" -> 1.25-1.57; caption says the |.| rows are >= 1 by construction. Generator:
  `scripts/sign/clock_table.py` -> `clock_table_out.txt` (new; reproduces sign_clock's one-sided values as a check).
- B4 (Sec. 7.4). "None of the 16 runs finds rank 4's two-dimensional code" -> "Only 1 of the 16 runs finds a
  two-dimensional code (rank 4: 8/8)".

MAJORs.
- M-RK1 (Sec. 7.2, G4). Clock decomposed: a and o each 0.02-0.04 of a map step, anti-parallel (cosine -0.79 to -0.999),
  in-plane at T = 1024; only the residual (0.001-0.025 of a map step) is off-plane. Generator: `rank/clock_decomp.py` (new).
- M-RK2 (Sec. 7.2, G4, G6). "show it being used" -> "consistent with it but do not discriminate it from plain removal";
  "each time by parking it off the plane" -> "its latent clock shrinks 14-85x in absolute size, and the residual
  (0.002-0.005 of a map step) lies off the plane (share 0.91-1.00)"; G4 heading "park" -> "where the residual lies".
- M-RK3 (Sec. 7.2 -> G4). The rank-sum p 0.018 sentence is moved to App. G4 with both caveats (p 0.29 on |c_perp| / |U|;
  s1 solves under plain continuation and starts near-clock-free; without s1 p 0.048).
- M-RK4 (G6). R2Clock "in the slot (off-plane 1.00)" -> "fixed unit clock by construction"; R2Clock dropped from the P4
  count; rank-2 control 2/6 reported; "parked" rows -> "off-plane ..., then removed (66x / 180x / 5.5x)".
- M-SG2 (Sec. 7.1, H7). "Every model that succeeds ends with clock-free heads" -> "from a small start"; "0.001-0.072" ->
  "arm means 0.001-0.051, no solved head above 0.14"; large-start solved runs listed in H7.
- M-SG3 (abstract, Sec. 1, contribution 3, Prop. 2, Table 1, Sec. 7.1, Sec. 9, E7, E14). Proposition 2(a) restated as a
  bound on the position factor for n|k| < pi ("serves ... and loses the rest" dropped); a per-move tick of pi is
  parity-compatible on the even torus (Prop. 2(b)); two solved signed torus runs from a start past pi carry 0.43 / 0.99 of
  usage on channels with per-pair clocks near pi (clock_table.py); the failure-signature claim rests on the measured
  strata.
- M-SG5 (G3). rank_clock's in-span at r = 2 noted as 1 by construction, not reported (it was not in the main text).
- M-SG6 (Sec. 7.3, H7). Step-basin numbers labelled exploratory with the observed re-run spread (0.944 -> 0.509 / 0.939,
  0.548 -> 0.450, 0.921 -> 0.839); "does not reach halfway" -> "no run from halfway reached 0.9 within this budget".
  Not made deterministic: that needs every cell re-run (not trivial).
- M-MP2 (L2). Action-key attention and same-cell mass labelled "(retrieval head)"; both-head means 0.223 vs 0.147, p 0.15.
- M-RC1 (contribution 4, Sec. 7.4, M3, M4). "never the code ... on an unchanged code" -> "never change the code family;
  the codes they learn stay far from rank 4's"; "no better than plain rank 2's (p 0.27-0.28)" -> "not measurably more
  readable (... below MDEs of 0.07-0.08)"; M4 adds the class shift (1D 8/16 vs 0/8, p 0.022). Generator:
  `recurrence/reader_mde.py` (new; MDEs 0.074 / 0.080, the audit's 0.080 / 0.087 used a different formula).
- M-RC2 (M2). Best head per seed: "0.03 -> 0.36 -> 0.46 -> 0.52 (head 0)" -> 0.03 -> 0.50 -> 0.64 -> 0.63; "64%" -> 81%;
  action-token mass 0.01-0.15. Generator: `recurrence/pass_strata_besthead.py` (new).
- M-DP1 (Sec. 6, F3, F5). "carry a global count in the residual stream (R^2 0.69 for P_t / t)" -> "carry a displacement
  code over the last ~64 moves (R^2 0.72 for the last 64 moves at moves 96-127); displacement older than 64 moves is not
  decodable (R^2 0.01), so no global count is detectable on the ring either"; "show which one each task gets" -> "training
  finds only rules that need no count". Generator: `depth/ring_probe_window.py` (new; its values differ from the audit's
  by <= 0.03, e.g. 2L P/t at t >= 64 0.31 vs 0.33).
- M-DP2 (F5). "Descriptive decomposition ... reproduced" -> the p-invariant mixture does not reproduce the one-layer
  accuracies (0.678 / 0.683 / 0.749 vs 0.710 / 0.677 / 0.824); retrace accuracy itself falls 0.797 -> 0.692.
- M-DP3 (Sec. 6, F5). "the floor on everything else" -> "at p_plus 0.5 and 0.75 (0.487 against 0.491-0.493; at 0.9 an
  unexplained 0.592 against 0.502)". Generator: `depth/ring_class_floors.py` (new).
- M-DP4 (Sec. 6, F1). Generators added: `summarize_torus.py --segments` -> `out/torus_segments_out.txt` (fixed-gap
  segments table; 98.5% / median 37 / minimum 11; wrap-only 99% / 65); torus_depth.py was re-run into the scratchpad so
  its committed JSONs are untouched. F1 table relabelled "mean of per-seed accuracies" (pooled: 0.896, 0.785).

MINORs (draft text).
- Causal (Sec. 9, K1): "92.5% of the score's variance" -> "92.5% of the within-row sum of squares (SS(P) / SS(S))";
  "-0.032" + "8/8 seeds lower, sign-flip p 0.008, uncorrected; Holm over the 24 contrasts 0.08"; "object identity removed"
  -> "token identity within type removed"; "kernel x content gain" -> "x per-pair content gain plus offset"; Spearman 0.84
  + "p 5e-5" with a committed generator (`scripts/whatwhere/whatwhere_stats.py`, new); "|.| models depend" -> "7 of 8 |.|
  runs lose at least 0.15"; K1 collapse sentence -> "the four lowest softplus runs fall to 0.03-0.07 (5-12%); |.| runs keep
  0.22-0.77 except Abs_r4 s0 (0.98)"; K1 notes the `.txt` came from `--report`.
- Rank (Sec. 7.2, G4-G6): random baseline "0.50-0.71" -> "0.40-0.71 (median)" (table: 0.58 -> 0.50, 0.50 -> 0.40); C1
  "57 of 62, against 28.7" -> "50 of 55 counting each network once, against 25.1" (`rank/c1_runlevel_once.py`, new; the
  audit's 24.7 used another RNG draw); r_beta "0.93-1.05" -> 0.79-1.10; G5 "14 clean / 3 drifting" -> 11 / 5; G6 C7 "pooled
  p 0.012" -> "test p 0.073; pooled 0.012 descriptive"; G4 "solved rank-D heads lam 0.000-0.001" -> "the clean head ...;
  the other head often drifts"; G4 escape F unsolved heads show the same residual geometry; G6 grad_hiding is one
  stochastic sample with dropout on; G4 notes the native-gauge metric (balanced gauge agrees, audit).
- Depth (Sec. 6, Table 3, F1, F3, F5): "exactly the measured 0.943" -> "decomposes"; ring run rows "wrap-only revisits
  included" plus plain-only values (`depth/ring_plain_bins.py`, new: 2L 0.910 / 0.828 / 0.742, 3L 0.999 / 0.996 / 0.988);
  Table 3 caption "eval mode" and "contrasts at n = 4 carry no test"; F3 d_w at t < w note; "(seed means)" on the
  0.874 / 0.493 sentence.
- Language (Sec. 8, Sec. 9, J1, J2, J4, J5): Spearman +0.905 + "exact permutation p 0.0046"; "text world 0.06-0.23
  (analysis)" -> "(23 of 24 learned-step models; the unconverged one 2.5)"; J5 "none of that kind" -> "one of that kind
  (MapWM s13)"; the 0.98 ceiling cut and the lift sizes declared post hoc; J1 aside net = noun offset by construction; J2
  and Sec. 8 "agreement 0.98-1.00" qualified (informative on the 10 clock models only).
- MapPoPE (L2): kernel argmax restricted to odd offsets: "3 (414), 7 (277), 10-11 (147) and 2 (168)" -> "3 (569), 7 (342),
  11 (156), 15 (21)" (`mappope/kernel_argmax_odd.py`, new); ~60 contrasts without Holm stated (p 0.03-0.04 do not
  survive); "+0.114 ... replicating the registered +0.115" -> "consistent with"; C0 - C +0.20 is a coherence statement;
  PoPE's 0.307 labelled "the more recency-weighted head".
- Recurrence (Sec. 7.4, M2, M3): "one pass 0.381" -> "(mean; 7 of 8 seeds below the 0.51 floor)"; model on 100 walks vs
  reader on 50 (<= 0.004 on matched walks); reader under-fits (L long-gap margin probably 0.03-0.05 too large); entropy
  column from 10 walks labelled.
- Sign (Sec. 7.3, H4, H7): Holm over the 8 basin tests stated; lattice tolerance 0.05 rad stated; latent_geometry's MECH
  init values sourced to the audit; ranges 18.8-22.2 / 13.2-16.0 and 2.1-4.1 / 8.1-8.9, unweighted; H4 "the unsolved runs of
  both classes have the relation" -> "the unsolved |.| runs and 2 of 3 unsolved signed runs (s1300 0.68)"; cone check
  labelled an identity.
- Floors (App. B): ring retrace "0.864 on the registered eval stream" added; text-world NEEDS-MAP floor row replaced by the
  200-walk non-retrace and object-cell rows. (The sign_floors "maps 10-25" MINOR refers to APPENDIX_draft6; draft 2 has no
  such sentence.)

### Left open

- MAIN_DRAFT2 Sec. 7.1 "signed steps cancel ... in 70-93% of channels and |.| in 2-10%" comes from `fig_wrap_share`
  (App. H6), outside the audit's scope and not audited.
- Numbers quoted from the audit's own scratch checks without a committed generator: the balanced-gauge medians
  (0.94-1.00, G4), SO4_high s1206's parity-stratum accuracy (H7), the escape-F unsolved-head residual geometry (G4), the
  reader's 5x-steps sensitivity (M3), the MECHANISM initialisation values 0.056 / 15.6 / 5.0 (H7). Each is attributed to
  the audit in the text.
- `step_basin.py` not made deterministic (exploratory label instead; rerunning every cell 2-3 times would cost hours of
  CPU).
- Figures: plan Figs. 8, 9 and 11 are now built from the corrected generators (2026-10-21; draft Figs. 4, 5, 10; see
  "Figures added" below). Still not built: plan T1 (schematic), 10 (loop reader), 7d-e, G2, H7, J2.
- The expansion reports' bodies and the plan's claims table (rows 22, 30-31, 35-37, 41, 44, 46, 56; X7, X9, C-1, Table 5)
  still carry the old numbers; their CORRECTED banners point to the audit, and the draft is current.
- `kernel_probe.py` still reads score_rank seeds from a directory listing (pin them before any pilot dir is added there).

## G-1 (RANKCLOCK) folded in (2026-10-23)

Source: `RANKCLOCK_RESULTS.md` (registered; read in full), `RANKCLOCK_PREREG.md` (arms Sec. 2, existence Sec. 3, analysis
Sec. 5, Amendment 1), `RANKCLOCK_ANALYSIS.txt`, `RANKCLOCK_VERDICTS.json`, `RANKCLOCK.md`, the CLAUDE.md rank block of
commit 80fc2f4. No results file, code, run directory or figure edited; nothing committed.

### Changes (old -> new)

- **Abstract.** After "... keep or re-find an installed solution." added: "Forbidding the clock by construction makes rank
  2 solve the torus on every seed (8 of 8), while zeroing only the observation steps does not (1 of 8)." The paragraph
  was re-wrapped to 120 columns. Length 316 -> 344 whitespace tokens (open item 8).
- **Sec. 1, contribution 3.** "Causal tests of the routes: freezing the move latents removes the collective route (1/8
  against 7/8), and a non-negative readout with a signed increment restores it (16/16)." -> "... (1/8 against 7/8), a
  non-negative readout with a signed increment restores it (16/16), and on the torus a rank-2 step map with the clock
  forbidden by construction solves 8/8, like rank 4, against 1/8 when only the observation steps are zeroed: the rank-2
  obstacle there is the clock, not capacity."
- **Sec. 4, after Conjecture 1.** Added: two empirical claims tested in Sec. 7.2 -- the clock blocks rank D on the torus
  (registered support) and the *clock-exit account* (rank above D helps because parking gives the clock a map-free exit;
  Prop. 4(c) shows the exit exists, not that training uses it; a conjecture, registered test PARTIAL).
- **Table 1.** Conjecture 1 row "consistent with rank 1 learning the ring; not tested" -> "...; its premise, a clock that
  persists at rank D, has the support of the row above; the feedback itself not tested". New rows: "At rank D on the
  torus, the clock is the obstacle | registered causal support (shared rank 2, T = 1024, 900 epochs) | clock forbidden
  8/8 against observation steps zeroed 1/8 (Sec. 7.2)"; "Clock-exit account: rank above D helps through parking |
  conjecture; registered test PARTIAL | explicit clock exits at rank 2 3/8 against 1/8, Fisher p 0.57, most runs still
  descending (Sec. 7.2)". Prop. 4 and Prop. 5 rows unchanged (proved statements; G-1 does not test them).
- **Sec. 7.2.** New paragraph "On the torus the rank-2 obstacle is the clock (registered)" between "Rank against the
  map's dimension" and "Why the extra rank helps": gates (r2 1/8, r4 8/8), the drift-forbidden arm defined from prereg
  Sec. 2 (pair steps +-W_out W_in (e(+x) - e(-x)) / sqrt 2, observation steps 0; clock zero by construction) 8/8 at
  1.000, 1.000 at 2T against r4 0.986 (descriptive); actions-only (observation steps 0, four move steps free, so the moves'
  own clock remains) 1/8 = r2's 1/8; DF vs AO Fisher p 0.0014; "removing the clock is sufficient and removing its
  observation part is not: what stops rank 2 is the clock, not the rank's capacity to hold the map". "Why the extra rank
  helps": old second sentence "The stored checkpoints are consistent with it but do not discriminate it from plain
  removal ..." -> the registered primary first (per-axis slot, +128 parameters, zero init, cancels common and
  axis-differential parts; 3/8 vs 1/8, Fisher p 0.57, accuracy 0.972 vs 0.918; per-token slot 3/8; PARTIAL against the
  6/8-at-p < .05 and 2/8 bars; every class contains a solution and keeps or re-finds it; 4 of 5 / 3 of 5 unsolved slot
  runs DESCENDING; "not supported at the registered bar, and not refuted"), then "The stored checkpoints are consistent
  with parking but do not discriminate it from plain removal ..." (audit wording kept: "consistent with", residual
  off-plane; nothing upgraded). Rest of the paragraph unchanged; re-wrapped.
- **Sec. 9, "What the two failure axes share".** Added after "... parking, that costs the map nothing.": "That the clock is
  what blocks rank D is tested on the torus: with the clock forbidden by construction, rank 2 solves on every seed
  (registered). That parking is why one more dimension helps is not shown: an explicit clock exit at rank 2 solves 3/8
  against 1/8 within 900 epochs, most unsolved runs still descending (registered PARTIAL), so it remains a conjecture."
  This is the one place outside Sec. 7.2 that gives the slot numbers.
- **Sec. 9, design guidance.** Added: "Where the inverse pairs of actions are known, also tying their steps to exact
  negatives removes the clock by construction, and rank D then suffices (rank 2 on the torus, 8/8)."
- **Sec. 10.** Budget bullet: the clock-exit test's unsolved slot runs mostly still descending (no numbers repeated).
  Rank bullet: the clock-route batch is shared rank only, one task and length. New bullet: weight decay 0.05 throughout
  except where marked; rank 4 at wd 0 solved 1/8 vs 8/8 (Fisher p 0.0014; App. G7), post hoc, unregistered, "whether rank
  4 needs weight decay is untested" (the only main-text mention). Theory bullet: "developed after the registered batches"
  -> "..., except the clock-route batch (Sec. 7.2), which was registered to test its rank account".
- **App. A3.** New row RANKCLOCK (G7): seeds 1900-1907, 900 epochs, standard attention (no `--fast-attn` in
  `run_rankclock.sh`), code check yes (Amendment 1), reproduced no.
- **App. E9.3** status sentence and **App. E14** rows E8.5 and E9.3: the premise (a remaining clock blocks rank D) has
  registered support on the torus; the feedback is not tested; parking as why rank > D helps is a conjecture with a
  PARTIAL registered test.
- **App. G7 (new).** Design; the two clock exits (prereg Sec. 1, Amendment 1 finding 1); arm table (prereg Sec. 2);
  existence table (prereg Sec. 3); all-arm results table and per-seed table (`RANKCLOCK_ANALYSIS.txt`); gates; registered
  readings table (P, P2, E2, X2, X2b with their rules); power; slot angle sizes; budget note; rule-2 check; post hoc list
  (wd 0, r4 end-state clock, DF at 2T, geometry medians). G6's R2Clock pilot distinguished from the G-1 slot arms.

### Numbers and sources

- r2 1/8, slotxy 3/8, slot 3/8, DF 8/8, AO 1/8, r4 8/8, r4wd0 1/8; S / T / D counts; per-seed accuracy, class and
  final-5% loss [R]: `RANKCLOCK_ANALYSIS.txt` (the bracketed number is `classify_run`'s tail, `analyze_rankclock.py` l.152).
- Accuracy at T = 512 / 1024 / 2048, mean +/- sd [R, descriptive]: `RANKCLOCK.md`, `RANKCLOCK_RESULTS.md` all-arms table.
- P PARTIAL (Fisher 0.5692; permutation 0.0500), P2 PARTIAL (0.5692; 0.2068), E2 THE CLOCK IS THE OBSTACLE (DF vs AO
  0.0014, AO vs r2 1.0000), X2 UNREADABLE, X2b NO ORDER (2 of 4, sign p 1.0000) [R]: `RANKCLOCK_VERDICTS.json`,
  `RANKCLOCK_ANALYSIS.txt`.
- Bars 6/8 at p < .05 and 2/8; power 6/8 vs 1/8 p 0.041 [prereg]: `RANKCLOCK_PREREG.md` Sec. 5.
- Arm definitions, parameters 204,373 / 204,501 / 204,437 / 204,757, DF sqrt 2 scaling, AO initial move clock 0.743, drift
  decomposition and 1.22 / 1.01 [prereg]: `RANKCLOCK_PREREG.md` Secs. 1-2, Amendment 1 findings 1 and 9.
- Existence: source r4 1.0000 / 0.0021, singular values 12.61 / 10.51 / 0.028 / 0.009, per-class frozen and trainable
  rows, 12.9 rad, 2-3x scales, AO 0.60 / DF 1.00 [prereg]: `RANKCLOCK_PREREG.md` Sec. 3 (from
  `runs/rankclock_pilot/construct_rankclock.json`, not re-read).
- Slot angles 0.142 / 0.124 / 0.068; budget class counts; rule-2 r -0.537 / -0.655 / -0.824; wd-0 1/8 vs 8/8 p 0.0014,
  6/8 STALLED, 0.766; r4 shrink 0.134, readgain 0.026, u_ratio 0.022; DF 1.000 +/- 0.001 vs r4 0.986 +/- 0.018 at 2T;
  geometry medians [R descriptive / post hoc as labelled]: `RANKCLOCK_RESULTS.md`, `RANKCLOCK_ANALYSIS.txt` (r4wd0
  geometry 0.105 / 1.56 from the analysis file only).
- 1800-epoch reference 0/8 vs 7/8 [R]: `LOOP_RANK_E1800_P1_RESULTS.md` (already in Sec. 7.2 / C4).

### Claim-map compliance

"Rank = D is hard because of the clock" is a do-not-claim (it fails in 1D). Every new sentence is scoped to the torus
and to the registered E2 reading ("on the torus the rank-2 obstacle is the clock"; "the rank-2 obstacle there"); none
says rank D is hard because of the clock in general, and Sec. 7.2 keeps rank 1 learning the ring beside it. The
clock-exit account, hiding feedback (Conjecture 1) and parking as a mechanism stay conjectures; "parking" still appears
as a measured mechanism nowhere. The wd-0 observation is in Limitations and App. G7 only, labelled post hoc.

### Rule-28 read of the touched sections (2026-10-23)

Read the abstract, Sec. 1 contributions, Sec. 4 (Conjecture 1 to Table 1), Sec. 7.2 whole, Sec. 9 first paragraph and
design guidance, Sec. 10, App. A3, E9.3, E14, G6-G7. Checked every new number against `RANKCLOCK_RESULTS.md`,
`RANKCLOCK_ANALYSIS.txt`, `RANKCLOCK.md` and `RANKCLOCK_PREREG.md`; the unsolved-slot counts (slotxy 4 D + 1 T of 5;
slot 3 D + 2 T of 5) recounted from the per-seed classes. Cross-references: "App. G7" (Sec. 7.2, Sec. 10, A3, E9.3,
E14) resolves; Table 1 cites only sections; no table or figure added to the main text, so Tables 1-7 and Figs. 1-11 keep
their numbers and order. Fixed during the read: "at p < .05" broken across a line (a line starting "<") -> "with Fisher
p below .05"; "keeps it" -> "keeps or re-finds it" (the slot classes were knocked off, peak loss ~0.2, and re-found);
"is now tested" -> "is tested" (no process history); the 2T number labelled descriptive; design guidance de-duplicated
("keep observations from stepping" stated once). Touched paragraphs re-wrapped to 120 columns.

## G-3 (SSM sign test) folded in (2026-10-23)

Source: `G3_SSM_RESULTS.md` (registered reading SIGN COST IN THE SSM; `G3_SSM_ANALYSIS.txt`, `G3_SSM_VERDICTS.json`).
Changes: abstract (ring sentence: "appears unchanged in a one-layer state-space model with the same rotation (0 of 8 against
8 of 8)"); contribution 3 (non-negative step clause: "and in a one-layer diagonal complex SSM, which has no softmax"); Sec.
7.3 new paragraph "The cost is not specific to softmax attention" and a Table 6 row; Related work (learnability cost "in
attention and in a diagonal complex SSM"); Limitations ("No recurrent or SSM baseline" replaced by the SSM test's scope:
one layer, no decay / gate / input scaling, one rate from scratch, exp unstable, softplus descending, |.| re-found not
held). Appendix: H10 (new), A3 row, E2 (g) "not built" -> built in the SSM, E14 row. Numbers: |.| 0/8 0.821 vs signed 8/8
1.000 (Fisher, perm p 0.0002); softplus 1/8; exp 8/8 RUNAWAY; retrace 0.864; constructions frozen = source 8/8, angle
error <= 2.3e-6; |.| continuations 8/8 SOLVED at 3e-3 and 1e-3 [all R]. Claim-map compliance: no "exp fails in the SSM",
no "held" at the registered bar, no Mamba-faithfulness claim.

## SIGNSEP (route vs barrier) folded in (2026-10-24)

Source: `SIGNSEP_RESULTS.md` (registered reading UNMEASURED; `SIGNSEP_ANALYSIS.txt`, `SIGNSEP_VERDICTS.json`). Changes:
Sec. 7.3 new paragraph "A collective route does not carry the non-negative step past pi" (W 0/8 vs signed 8/8 at 3e-3,
Fisher p 0.0002; P 7/8, tau_sum 2.00 on every seed; W tau_sum 0.13-0.70; F 4/8 vs |.| 3/8; 1e-3 unreadable, signed 5/8,
earlier 7/8 twice; F 3/8 at 1e-3 vs ACCOUNTS T2 1/8 [all R]); Discussion clause after the hybrid's 16/16; Prop. 4 status row;
Limitations bullet (ring, one start, one readable rate, half of the 2x2, tau mobility). Appendix: H11 (new), A3 row, E8.6
sentence (W's class keeps the barrier and has a collective route; prereg W1, W2), E14 row; the "## H. Sign details" heading
moved from above G8 (misplaced by an earlier fold) to above H1. Claim-map compliance: only the half-reading BARRIER BLOCKS is
stated as a finding; no "the barrier is THE obstacle", no "routes are irrelevant", nothing at 1e-3. Draft 6 Sec. 4.3: one
sentence after "is open"; its PDF rebuilt (source changed).

## G-4 (torus best-rate sign sweep) and RANKRECIPE (rank-2 recipe search) folded in (2026-10-24)

Sources: `TORUS_LRSWEEP_RESULTS.md` (registered reading COST AT BEST RATES; `TORUS_LRSWEEP_ANALYSIS.txt`,
`TORUS_LRSWEEP_VERDICTS.json`) and `RANKRECIPE_RESULTS.md` (registered reading RANK D+1 NEEDED SURVIVES A THREE-RECIPE
SEARCH; `RANKRECIPE_ANALYSIS.txt`, `RANKRECIPE_VERDICTS.json`). Changes: abstract (sign sentence: best rate on the ring and
the torus; rank sentence: 0 of 8 at each of three recipes); contribution 3 (both); Sec. 7.2 new paragraph "Nor does a recipe
search" (0/24; best recipe lr 3e-3 0.910 vs 1.000, decision p 0.0002 both routes; "rank D + 1 with weight decay ... rank D
fails at the best of three recipes"); Sec. 7.3 best-rate paragraph (torus |.| 3e-3 0/8 0.890 vs signed 1e-3 8/8 1.000,
selection-aware p 0.0002 / 0.0095; signed 8, 0, 0 of 8 on the torus grid, all descending above 1e-3) and a Table 6 row;
Limitations ("swept on the ring only" replaced: ring four rates, torus three, decay coupled, no torus decay control, 5/8 |.|
descending; rank bullet: three recipes, both higher-decay, 12/24 descending, the stated claim form). Appendix: G9 and H12
(new, with tables and per-seed rows), two A3 rows, an H1 pointer to H12 [all numbers R: re-read from the analysis files;
the G-4 cell means re-derived from the checkpoints]. Claim-map compliance: no "at any learning rate", no "no recipe rescues
rank 2". Draft 6: abstract, Sec. 4.1 and Limitations carry the torus best-rate result; its PDF rebuilt.

## G-3 block A2 (SSM at lr 1e-3) and FACTORIAL_N8 (torus depth, fresh seeds) folded in (2026-10-25)

Sources: `G3_SSM_A2_RESULTS.md` (registered: SIGN COST IN THE SSM at 1e-3 for |.| and softplus; exp NO SIGN COST;
`G3_SSM_A2_ANALYSIS.txt`, `G3_SSM_A2_VERDICTS.json`) and `FACTORIAL_N8_RESULTS.md` (registered: REPLICATES;
`FACTORIAL_N8_ANALYSIS.txt`, `FACTORIAL_N8_VERDICTS.json`). Changes: abstract (sign sentence: "non-negative steps" -> "|.|
steps"; SSM clause now |.| and softplus at two rates, exp 6 of 8 against 7 of 8); Sec. 1 failure list ("a |.| or softplus
step"); contributions 2 (torus rate replicated, n = 8, four layers at the retrace floor) and 3 (|.| step; SSM softplus pays,
exp does not at 1e-3, "not to non-negativity as such"); Fig. 1 and Fig. 3 captions (replication pointer); Sec. 6 torus
paragraph (fresh 0.999 / 0.729 / 0.830 / 0.859; pooled 0.999 / 0.724 / 0.845 / 0.862; 4L - retrace -0.016, pooled -0.000;
loop +0.014; short gap 0.913 vs 0.911; long range 0.41 vs 0.51); Sec. 7.3 attention exp sentence scoped "In attention",
Table 6 SSM 1e-3 row, SSM paragraph extended (A2 numbers, exp mechanism, what is not separated); Discussion ("|.| steps"),
design guidance (|.| and softplus paid everywhere; exp escaped once), Sec. 9 integrator sentence, related work;
Limitations (torus n = 8; SSM bullet rewritten). Appendix: A3 rows, C2 fresh rows, F pointer and F7 (new), H10 pointers
(A2 run; cross-rate joint superseded), H13 (new, tables + per seed), E14 E8.6 check. Numbers [all R: re-read from the
analysis files and eval.json; exp channel counts from `docs/audits/2026-10-25/g3_a2_exp_mech_out.txt`, analysis].
Claim-map compliance: no "every non-negative parameterisation pays", no "exp succeeds because of its start", no exchange
rate beyond "more than four", pooled numbers labelled descriptive. Figures not regenerated (Figs. 1-4 still plot the n = 4
FACTORIAL models; captions point to F7). Draft 6: title, Sec. 4.1, Sec. 5, Limitations; its PDF rebuilt.

## IMAGE_TORUS (pixel observations) folded in (2026-10-26)

Source: `IMAGE_TORUS_RESULTS.md` (registered: P1 PATH WINS IN PIXELS, partly pilot-informed; P2 THE OBSERVATION STEP COSTS
IN PIXELS and G GAIN 1 RESCUES MAPWM, both PILOT-INFORMED, NOT BLIND; P1' UNMEASURED; `IMAGE_TORUS_ANALYSIS.txt`,
`IMAGE_TORUS_VERDICTS.json`). Changes: abstract (last sentence: class images learned to the perception ceiling by an
action-driven path layer, 0.995 vs 0.701; an image-driven step stops training on most seeds); Sec. 1 ("and shown in
pixels"); contribution 5 ("Maps from words and pixels", two sentences); Sec. 8 new subsection "Maps from pixels" with
Table 8 (0.995 8/8 at ceiling vs 0.701; long-gap / wrap-only 0.994 vs retrace 0.51; pixel cost 0.005; probe 0.985 vs 0.965;
image-driven step 0.668, 5/8 memoryless, 3/8 at 0.92-0.96; gain 1 0.994 8/8; S2b +0.287, p 0.039); Sec. 9 What and where
(one sentence) and design guidance (one clause); Limitations (pixel bullet: class images on a grid, not first-person,
pilot-informed P2/G, G a total effect). Appendix: A3 row, K4 (new: task, disclosure, contrasts, per-seed table, end states,
strata / OOD / probe / context-destruction table) [all numbers R: re-read from the analysis files, eval_image.json and the
training logs]. Claim-map compliance: no "leak" for this batch (0 NOISY seeds), no dose-response in the gain, nothing about
first-person or 3D vision, disclosures stated.

## Critical review 2026-10-26, Part B, applied (2026-10-26)

Source: `docs/audits/2026-10-26/CRITICAL_REVIEW.md` Part B. Rule-28 end-to-end read of the main text before and after the
cut. Only `docs/paper/` files changed (MAIN_DRAFT2.md, MAIN_APPENDIX2.md, this file, the three draft-2 PDFs). Results files,
CLAUDE.md and code untouched, so the review items addressed to them are open (listed at the end).

**Length.** Main file 18,000 -> 11,570 words (`wc -w`, references and header note included; 17,231 -> 10,721 without the
reference list). Per section (whitespace tokens, captions and tables included): abstract 321, 1 757, 2 344, 3 669, 4 1,342,
5 291, 6 782, 7 41, 7.1 206, 7.2 1,066, 7.3 1,775, 7.4 412, 8 717 + pixels 348, 9 905, 10 587, references 849. Figures 11
-> 10, tables 8 -> 7 in the main text. Appendix 35,444 -> 36,209 (moved material plus two scope paragraphs).

**Finding by finding.**
- B-B1 (GAUGE missing from the main text). Added: Lemma 1 now says it is a statement about representation and that its
  learnability reading was tested and not confirmed; Table 1 row "Lemma 1 read as learnability: registered test
  UNMEASURED (+1/+2: |.| 2/8, no better than +1/-1)"; Sec. 7.3 paragraph "Losing the through-zero route does not account
  for the difficulty (registered)" with DRAFT6's numbers (index RoPE 0.649 0/8; sign-coherent 0.92 vs 0.01; |.| 2/8 vs 1/8
  on +1/-1, Fisher p 1.00; signed 5/8; unmeasured because +1/+2 is harder for signed too) and DRAFT6's closing sentence;
  +1/+2 row in the new Table 5; Sec. 9 "losing the through-zero route does not by itself account for the cost (+1/+2)";
  Limitations bullet. App. H4's "There is no evidence of a sign cost without an inverse pair" reworded to "no evidence
  either way ... |.| solves +1/+2 no more often than +1/-1".
- B-B2 (sign framing). Sec. 7.3 retitled "How the step is parameterised, and where its angles start"; "barrier
  parameterisations" defined in Sec. 3 (DRAFT6's term); abstract rewritten: |s| and softplus(s) from small starts pay;
  where the angles start relative to pi matters (|.| past pi learns the ring by wrapping; exp no cost in the SSM at one
  rate "consistent with its past-pi start", not "which starts many channels past pi"). New Table 5 (start x move set x
  model: MECHANISM, TORUS_START, GAUGE, G3 A2) carries the account; the text marks the start lever as registered and
  causal on the ring, not transferring to the torus, and only consistent (analysis) for exp. "exp has not been run in
  attention on the torus" stated in Sec. 7.3 and Limitations (the batch to test it is being built elsewhere; not
  mentioned in the paper). "sign gap" / "the sign of the phase step" removed from the main text.
- B-M2 (IMAGE_TORUS primary swap). Sec. 8 pixels: "The batch's registered primary compared the default MapWM ... reads
  unmeasured (0.668 against 0.701); after a pilot stop rule fired the path arm was changed to the action-driven step
  (Amendment 2; partly pilot-informed)." Same in Limitations.
- B-M3 ("the what-to-where rule at full strength"). Removed. Design guidance now: "not the token-level leak (no
  per-instance offsets) but a collapse of encoder and phase together"; the gain is called "a total effect on the step
  input and the residual stream at once", with the chain "since the action-driven model is fine at token scale, the cost
  belongs to the step".
- B-M4 (pixels oversold). Subsection retitled "Pixel observations" (one paragraph + Table 7): "Its phase never sees a pixel:
  what this shows is that a content-addressed CNN with an action-driven phase reaches the perception ceiling, and that the
  revisit loss alone trains the encoder (probe 0.985 vs 0.980; RoPE 0.965)". Pixels dropped from contribution 5 and
  reduced to "the result survives a pixel front end" in the abstract; the P2 collapse moved to design guidance and
  Limitations; "Maps from pixels" and "path integration survives pixels" headings gone; the What-and-where pixel sentence cut.
- B-M5 (RANKRECIPE "search favours rank 2"). Replaced in Sec. 7.2 with "The selection cannot hurt rank 2, but the search is
  limited: neither alternative is a recipe at which rank 4 is known to learn (at 3e-3 the related signed rank-4
  observation-step-free model solves 0/8 on this torus within 900 epochs, App. H12; the higher decay shrinks the angles
  most and leaves the most rank-2 runs stalled). The result reads: rank 2 fails at each of three recipes." Abstract and
  contribution 3: "at three recipes". Limitations bullet as the review asks. App. G9: "Scope of the search" paragraph.
  Wording note: the review's "the other raises decay, the direction RANKWD shows rank 4 needs less of" is not what RANKWD
  shows (rank 4 needs decay, 8/8 vs 1/8 at 0); not used. The registered label in G9 is kept verbatim.
- B-M6 (survives / needed). "Rank 4 in turn needs weight decay" -> "Within 900 epochs rank 4, in turn, learned only with
  weight decay"; "Twice the budget does not rescue rank 2" + "Nor does a recipe search" -> "Rank 2 still fails at twice
  the budget and at three recipes"; the cost-at-best-rates heading now says "within 900 epochs"; "Why the extra rank
  helps" -> "Why the extra rank helps is open". Headings in the results files and CLAUDE.md rows not edited (out of scope).
- B-M7 (citations). Larson 2026 (2609.12259) in Sec. 7.2 (reconciled as in BIG_PICTURE: agree on capacity, rank-D
  solution exists and is held; differ on search: sparse revisit loss and shared readout need one more dimension) and Sec.
  9; Howe 2026 (2609.18966) in Sec. 9 (pre-registered, exact-init solutions pulled off by Adam through an additive pathway;
  deleting it lets training find the automaton, as forbidding the clock does here); Heo et al. 2026 (2603.05573) in Sec. 6
  (error falls exponentially with depth: expressivity, not what training finds) and Sec. 9. Titles and authors checked on
  arXiv 2026-10-26; references added.
- B-M8 (length). Done as above. Props. 5-6, Conjecture 1 and Remark 2 condensed into one "Further statements" paragraph
  (numbering kept so the appendix cross-references hold); Table 4 and Fig. 5 (clock end state) moved to App. G3 as Table
  G3a and Fig. G2; Table 3 (revisit classes) dropped from the main text (identical table already in App. F1); SIGNSEP to
  three sentences; exp to five; the step-basin / straight-path / cone analysis paragraph of old Sec. 7.3 cut to its App.
  H7 home; Sec. 7.2 clock-exit analysis cut to two sentences (App. G4); Sec. 8 text-world asides and clock paragraphs
  shortened (App. J); captions shortened. Every registered number that carries a claim is kept.
- B-M1, B-M9 (triage, LSTM/GRU baseline). GPU items, not in scope; Limitations now says "There is no LSTM or GRU baseline".
- MINORs applied: abstract exp sentence aligned with Limitations; signed anchor at 1e-3 "reached exact codes on only four
  seeds" beside the exp sentence (Sec. 7.3, Limitations); "diagonal complex SSM without decay or gate" at first mention;
  SIGNSEP in Sec. 9 now "half-read", with F's 4/8 in Sec. 7.3; "more than four" kept. MINORs on CLAUDE.md, project_state,
  IMAGE_TORUS_RESULTS line 196 and RANKRECIPE's Checks section are outside docs/paper and left open.

**Numbers.** No new number; every number in the revised text was already in the main text or appendix [R], except the
Larson / Howe / Heo descriptions (arXiv abstracts, 2026-10-26) and "ring best-rate |.| 2/8" now stated in the abstract and
contribution 3 as "at most 2/8 on the ring" (the old abstract's "0 of 8 against 8 of 8 ... on the ring and on the torus"
was wrong for the ring's best-rate cell, 2/8; fixed).

**Cross-references.** Main-text figures renumbered (old 6-11 -> 5-10), tables (old 5 -> 3, 6 -> 4, 7 -> 6, 8 -> 7; new
Table 5). Appendix references updated: C9 and O ("Table 6"), E7.1' and O ("Table G3a"), O's F1 row.

## Word counts

(Superseded by the 2026-10-26 counts above. Counted before the audit was applied. The audit edits added about 830 whitespace tokens to the file (12,934 -> 13,765), mostly in Secs. 4, 6, 7.1, 7.2 and 8 and Table 7; the per-section counts below are not recounted.)

Counted per paragraph (whitespace tokens). "Body" excludes figure captions, table captions and table cells.

| section | body | captions + tables |
|---|---|---|
| Title | 19 | -- |
| Abstract | 300 | -- |
| 1 Introduction | 905 | 132 (Fig. 1) |
| 2 Background | 486 | -- |
| 3 Setup and how map use is measured | 711 | 113 (Fig. 2) |
| 4 Theory | 1,345 | 233 (Table 1) |
| 5 One path layer solves the long-walk map | 312 | 90 (Table 2) |
| 6 What index depth buys | 806 | 340 (Fig. 3, Table 3; + Fig. 4, not counted) |
| 7 opening | 67 | -- |
| 7.1 The clock and the signature | 182 | 160 (Table 4; + Fig. 5, not counted) |
| 7.2 Per-head rank | 639 | 416 (Table 5, Figs. 6-7) |
| 7.3 The sign of the phase step | 1,242 | 509 (Table 6, Figs. 8-9) |
| 7.4 Readers cannot repair a code | 536 | -- (+ Fig. 10, not counted) |
| 8 Maps from words | 838 | 244 (Table 7, Fig. 11) |
| 9 Discussion and related work | 1,152 | -- |
| 10 Limitations | 227 | 19 |
| **Sections 1-10 + abstract** | **about 9,750** | **2,256** |
| References | 769 | -- |

Target was ~8-10k words of main text including a ~1-page theory section: met (theory section 1,345 + table 233). Appendix
24,781 words (A 1,982; B 624; C 2,812; D 886; E 6,123; F 1,929; G 1,951; H 2,788; I 764; J 1,668; K 624; L 890; M 879;
N 218; O 476).

## Every number and its source

R = registered decision or declared secondary; A = analysis (post hoc, eval-only); A-audited = analysis from
`docs/paper/expansion/scripts/` (or `docs/audits/2026-10-20/`) cleared or corrected by the analysis-script audit (formerly
"A-pending"); P = theory (proved); C = conjecture; D = descriptive / structural.

**Abstract and Sec. 1** (contributions repeat numbers below).
- retrace 0.843 (T = 1024), 0.934 (T = 128) [A]: `docs/audits/2026-10-16/sign_floors_out.txt`, `paper2x2_T128_floors_out.txt`.
- path 0.998 8/8 vs RoPE 0.731 0/8, p 0.0002 [R]: `SIGN_MATCHED_RESULTS.md`.
- Dyck about three, ring about three, torus more than four [R]: `DYCK_MDEPTH_RESULTS.md`, `CANCEL_RESULTS.md` (ring: declared secondary), `FACTORIAL_RESULTS.md`.
- rank 2 at most 2/8 vs rank 4 8/8 [R]: `RANK_MI_RESULTS.md`, `RANK_SEP_RESULTS.md`; twice the budget 0/8 vs 7/8 [R]: `LOOP_RANK_E1800_P1_RESULTS.md`.
- non-negative 0/8 vs 8/8 [R]: `SIGN_MATCHED_RESULTS.md`; best of four rates on a ring [R]: `LRSWEEP_RESULTS.md`.
- non-negative readout with signed increment 16/16 [R]: `SIGNROUTE_RESULTS.md`.
- frozen move latents 1/8 vs 7/8 [R]: `ACCOUNTS_RESULTS.md` T2.
- text world 0.979 vs 0.505 (fresh seeds) [R]: `TEXTWORLD_RESULTS.md` CORRECTED block, `docs/audits/2026-09-27/fresh_seeds.txt`.
- loop / depth +0.079 / +0.096 [R]: `LOOP_RANK_RESULTS.md`; DELTA_FREE +0.122 [R]: `DELTA_FREE_RESULTS.md`.
- Non-retrace stratum 0.956 / 0.978 fresh vs 0.411, constant 0.506 [A-audited]: `language/tw_strata_200walks_out.txt`
  (path re-scored all seeds 0.9555, fresh 0.9780; RoPE 2L 0.4109; best constant 0.5058; B1 / M-LG1).

**Fig. 1** (`figs/fig_summary_out.txt`): (a) FACTORIAL 0.719 / 0.860 / 0.864 / 1.000, floors 0.509 / 0.844 [R; floors A];
(b) rank 2 0 / 7 (RANK_MI; RANK_PROJ), |.| ring 0 / 8 (WRAP_EXISTENCE), |.| torus 0 / 8 re-found (GENERALITY) [R]; (c)
strata 1.000 / 0.995 / 0.982; 0.950 / 0.671 / 0.557; 0.870 / 0.630 / 0.515; 0.784 / 0.462 / 0.471 [R secondaries].

**Sec. 2.** MapFormer Table 2, Figs. 3d / 9b: `papers/txt/mapformer.txt` (as in draft 1 notes); reproduction 0.989 / 0.987
[D]: `PAPER_TASK_ACCURACY.md`; Sarrof, Grazzi, Mamba-3: `review_draft5/PRIOR_ART.md`; Howard cases, Alipour, GridPE,
Mochizuki-Freeman, Kabir: `expansion/LANGUAGE.md` 4.1, `docs/lit/LIT_HOWARD_TRANSFORMER.md`.

**Sec. 3.** Architecture and recipe as draft 1 (`APPENDIX_draft6.md` A1-A4); run classes `stats_core.classify_run`.

**Sec. 4.** Prop. 1 [P]: DEPTH 3.2. Remark 1 [cited]: DEPTH 3.3 (Liu et al. Lemma 6; Yao et al. via Liu App. A.5; Hahn and
Rofin Cor. 5 / Thm. 6); 1/2048, 4nT, 64 x 2048, 512: DEPTH 3.3. Clock/map, Prop. 2(a) [P]: SEARCH S5; Prop. 2(b) [P]:
LANGUAGE T1. Prop. 3 [P]: v3 P1(a), P1(f); (g) exp new here (proof in E2, not built). Lemma 1 [P]: v3 T1. Prop. 4 [P]:
SEARCH S1-S3, S6; RANK Prop. 1. Prop. 5 [P under A1-A3] and Conjecture 1 [C]: RANK Props. 2-3. Remark 2 [definitional]:
SEARCH S8. Prop. 6 [P]: RECURRENCE T1-T2, MAPPOPE Theory 2. Prop. 7 [P given delta = 0]: MAPPOPE Prop. 1, Cor. 1c. Prop. 8
[P]: v3 P6. Table 1: "1.4e-6 rad" from `WRAP_EXISTENCE_RESULTS.md` / `GENERALITY_RESULTS.md`.

**Sec. 5 and Table 2.** As draft 1 (sources there): T = 128 floors 0.507 / 0.598 / 0.934, wrap 0.000 [A]; T = 1024 0.506 /
0.575 / 0.843, wrap 0.079 [A]; 0.805 / 0.971 / 1.000, +0.243 [R] `PAPER2X2_RESULTS.md`; 0.731 / 0.894 / 0.998 [R];
0.998 +/- 0.004 vs 0.731 +/- 0.008, +0.267 [R]; 0.72-0.74 below retrace [A]; 1.000 vs 0.722 (GENERALITY) and 0.719
(FACTORIAL) [R]; strata 1.000 / 0.995 / 0.982, 0.784 / 0.462 / 0.471, floors 0.506 / 0.500 / 0.507, retrace 0.911 [R / A]:
`figs/fig_strata_modes_out.txt`.

**Sec. 6.** Dyck 0.996, 0.648 / 0.889 / 0.964 / 0.982, floor 0.594, +0.353 / +0.130 / +0.045 / +0.024, p 0.0078, 3x
0.997-0.998, mixture +0.110, base 32 vs 10000 [R]: `DYCK_MDEPTH_RESULTS.md`. Ring fresh seeds 0.710 / 0.949 / 0.998;
0.677 / 0.962 / 0.998; 0.824 / 0.987 / 0.999; within 0.002; p_plus 1.0 1.000 [R secondary]: `fresh_seeds.txt`,
`CANCEL_RESULTS.md`. Torus 1.000 vs 0.719 / 0.860 / 0.864, p 0.029, +0.140 / +0.145, 203,925 vs 798,741; strata 0.769 ->
0.943 / 0.938, 0.458 / 0.507, 0.457 / 0.509, floors 0.515 / 0.517 [R; floors A]: `FACTORIAL_RESULTS.md`,
`figs/fig_factorial_strata_out.txt`. Fig. 3 caption 0.844: `figs/fig_exchange_rate_out.txt` (C-12 applied). "Why the rate
differs": ring 1L retrace 0.68-0.80, lap 0.99-1.00, 3L 0.997 / 0.993 / 0.972 [A-audited]: `depth/out/ring_depth_out.txt`;
neither-class floors 0.491-0.493 / 0.502 and 1L 0.487 / 0.592 [A-audited]: `depth/out/ring_class_floors_out.txt`,
`ring_depth_out.txt`; window code R^2 0.72, older displacement 0.01 [A-audited]: `depth/out/ring_probe_window_out.txt`
(M-DP1; the old R^2 0.69 vs 0.18 is no longer quoted); torus R^2 <= 0.21 [A-audited]: `depth/out/torus_probe_out.txt`;
0.996 / 1.000, attention 0.55-0.58, 0.00-0.01, 0.81, 0.911 and 0.943 decomposition [A-audited]:
`depth/out/torus_depth_out.txt`; 0.874, 0.493 (seed means), 98.5% and median 37 [A-audited]:
`depth/out/torus_segments_out.txt` (`summarize_torus.py --segments`, M-DP4). Table 3 [A-audited]: `torus_depth_out.txt`. LOOP_RANK strata 0.671 -> 0.927-0.990,
0.557 -> 0.747-0.887 [R secondary]: `LOOP_RANK_STRATA.json` via RECURRENCE Sec. 1.

**Sec. 7.1 and Table 4** [A-audited]: `sign/clock_table_out.txt` (one definition, B3 / M-SG1; run-level arm means: MECH SO
0.018, SO_absmag 0.015, GAUGE SO_m 0.016; AO 1.471, AO_absmag 1.571, AO_m 1.559; SO_absmag_fz 1.181 (solved 0.058);
AO_high 0.317; TORUS_START SO4 0.001, AO4 1.248, SO4_high solved 1.390 / 1.777 with near-pi usage 0.43 / 0.99; rank arms
unsolved 0.595 / 0.694 / 0.541, medians 0.579 / 0.792 / 0.575; solved rank-2 heads 0.03-0.07 / 0.54, 0.68; rank 3 / 4
solved 0.043 / 0.049 / 0.051; solved heads <= 0.138); 0.82 usage on cancelling channels: `sign/sign_clock_out.txt`;
collapsed runs: `rank/sweep_final_out.txt` (basin COLLAPSE, indep 0.00-0.03). Strata [R secondary]:
`fig_strata_modes_out.txt`. Usage-weighted cancelling 70-93% vs 2-10% [A]: `FIGS_NOTES.md` (fig_wrap_share table, App. H6).

**Sec. 7.2 and Table 5.** Six arms, SOLVED, accuracy, p 0.0070, +0.102 p 0.027 Holm 0.054 (C-11), 478-532, 1800 epochs
0/8 (0.908) vs 7/8, Fisher 0.0014, +0.086 p 0.0003, 0.9955 / 0.981 / 7/8 / 7/8 / 1/8 p 0.0009, 2D 1/8 vs 8/8, 3D 1/8,
4/8, 8/8, ring 7/8 vs 8/8 [R]: as draft 1; R1DF 24 vs 58 epochs [R]: `RANK_TESTS_RESULTS.md`. SCORE_RANK 3/12 vs 3/12, 8/8
[R]: `SCORE_RANK_RESULTS.md`. "Why the extra rank helps" [A-audited]: `rank/summarise_final_out.txt` (off-plane 0.96-1.00 in
9 of 10 sets, readgain 0.09-0.27; random-direction median 0.40-0.71 per the audit's Beta computation), `rank/clock_decomp_out.txt`
(a, o 0.017-0.037, cosine -0.79 to -0.999, residual 0.0007-0.025; escape F |c| 14-85x, residual 0.002-0.005, share
0.91-1.00), `rank/c1_runlevel_once_out.txt` (50/55 vs 25.1), `rank/escape_start_out.txt` (16/16 heads; the rank-sum p 0.018
now in App. G4 only). G-1 (RANKCLOCK) paragraph and the primary in "Why the extra rank helps" [R]: see "G-1 (RANKCLOCK)
folded in" above.

**Sec. 7.3 and Table 6.** As draft 1 (SIGN_MATCHED, RING_DIAL, WRAP_EXISTENCE, GENERALITY, LRSWEEP, ROBUST, MECHANISM,
ACCOUNTS) [R]. Exp / softplus 1/8-3/8 [R breadth cells, fixed rates]: `SIGNROUTE_RESULTS.md`. Routes paragraph
[A-audited]: `sign/latent_reach_out.txt` (1.000; 0.339 / 0.561), `sign/path_profiles_out.txt` (0.60-1.92 vs 0.04-0.48;
0.30 -> 0.72 vs 0.30), `sign/latent_geometry_out.txt` (cone boundary 0.30 / 0.36 / 0.25; the 0.056 initialisation value
recomputed by the audit), `sign/basin_summary_out.txt` (frozen-content qualification; exploratory, M-SG6). Hybrid 8/8 at both rates, 1.000, |.| 0/8 and 2/8, Fisher 0.0002 / 0.007, u-ratio 0.043 /
0.002, cosine -0.996 / -1.000, abs_tied 0/16 [R; end states descriptive in the file]: `SIGNROUTE_RESULTS.md`. Clock / map
1.18 for frozen latents [A-audited]: `sign/clock_table_out.txt`. CARoPE 0.667 vs 0.722 p 0.040; CoPE 0.887 p 0.0002, +0.157; line 1/16 [R]:
`GENERALITY_RESULTS.md`, `ACCOUNTS_RESULTS.md`. Selective RoPE +0.236, 0.961 -> 0.507, 0/8 vs 8/8, 0.961 vs 1.000 p 0.0002,
all descending; observation / move step 0.86-0.96 vs 0.027; state-tracking 0.769 0/8, negative-mass share 0.00 [R;
readouts declared secondaries, descriptive]: `SROPE_TORUS_RESULTS.md`.

**Sec. 7.4.** LOOP_RANK 0.894 / 0.973 / 0.990, p 0.0034 / 0.0012, 2/8 / 5/8, rank 4 0.998 [R]: `LOOP_RANK_RESULTS.md`.
Reader 0.866 / 0.894, 1.000 / 0.998, 0.32-0.44, 0.893 / 0.897, p 0.0002, 0.381, 1/16 2D codes, half 1D [A-audited]:
`recurrence/out/SUMMARY.txt`, `GEOMETRY.txt` (RECURRENCE 5.2-5.4); MDE 0.074 / 0.080 [A-audited]: `recurrence/out/READER_MDE.txt`;
7 of 8 seeds below the floor: `recurrence/out/pass_strata_L.json`. PAPER2X2 0.999 / 0.971 / 0.679 / 0.805,
+0.154 [R]. Probes 0.995 vs 0.70, 0% vs 53-100%, 0.110 of 0.126 [A-audited]: `mappope/probe_stats_out.txt`,
`kernel_probe_paper2x2_out.txt`. DELTA_FREE +0.122 p 0.0078, +0.180, 50-100% with peaks at 3, 4, 7, 8, -0.018 p 0.31,
0.987 -> 0.62 [R; mechanism readouts declared secondaries]: `DELTA_FREE_RESULTS.md`. T = 1024 C 0.617 -> 0.960, C0 +0.06, C0
0.80 / 0.91 and 0.40 / 0.48, rank 4 0.99 and 0.97 [A-audited; 40 walks per seed; per-stratum head, B2]:
`mappope/probe_stats_hstratum_out.txt`, `kernel_probe_score_rank_hstratum_out.txt`.

**Sec. 8 and Table 7.** Retrace 0.934, constant 0.505, reversal-copy 0.597 [A-audited / R]: `language/tw_retrace_floor_out.txt`,
`TEXTWORLD_RESULTS.md`. 0.979 +/- 0.048, 0.505, +0.474 p 0.0022, 0.793 [R]: `fresh_seeds.txt`. 0.993 re-scored [R, registered
re-score]: `docs/audits/2026-10-04/DROPOUT_RESCORE.md`. Table 7, 13.4%, 0.956 / 0.978 / 0.934, 0.411, 0.454, 0.506; 6.6%,
0.925 / 0.967, 0.129, 0.004, 0.088; 0.927 / 0.926 [A-audited]: `language/tw_strata_200walks_out.txt` (all 200 walks;
B1 / M-LG1). Cancel 0.002-0.012, synonyms 1.000, 0.39-0.51, 31-38 channels, cosine +0.999 to +1.000 [R
secondaries]: `TEXTWORLD_RESULTS.md`, `fig_textworld_out.txt`. Step table 0.004-0.03, 0.04-0.16 [A-audited]:
`language/tw_step_account_out.txt`. DirOnly 0.970-0.974 [R, eval mode]: `TW_NORMSTEP_RESULTS.md`. Collapse 0.989-1.000 ->
0.952-0.975, 71-100%; lift 0.962-0.981 -> 0.999-1.000 [A-audited]: `language/tw_attention_asides_out.txt`,
`tw_attention_asides_dironly_out.txt`. 10 of 24, 26-38, 0.06, 0.007, 0.98-1.00, 79-99%, 0.009, 0.03-0.65, 0.09-0.26,
Spearman +0.905 [A-audited]: `tw_step_account_out.txt`, `tw_clock_unit_out.txt`, `usage_tick_out.txt`,
`tw_attention_asides_out.txt`, `tw_common_used_out.txt`; its p 0.0046: `scripts/whatwhere/whatwhere_stats_out.txt`. Leak 16/16 vs 0/8, 8/8 vs 0/8 [R]: `LEAK_RESULTS.md`,
`GAIN_PHASE_RESULTS.md`.

**Sec. 9.** 0.9955, 7/8, 8/8, 0-2/8, 0/8, 16/16 [R]. EM 1.000, 0.578 -> 0.928 (budget contrast not registered) [R pieces]:
`WARM_RESULTS.md`, `SPREAD_RESULTS.md`, `SEARCH_RESULTS.md`. What/where 0.998, 0.966 (-0.032), 92.5%, 0.113, 0.25-0.31,
Spearman 0.84 [A-audited]: `docs/audits/2026-10-20/WHAT_WHERE_T1024.md`; sign-flip p 0.0078, t p 0.0116, SS(P)/SS(S)
0.925, Spearman 0.838 p 5.0e-5: `scripts/whatwhere/whatwhere_stats_out.txt`. rho 0.064-0.079, 0.06-0.23, 21.8-22.1, 9.9-12.3, 4.5-6.6
[A-audited]: `language/crossover_usage_out.txt`, `usage_tick_out.txt`. 1.5053 vs 1.5050 [pilot]:
`TINYSTORIES_PILOT_RESULTS.md`; +0.0056 bpc, 3/3 [R, n = 3]: `CODE_FULLVAL_RESULTS.md`; R^2 0.994-0.999 [A]:
`docs/audits/2026-10-12/alpha_language_out.txt`. Egocentric +0.050 -> +0.488 (n = 8, 16-epoch recipe, index arm on the
0.508 floor; not registered): `KNOB_SWEEP_n8.md`.

**Sec. 10.** Quarter-size gap and descending |.| runs [R]: `ROBUST_RESULTS.md`; Selective RoPE all descending [R]:
`SROPE_TORUS_RESULTS.md`; exp / softplus initial steps +1.051 / +0.706 [R descriptive]: `SIGNROUTE_RESULTS.md`. Clock-slot budget, shared rank only, wd 0 1/8
vs 8/8 (post hoc) [R / post hoc]: `RANKCLOCK_RESULTS.md`.

**Appendix.** Every number is copied with its source named inline. New appendix material by source: A3 (prereg / results
headers of each batch); A5 `KNOB_SWEEP_n8.md`; B additions (SCORE_RANK, SROPE_TORUS, DELTA_FREE results files and their
floors outputs; `language/tw_retrace_floor_out.txt`); C10, H8, H9 `SIGNROUTE_RESULTS.md`; E (v3, SEARCH S1-S8, RANK Props.
1-3, DEPTH 3.2-3.4, RECURRENCE T1-T2, MAPPOPE Prop. 1 / Theory 2, LANGUAGE T1 / 3.1-3.3); F1-F6 DEPTH 4.1-4.6; G2
`SCORE_RANK_RESULTS.md`; G3 `sign/rank_clock_out.txt`; G4-G6 RANK C1-C9; H4 GAUGE + SEARCH S4; H7 SEARCH 3.1-3.4 and
`sign_clock_out.txt`; I `SROPE_TORUS_RESULTS.md`; J1-J6 LANGUAGE 2.1-2.6, 3.3-3.4; K1 `WHAT_WHERE_T1024.md`; K3 LEAK,
GAIN_PHASE; L1 PAPER2X2, MAPPOPE_PAIR, GAIN_GRAIN, SCORE_RANK, DELTA_FREE; L2 MAPPOPE probes; M RECURRENCE 1, 5.2-5.6; N
PAPER_LEVEL 3.2 checked against WARM / NOLEAK / RECENCY_EM / SEARCH / SPREAD results files; O the reports' reproduce
sections.

## Deviations from the plan, and why

1. **Figure 1 is `fig_summary`** (the user's instruction). The plan's Fig. 1 (exchange rate) is Fig. 3; plan Figs. 3-7
   are Figs. 6-9 and 11 (plan Fig. 7 `fig_textworld` is Fig. 11). Plan Figs. 8 (segments), 9 (clock end state) and 11
   (kernel peak) were built on 2026-10-21 from the audit-corrected generators and are draft Figs. 4, 5 and 10 ("Figures
   added" below); Tables 3 and 4 and App. L2 still carry the same numbers. Not built: T1 schematic, 10 loop reader,
   7d-e, G2, H7, J2; their content is carried by Table 7 (text-world strata), App. M3 (loop reader), App. J1 / J5 (for
   7d-e), App. G6 (for G2), App. H7 (for H7), App. J4 (for J2).
2. **Theory status table is Table 1** (plan: "Table 4" at the end of Sec. 4), so that tables are numbered in order of
   first mention. Table 1 cites sections, not later tables or figures.
3. **The plan's opening sentence for Sec. 4** ("This account was developed after the registered batches ...") moved to
   Limitations, per the user's style instruction (no process history; scope stated once).
4. **Proposition numbering.** Plan 4.1 + 4.2 -> Prop. 1; 4.3 -> Remark 1 (cited); 4.4 -> Prop. 2 (signature + tick rule);
   4.5 -> Prop. 3 + Lemma 1; 4.6 -> Prop. 4; 4.7 -> Prop. 5 + Conjecture 1; 4.8 -> Remark 2; 4.9 -> Prop. 6; 4.10 -> Prop. 7;
   4.11 -> Remark 3 (App. E13, used in Sec. 9); CARoPE (v3 P6) -> Prop. 8 (stated in Sec. 7.3). App. E's header maps
   each to its E-label.
5. **CPU-3 (clock-metric harmonisation)**, done in part by applying the audit (B3, M-RK1): Table 4, G3 and H7 now use
   one channel ratio (the symmetric clock / map ratio of Sec. 4 / E7.1', `sign/clock_table.py`). RANK's latent quantities
   (off-plane share, readgain) stay a separate, latent metric in Sec. 7.2 and App. G4, now with the a + o decomposition
   (`rank/clock_decomp.py`); they are not recomputed with per-pair clocks (the audit shows the per-pair clock omits the
   observation step and so overstates the drift). X3's clean-head / other-head split is applied (Table 4, G3).
6. **CPU-4 (rank-3 pair clocks) not done**: per X14 neither account of rank 3's 6/8 appears in the main text.
7. **CPU-5, CPU-6, CPU-8 not done**: the MAPPOPE T = 1024 probe remains at 40 walks per seed (stated in App. L2); no
   polarisation check; no loss-by-stratum for LOOP_RANK.
8. **CPU-7 (exp-step construction)**: stated and proved as Proposition 3(g) / App. E2(g); not built in a model (stated).
9. **G-6 (delta freedom) is now registered (DELTA_FREE).** The plan's predicted "sign reversal of the interaction" is not
   used; the text uses the licensed sentence (Amendment 1 MAJOR 5): the pinned offset fixes the kernel peak, helps a path
   phase, hurts an index phase. The cross-batch interaction (-0.140 vs +0.154) appears only in App. L1, as the file gives it.
10. **G-2 ring part ran as SIGNROUTE.** Its Amendment 1 withdrew the "decisive" label: the hybrid removes both the missing
    collective route and the far-side cancelling point. The main text says exactly that ("a non-negative increment,
    which removes both the collective route and the near-side cancelling point, is what costs") and does not claim to
    separate S3 from the wrap barrier. The exp / softplus breadth cells are reported as fixed-rate results with their
    initialisation (a constant positive step), not as best-rate results.
11. **Selective RoPE added** (7.3, App. I, Limitations) with the claim map's wording ("does not learn the torus map within
    900 epochs in its published configurations"; never "cannot learn maps").
12. **What/where at T = 1024 (CPU-1) is complete**; the final file's numbers (identity removed 0.966, -0.032, p 0.012;
    shared kernel x gain 0.998; non-negative and index costs 0.25-0.31; Spearman 0.84) replace the plan's partial ones.
13. **Abstract.** The plan's abstract with: "In line with this" (not "Accordingly"); the SIGNROUTE clause added ("vanishes
    when only the readout, not the increment, is non-negative", ring); "learns which words move it" dropped for length.
    "On a ring" kept for the best-rate claim (C-18). 300 words by token count (the plan's version counts about 280).
14. **Three ranges differ from the reports' summaries; the output files were used.** |.| clock / map arm means were
    1.22-1.68 in `sign_clock_out.txt` (SEARCH's summary says 1.10-1.68; no arm mean of 1.10 appears in the output; now
    1.25-1.57 under the audit's symmetric definition);
    removing as many map channels gives 0.03-0.65 as means over draws (LANGUAGE's summary says 0.02-0.65; 0.02 is a draw
    interval bound); cone-boundary share 0.25-0.36 including GAUGE (the plan quotes 0.30-0.36, MECHANISM arms only).
15. **Protocol sentence (C-3)** is label-based ("every contrast reported without '(analysis)' is registered or a declared
    secondary unless marked 'not registered'") instead of listing tables and figures, so that no table is cited before
    Table 1. Unregistered items in the main text are marked "not registered" (EM budget contrast, egocentric actions).
16. **Leak (C-16).** The main text cites the convergence result, which replicated on fresh seeds; the in-distribution
    accuracy gain (+0.0107) and its non-replication (+0.0045, p 0.46) are in App. K3.
17. **Egocentric actions** (plan: two sentences) are sourced to `KNOB_SWEEP_n8.md` in a new App. A5, marked not
    registered, with the index arm at the floor stated there.
18. **Prop. 2(a) wording**: "serves revisits with n|k| well below pi" (the derived condition) instead of the plan's "gap
    < pi / k".
19. **Sec. 8 rule sentence.** The plan's "a word gets a step iff its count per move is fixed" is narrowed: words whose count
    per move varies get no net step; the remaining steps belong to once-per-move words or cancel within their sentence
    (aside words have large steps, so the plain "iff" overstates).
20. **Clock-seed counts (C-17, X10)**: registered 31-38 (from the first visit) for the four registered clock seeds; 26-38
    (consecutive visits, analysis) for the 10 of 24 learned-step models, each with its definition.
21. **"One index layer learns retrace and the clock"** is written "the one-lap offset" in Sec. 6 and Remark 1, to keep
    "clock" for the phase decomposition of Sec. 4.
22. **8-page cut (plan 3.5) not produced.**
23. **Appendix**: lettered A-O as the plan; App. E copies every statement and proof the main text uses; v3 P1(c)-(e),
    P2, P5 and P7 are pointed to, not copied (not used). Old draft-1 App. K / L replaced by new K, L; M, N, O new.
24. **References added**: Alipour et al. 2025, GridPE (Li et al.), Mochizuki-Freeman et al. 2023, Kabir et al. 2025, Yao et
    al. 2021, all marked * (not metadata-checked). Sarkar and Howard 2026 and RoMem (optional in the plan) not added.

## Corrections C-1..C-18 and contradictions X1-X17: where each landed

C-1 Sec. 8 para 1-2, Table 7, App. B / J6. C-2 Sec. 9 "Text". C-3 Sec. 3 protocol (deviation 15). C-4 Sec. 9 "What and
where", App. K1. C-5 Sec. 8 clock paragraph (counts moves, idle channels, 0.009, the three below ceiling, Spearman +0.905;
"solve as well" dropped). C-6 Limitations (route account; per-channel |.| dynamics underived). C-7 App. H4. C-8 Remark 1
(last sentence). C-9 Sec. 6 accounting. C-10 Sec. 7.4, App. M1 (reader, not search at constant capacity). C-11 Sec. 7.2
(Holm 0.054). C-12 Fig. 3 caption. C-13 Prop. 2, Sec. 7.1. C-14 Sec. 9 "integrator view". C-15 App. L. C-16 App. K3.
C-17 Sec. 8, App. J2. C-18 abstract, Sec. 7.3. X1 Prop. 5 / Conjecture 1 labels, Table 1. X2 deviation 5. X3 Table 4, G3.
X4 App. M4 ("the 1D code is the collapsed end state"). X5 Sec. 7.4. X6 Sec. 1 ("at the level of what one attention pass
can read"). X7 Sec. 7.4 last paragraph. X8 App. L2 last sentence; main text scopes the readout claim to T = 1024. X9 C-1.
X10 C-17. X11 Sec. 9 two sentences + App. N (1800-epoch caveat in N). X12 lettering. X13 E4 dropped (nothing on Level 1.5).
X14 deviation 6. X15 App. G4 ("consistent, not a test"; never as a sign account). X16 Remark 1. X17 Sec. 2.1 ("as organised
by GRAPE, Selective RoPE and Mamba-3"; only measurements claimed).

## Claim-map and withdrawn-list compliance (checked in the end-to-end read)

Not used anywhere: +0.461; Dyck +0.168; Match-Query +0.346 and 0.730 vs 0.154; "loop beats three real layers"; "the rank
defect is the periodic code" (Sec. 7.4 says long-gap or wrapped revisits); "two defects, two fixes"; "the model tracks
state changes"; TinyStories "nothing cancels" / "spatial 0%" / "20x"; "the held non-negative model drifts off the
cancelling code" (Sec. 7.3 cites the usage-weighted 98%); "weight decay drives the drift"; "the wrap is CoPE's obstacle"
(Sec. 7.3 says the line fails too); "freeing delta reverses the interaction"; "Selective RoPE cannot learn maps"; "robust to
the recipe" / "at any learning rate"; "no cost without an inverse pair" (App. H4 says only "There is no evidence of a
sign cost without an inverse pair", the claim map's GAUGE wording); the opposition score
0.06 vs 1.92 (not wrap-aware). "Rank = D is hard because of the clock" is a do-not-claim in the claim map: the text states
the route inventory as proved, the hiding feedback as a conjecture, the drift-on-the-weakest-axis geometry as analysis, and
says "in line with this" / "accounts for", not "because". (2026-10-23: the registered G-1 E2 reading licenses the scoped "on the torus the
rank-2 obstacle is the clock"; see the G-1 section.) Sharing vs per-head latents: only the descriptive "whether the
latent is shared or per head" (both 8/8), as in draft 1.

## Rule-28 end-to-end read (done 2026-10-21)

Read both files start to end after assembly. Fixed during the read:
- Abstract: an analysis claim ("depth reaches only revisits a few direction runs away") removed; "Accordingly" -> "In
  line with this".
- Sec. 1: "reach grows by a few runs per layer" (the per-layer hop model is speculation) -> "reach a few direction runs".
- Sec. 3: protocol sentence cited Tables 2, 5, 6 before Table 1 -> label-based (deviation 15).
- Sec. 4: usage q_c and "blind" were used before being defined -> defined in the notation paragraph; U, beta and c in
  Prop. 4(c) defined in place.
- Table 1 cited Table 4 and Fig. 5 before they appear -> section references.
- Sec. 7.1: "every model that succeeds ends clock-free" -> "ends with clock-free heads" (two solved rank-2 runs keep a
  drifting head); rank-2 range labelled "head means".
- Sec. 7.2: "as Proposition 5 predicts" -> "the geometry Proposition 5 leads to" (RANK C1: not discriminating).
- Sec. 8: NEEDS-MAP numbers labelled re-scored; DirOnly's two ranges tied to their streams; the word-step rule narrowed.
- Sec. 9: design guidance "let only the tokens that move step the phase" contradicted Sec. 8 (aside words need steps) ->
  "keep observations from stepping the phase"; unregistered items labelled.
- App. C9's "all main-text accuracies are eval mode" contradicted Sec. 8 -> "except where marked re-scored".
- Old appendix section references renumbered (Secs. 4 / 5 / 6.1 / 6.2 / 7 -> 5 / 6 / 7.2 / 7.3 / 8; Proposition 1 -> 3;
  Proposition 2 (CARoPE) -> 8).
Checked: figure numbers 1-8 in order of first mention (after the figures of 2026-10-21, Figs. 3-11 are in order; Fig. 2 is cited in Sec. 1 before Fig. 1, see "Figures added") (MapFormer's own "Fig. 3d" / "Fig. 9" / "Table 2" are marked
"their"); tables 1-7 in order; every "App. X" in the main text exists in the appendix; every proposition used is stated
before use (Prop. 8 is stated where first used, in Sec. 7.3).

### Second rule-28 read, after applying the script audit (2026-10-21)

Read MAIN_DRAFT2 start to end and every changed appendix block. Checked: every new or changed number against its
generator's output (`tw_strata_200walks_out.txt`, `clock_table_out.txt`, `clock_decomp_out.txt`,
`c1_runlevel_once_out.txt`, `ring_probe_window_out.txt`, `ring_class_floors_out.txt`, `ring_plain_bins_out.txt`,
`torus_segments_out.txt`, `kernel_probe_score_rank_hstratum_out.txt`, `probe_stats_hstratum_out.txt`,
`kernel_argmax_odd_out.txt`, `PASS_STRATA_BESTHEAD.txt`, `READER_MDE.txt`, `whatwhere_stats_out.txt`); every "App. X"
reference resolves; Tables 1-7 and Figures 1-8 still in order of first mention; one clock / map definition, stated in Sec.
4 and E7.1', used by Table 4, G3 and H7. Fixed during the read: the abstract's "removes it" (antecedent lost) -> "removes
the clock"; Sec. 7.1's first claim excluded the solved rank-2 runs' drifting heads -> "clock-free heads ... and every
solved rank-2 run with at least one clean head"; Prop. 4(c)'s c now names the observation step (as Sec. 7.2 uses it);
Prop. 6 "code class" -> "code family" (the geometry classes of App. M4 do shift); the Sec. 7.4 heading "acts on the
readout term only" -> "repairs the readout, not the code" (the measured code term moves +0.06); G4 random baselines in the
unsolved-head sentence -> medians; "G-2" in G4 -> RANK_THEORY's G2 (the plan's G-2 is the sign-route batch); paragraphs
touched by the edits re-wrapped to 120 columns. Claim map: no withdrawn claim introduced; "parking" now appears only as
the proved route (Prop. 4(c), Sec. 9) and never as a measured mechanism.

### Figures added (2026-10-21; post hoc drawings, CPU, from the audit-corrected generators)

Old Figs. 4-8 renumbered to 6-9 and 11 everywhere in MAIN_DRAFT2 (MapFormer's own "their Fig." references untouched).
Every caption number below is in the figure's `_out.txt` (`docs/paper/figs/`).
- **Fig. 4 `fig_segments`** (Sec. 6, after the Table 3 paragraph; plan Fig. 8) [A-audited data, drawing post hoc]: per-seed
  bins from `depth/segments_bins.py` -> `depth/out/segments_bins.json` (new generator; reads the `torus_depth_*.npz` /
  `ring_depth.npz` rows in $DEPTH_SCRATCH), asserted by the figure script equal to `torus_depth_out.txt` (Table 3 rows),
  `torus_segments_out.txt` (gap 32-127), `ring_plain_bins_out.txt` ('plain only') and `ring_depth_out.txt` (retrace).
  Seed means (M-DP4 labelling); ring population plain only (depth MINOR); 1L curves labelled eval mode. Text: Sec. 6 ring
  sentence now also gives the plain-only 0.999 / 0.996 / 0.988 (Fig. 4b).
- **Fig. 5 `fig_clock_endstate`** (Sec. 7.1, after Table 4; plan Fig. 9) [A-audited]: `sign/clock_table_out.txt` run
  ratios (one definition, B3 / M-SG1); collapsed = unsolved, both heads < 0.15 and COLLAPSE in `rank/sweep_final_out.txt`
  (per-head r2 s5, s6; block-diagonal s1, s7; block-diagonal s3, s5 are COLLAPSE there but keep one clocked head and are
  drawn as ordinary unsolved runs). Rows added beyond the plan's list: rank 3 and |.| past pi (both Table 4 rows). No side
  panel (M-RK1 / M-RK2).
- **Fig. 10 `fig_kernel_peak`** (Sec. 7.4, end; plan Fig. 11) [A-audited]: (a) `mappope/kernel_argmax_odd_out.txt` (odd
  offsets: RoPE 3 / 7 / 11 / 15 = 569 / 342 / 156 / 21 of 1,088; PoPE 1 = 1,068) and K(3) > K(1) shares from
  `kernel_probe_paper2x2_out.txt`; (b) `kernel_probe_paper2x2.json` per seed, means asserted equal to `probe_stats_out.txt`;
  (c) `kernel_probe_score_rank_hstratum.json` (HEAD_RULE=stratum, B2), means asserted equal to
  `probe_stats_hstratum_out.txt`. Panel (a) shows the argmax distribution, not a kernel curve (no committed generator
  outputs K(D) itself).
- Pre-existing, not changed: Fig. 2 is first cited (Sec. 1, "(Fig. 2)") before Fig. 1.

## Open items

1. **Script audit applied** (see top). What it leaves open is listed there under "Left open".
2. Figures of deviation 1: plan Figs. 8, 9, 11 built (draft Figs. 4, 5, 10); T1, 10, 7d-e, G2, H7, J2 still to build. Move Tables 3 / 4 to the appendix if space is short.
3. CPU-4, CPU-5, CPU-6, CPU-8 (T = 1024 kernel probe at 100 walks). CPU-3 is done for the channel ratio (Table 4, G3,
   H7); G4's latent metric stays separate by design.
4. App. A3 status cells to verify: "reproduced: yes" for LOOP_RANK and TW_NORMSTEP rests on both batches being named in
   `REPRO_AUDIT_nonsign.md`; SCORE_RANK's code check is its Amendment 1 (independent code audit before launch);
   KNOB_SWEEP's seed numbers are not given in `KNOB_SWEEP_n8.md` ("8 seeds").
5. The new references (deviation 24) need metadata checks; Yao et al. is cited via Liu et al. App. A.5 and is not in
   `papers/`.
6. Plan GPU items still open: G-5 (depth routes), G-7 (long text world). RNN_TORUS (recurrent baseline) landed 2026-10-27 and is folded in (abstract clause, contribution 2, Sec. 6 paragraph, Limitations line replaced; App. A3 row, F8 new, O row; main text 11,706 -> 11,956 words). EXPTORUS landed 2026-10-26 and is folded in (Sec. 7.3, Table 5, abstract, Discussion, design guidance, Limitations; App. A3, E8.6, H14; main text 11,570 -> 11,704 words). G-3 block A2 and FACTORIAL
   to n = 8 landed 2026-10-25 and are folded in. G-3 block A (2026-10-23), G-4 and RANKRECIPE (2026-10-24) landed and are folded in. G-1 landed (2026-10-23) and is folded in (section "G-1 (RANKCLOCK) folded in"): E2 THE CLOCK IS
   THE OBSTACLE, P PARTIAL. Follow-ups named by its results file, not run: a registered r4 wd 0.05 vs wd 0 contrast, and
   a longer budget for slotxy and r2 together (slower vs unable).
7. `fig_summary` panel (b)'s title says "training keeps it" for all three bars; the torus |.| bar is re-found, not kept.
   The caption says so; the figure title itself would need a regenerated figure.
8. The abstract is 344 whitespace tokens after G-1 (316 before); TMLR has no limit, but trim if a venue requires 250.
