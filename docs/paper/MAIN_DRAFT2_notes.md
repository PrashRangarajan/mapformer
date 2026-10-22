# Notes on MAIN_DRAFT2 (2026-10-21)

Drafted from `docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md`, draft 1 (`MAIN_DRAFT1.md`, `MAIN_APPENDIX1.md`), the theory
appendix v3, the seven expansion reports, and the results that landed after the plan (DELTA_FREE, SROPE_TORUS, SIGNROUTE,
`docs/audits/2026-10-20/WHAT_WHERE_T1024.md`, `figs/fig_summary.png`). Files: `MAIN_DRAFT2.md` (main),
`MAIN_APPENDIX2.md` (A-O). Nothing committed; no results file, code, run directory, figure or existing draft was edited
(the appendix reuses draft-1 appendix blocks by copy, with section references renumbered).

## Script-audit status

`docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md` did **not exist** when this draft was finished (checked at the end of
the session). Every number below labelled **[A-pending]** comes from `docs/paper/expansion/scripts/` and is **pending
script audit**. Numbers labelled [A] come from older, separately audited or refereed analysis files
(`docs/audits/2026-10-16/`, `docs/paper/figs/*_out.txt`, `docs/WHAT_WHERE_CHECKS.md`, `docs/audits/2026-10-20/`). When the
audit lands: apply its corrections to Sec. 6 (Table 3, the 0.943 accounting, probes, attention), Sec. 7.1 (Table 4),
Sec. 7.2 ("Why the extra rank helps"), Sec. 7.3 ("Why the non-negative step does not find it"), Sec. 7.4 (one-pass reader,
kernel probes), Sec. 8 (Table 7, step table, asides, clock channels), Sec. 9 (used-channel rho), and App. F, G3-G6, H4,
H7, J1-J6, L2, M2-M4. The what/where T = 1024 script (`docs/audits/2026-10-20/causal_whatwhere_T1024.py`) is not in the
audit's list; treat K1 as [A], unaudited.

## Word counts

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
| 6 What index depth buys | 806 | 340 (Fig. 3, Table 3) |
| 7 opening | 67 | -- |
| 7.1 The clock and the signature | 182 | 160 (Table 4) |
| 7.2 Per-head rank | 639 | 416 (Table 5, Figs. 4-5) |
| 7.3 The sign of the phase step | 1,242 | 509 (Table 6, Figs. 6-7) |
| 7.4 Readers cannot repair a code | 536 | -- |
| 8 Maps from words | 838 | 244 (Table 7, Fig. 8) |
| 9 Discussion and related work | 1,152 | -- |
| 10 Limitations | 227 | 19 |
| **Sections 1-10 + abstract** | **about 9,750** | **2,256** |
| References | 769 | -- |

Target was ~8-10k words of main text including a ~1-page theory section: met (theory section 1,345 + table 233). Appendix
24,781 words (A 1,982; B 624; C 2,812; D 886; E 6,123; F 1,929; G 1,951; H 2,788; I 764; J 1,668; K 624; L 890; M 879;
N 218; O 476).

## Every number and its source

R = registered decision or declared secondary; A = analysis (post hoc, eval-only); A-pending = analysis from
`docs/paper/expansion/scripts/`, pending script audit; P = theory (proved); C = conjecture; D = descriptive / structural.

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
- NEEDS-MAP 0.961 vs 0.138 [A-pending]: `language/tw_strata_out.txt` (path re-scored, fresh seeds 2-7 0.9609; RoPE 2L 0.1376).

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
differs": ring 1L retrace 0.68-0.80, lap 0.99-1.00, 3L 0.997 / 0.993 / 0.972, R^2 0.69 vs 0.18 [A-pending]:
`depth/out/ring_depth_out.txt`; torus R^2 <= 0.21 [A-pending]: `depth/out/torus_probe_out.txt`; 0.996 / 1.000, 0.874,
0.493, attention 0.55-0.58, 0.00-0.01, 0.81, 98.5% and median 37, 0.911 and 0.943 accounting [A-pending]:
`depth/out/torus_depth_out.txt`. Table 3 [A-pending]: same file (DEPTH 4.1). LOOP_RANK strata 0.671 -> 0.927-0.990,
0.557 -> 0.747-0.887 [R secondary]: `LOOP_RANK_STRATA.json` via RECURRENCE Sec. 1.

**Sec. 7.1 and Table 4** [A-pending]: `sign/sign_clock_out.txt` (arm means: MECH SO 0.018, SO_absmag 0.015, GAUGE SO_m
0.016; AO 1.683, AO_absmag 1.632, AO_m 1.549; SO_absmag_fz 1.394; AO_high 0.312 with 0.82 usage on cancelling; TORUS_START
SO4 0.001, AO4 1.219) and `sign/rank_clock_out.txt` (unsolved ch-clock head means 0.625 / 1.910 / 0.509; solved rank-2
clean heads 0.029-0.072, drifting 0.936 / 1.431; rank 3 / 4 solved 0.043 / 0.049 / 0.051). Strata [R secondary]:
`fig_strata_modes_out.txt`. Usage-weighted cancelling 70-93% vs 2-10% [A]: `FIGS_NOTES.md` (fig_wrap_share table, App. H6).

**Sec. 7.2 and Table 5.** Six arms, SOLVED, accuracy, p 0.0070, +0.102 p 0.027 Holm 0.054 (C-11), 478-532, 1800 epochs
0/8 (0.908) vs 7/8, Fisher 0.0014, +0.086 p 0.0003, 0.9955 / 0.981 / 7/8 / 7/8 / 1/8 p 0.0009, 2D 1/8 vs 8/8, 3D 1/8,
4/8, 8/8, ring 7/8 vs 8/8 [R]: as draft 1; R1DF 24 vs 58 epochs [R]: `RANK_TESTS_RESULTS.md`. SCORE_RANK 3/12 vs 3/12, 8/8
[R]: `SCORE_RANK_RESULTS.md`. "Why the extra rank helps" [A-pending]: `rank/summarise_final_out.txt` (off-plane 0.96-1.00 in
9 of 10 sets, random 0.50-0.71, readgain 0.09-0.27), `rank/c1_runlevel_out.txt` (57/62 vs 28.7), `rank/escape_start_out.txt`
(16/16 heads, rank-sum p 0.018).

**Sec. 7.3 and Table 6.** As draft 1 (SIGN_MATCHED, RING_DIAL, WRAP_EXISTENCE, GENERALITY, LRSWEEP, ROBUST, MECHANISM,
ACCOUNTS) [R]. Exp / softplus 1/8-3/8 [R breadth cells, fixed rates]: `SIGNROUTE_RESULTS.md`. Routes paragraph
[A-pending]: `sign/latent_reach_out.txt` (1.000; 0.339 / 0.561), `sign/path_profiles_out.txt` (0.60-1.92 vs 0.04-0.48;
0.30 -> 0.72 vs 0.30), `sign/latent_geometry_out.txt` (cone boundary 0.30 / 0.36 / 0.25 vs 0.056), `sign/basin_summary_out.txt`
(frozen-content qualification). Hybrid 8/8 at both rates, 1.000, |.| 0/8 and 2/8, Fisher 0.0002 / 0.007, u-ratio 0.043 /
0.002, cosine -0.996 / -1.000, abs_tied 0/16 [R; end states descriptive in the file]: `SIGNROUTE_RESULTS.md`. Clock / map
1.39 for frozen latents [A-pending]. CARoPE 0.667 vs 0.722 p 0.040; CoPE 0.887 p 0.0002, +0.157; line 1/16 [R]:
`GENERALITY_RESULTS.md`, `ACCOUNTS_RESULTS.md`. Selective RoPE +0.236, 0.961 -> 0.507, 0/8 vs 8/8, 0.961 vs 1.000 p 0.0002,
all descending; observation / move step 0.86-0.96 vs 0.027; state-tracking 0.769 0/8, negative-mass share 0.00 [R;
readouts declared secondaries, descriptive]: `SROPE_TORUS_RESULTS.md`.

**Sec. 7.4.** LOOP_RANK 0.894 / 0.973 / 0.990, p 0.0034 / 0.0012, 2/8 / 5/8, rank 4 0.998 [R]: `LOOP_RANK_RESULTS.md`.
Reader 0.866 / 0.894, 1.000 / 0.998, 0.32-0.44, 0.893 / 0.897 p 0.27-0.28, p 0.0002, 0.381, 1/16 2D codes, half 1D
[A-pending]: `recurrence/out/SUMMARY.txt`, `GEOMETRY.txt` (RECURRENCE 5.2-5.4). PAPER2X2 0.999 / 0.971 / 0.679 / 0.805,
+0.154 [R]. Probes 0.995 vs 0.70, 0% vs 53-100%, 0.110 of 0.126 [A-pending]: `mappope/probe_stats_out.txt`,
`kernel_probe_paper2x2_out.txt`. DELTA_FREE +0.122 p 0.0078, +0.180, 50-100% with peaks at 3, 4, 7, 8, -0.018 p 0.31,
0.987 -> 0.62 [R; mechanism readouts declared secondaries]: `DELTA_FREE_RESULTS.md`. T = 1024 C 0.617 -> 0.960, C0 0.81-0.84
and 0.41-0.46, rank 4 0.99 and 0.84 [A-pending; 40 walks per seed]: `mappope/probe_stats_out.txt`.

**Sec. 8 and Table 7.** Retrace 0.934, constant 0.505, reversal-copy 0.597 [A-pending / R]: `language/tw_retrace_floor_out.txt`,
`TEXTWORLD_RESULTS.md`. 0.979 +/- 0.048, 0.505, +0.474 p 0.0022, 0.793 [R]: `fresh_seeds.txt`. 0.993 re-scored [R, registered
re-score]: `docs/audits/2026-10-04/DROPOUT_RESCORE.md`. Table 7, 5.9%, 0.961, 0.944-1.000, 0.138, 0.112 [A-pending]:
`language/tw_strata_out.txt`. Cancel 0.002-0.012, synonyms 1.000, 0.39-0.51, 31-38 channels, cosine +0.999 to +1.000 [R
secondaries]: `TEXTWORLD_RESULTS.md`, `fig_textworld_out.txt`. Step table 0.004-0.03, 0.04-0.16 [A-pending]:
`language/tw_step_account_out.txt`. DirOnly 0.970-0.974 [R, eval mode]: `TW_NORMSTEP_RESULTS.md`. Collapse 0.989-1.000 ->
0.952-0.975, 71-100%; lift 0.962-0.981 -> 0.999-1.000 [A-pending]: `language/tw_attention_asides_out.txt`,
`tw_attention_asides_dironly_out.txt`. 10 of 24, 26-38, 0.06, 0.007, 0.98-1.00, 79-99%, 0.009, 0.03-0.65, 0.09-0.26,
Spearman +0.905 [A-pending]: `tw_step_account_out.txt`, `tw_clock_unit_out.txt`, `usage_tick_out.txt`,
`tw_attention_asides_out.txt`, `tw_common_used_out.txt`. Leak 16/16 vs 0/8, 8/8 vs 0/8 [R]: `LEAK_RESULTS.md`,
`GAIN_PHASE_RESULTS.md`.

**Sec. 9.** 0.9955, 7/8, 8/8, 0-2/8, 0/8, 16/16 [R]. EM 1.000, 0.578 -> 0.928 (budget contrast not registered) [R pieces]:
`WARM_RESULTS.md`, `SPREAD_RESULTS.md`, `SEARCH_RESULTS.md`. What/where 0.998, 0.966 (-0.032), 92.5%, 0.113, 0.25-0.31,
Spearman 0.84 [A]: `docs/audits/2026-10-20/WHAT_WHERE_T1024.md`. rho 0.064-0.079, 0.06-0.23, 21.8-22.1, 9.9-12.3, 4.5-6.6
[A-pending]: `language/crossover_usage_out.txt`, `usage_tick_out.txt`. 1.5053 vs 1.5050 [pilot]:
`TINYSTORIES_PILOT_RESULTS.md`; +0.0056 bpc, 3/3 [R, n = 3]: `CODE_FULLVAL_RESULTS.md`; R^2 0.994-0.999 [A]:
`docs/audits/2026-10-12/alpha_language_out.txt`. Egocentric +0.050 -> +0.488 (n = 8, 16-epoch recipe, index arm on the
0.508 floor; not registered): `KNOB_SWEEP_n8.md`.

**Sec. 10.** Quarter-size gap and descending |.| runs [R]: `ROBUST_RESULTS.md`; Selective RoPE all descending [R]:
`SROPE_TORUS_RESULTS.md`; exp / softplus initial steps +1.051 / +0.706 [R descriptive]: `SIGNROUTE_RESULTS.md`.

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
   are Figs. 4-8. The plan's new figures (T1 schematic, 8 segments, 9 clock end state, 10 loop reader, 11 kernel peak,
   7d-e, G2, H7, J2) were **not built**: CPU-10 had not been done, and this task excluded editing or making figures.
   Their content is carried by tables: Table 3 (segments, for Fig. 8), Table 4 (clock end states, for Fig. 9), Table 7
   (text-world strata), App. M3 (loop reader, for Fig. 10), App. L2 (kernel peak, for Fig. 11), App. J1 / J5 (for 7d-e),
   App. G6 (for G2), App. H7 (for H7), App. J4 (for J2).
2. **Theory status table is Table 1** (plan: "Table 4" at the end of Sec. 4), so that tables are numbered in order of
   first mention. Table 1 cites sections, not later tables or figures.
3. **The plan's opening sentence for Sec. 4** ("This account was developed after the registered batches ...") moved to
   Limitations, per the user's style instruction (no process history; scope stated once).
4. **Proposition numbering.** Plan 4.1 + 4.2 -> Prop. 1; 4.3 -> Remark 1 (cited); 4.4 -> Prop. 2 (signature + tick rule);
   4.5 -> Prop. 3 + Lemma 1; 4.6 -> Prop. 4; 4.7 -> Prop. 5 + Conjecture 1; 4.8 -> Remark 2; 4.9 -> Prop. 6; 4.10 -> Prop. 7;
   4.11 -> Remark 3 (App. E13, used in Sec. 9); CARoPE (v3 P6) -> Prop. 8 (stated in Sec. 7.3). App. E's header maps
   each to its E-label.
5. **CPU-3 (clock-metric harmonisation) not done.** Table 4 uses SEARCH's per-inverse-pair clock throughout (X2's chosen
   definition). RANK's off-plane share and readgain are reported with RANK's own definition (the clock common to all
   moves) in Sec. 7.2 and App. G4, labelled as a different metric; App. G4 says the two agree that what is read in solved
   heads is small and are not numerically comparable. X3's clean-head / other-head split is applied (Table 4, G3).
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
14. **Three ranges differ from the reports' summaries; the output files were used.** |.| clock / map arm means are
    1.22-1.68 in `sign_clock_out.txt` (SEARCH's summary says 1.10-1.68; no arm mean of 1.10 appears in the output);
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
says "in line with this" / "accounts for", not "because". Sharing vs per-head latents: only the descriptive "whether the
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
Checked: figure numbers 1-8 in order of first mention (MapFormer's own "Fig. 3d" / "Fig. 9" / "Table 2" are marked
"their"); tables 1-7 in order; every "App. X" in the main text exists in the appendix; every proposition used is stated
before use (Prop. 8 is stated where first used, in Sec. 7.3).

## Open items

1. **Script audit pending** (see top). Apply `ANALYSIS_SCRIPT_AUDIT.md` when it lands.
2. Build the figures of deviation 1 (CPU-10), then move the corresponding tables to the appendix if space is short.
3. CPU-3 (one clock definition for Table 4 and G4), CPU-4, CPU-5, CPU-6, CPU-8 (T = 1024 kernel probe at 100 walks).
4. App. A3 status cells to verify: "reproduced: yes" for LOOP_RANK and TW_NORMSTEP rests on both batches being named in
   `REPRO_AUDIT_nonsign.md`; SCORE_RANK's code check is its Amendment 1 (independent code audit before launch);
   KNOB_SWEEP's seed numbers are not given in `KNOB_SWEEP_n8.md` ("8 seeds").
5. The new references (deviation 24) need metadata checks; Yao et al. is cited via Liu et al. App. A.5 and is not in
   `papers/`.
6. Plan GPU items still open and named in Limitations: G-1 (clock-slot rank test; decides Conjecture 1 vs
   over-parameterisation), G-3 (SSM), G-4 (torus best-rate sweep), G-5 (depth routes), G-7 (long text world), recurrent
   baseline. If G-1 lands, Prop. 5 / Conjecture 1 and Sec. 7.2's last paragraph change.
7. `fig_summary` panel (b)'s title says "training keeps it" for all three bars; the torus |.| bar is re-found, not kept.
   The caption says so; the figure title itself would need a regenerated figure.
8. The abstract is about 300 tokens; TMLR has no limit, but trim if a venue requires 250.
