# Notes on MAIN_DRAFT1 (2026-10-18)

Drafted from `docs/paper/OUTLINE_main_v2.md`. Files: `docs/paper/MAIN_DRAFT1.md` (main), `docs/paper/MAIN_APPENDIX1.md`
(appendices A-L), theory detail and check scripts in `docs/paper/APPENDIX_theory_v3.md` (unchanged). Nothing committed;
no results file, code, run directory, figure or existing draft was edited.

## Word counts

Counted by paragraph (`wc`-style tokens); "body" excludes figure captions, table captions and table cells.

| section | body | captions + tables |
|---|---|---|
| Title | 14 | -- |
| Abstract | 222 | -- |
| 1 Introduction | 608 | -- |
| 2 Background | 678 | -- |
| 3 Setup | 698 | -- |
| 4 Sharper measurement (Table 1) | 420 | 91 |
| 5 Exchange rate (Figs. 1, 2) | 598 | 288 |
| 6 opening | 56 | -- |
| 6.1 Rank (Table 2, Fig. 3) | 465 | 268 |
| 6.2 Sign (Table 3, Figs. 4-6) | 779 | 684 |
| 6.3 Failure signature | 129 | -- |
| 7 Maps from words (Fig. 7) | 330 | 161 |
| 8 Discussion | 474 | -- |
| 9 Limitations | 166 | -- |
| **Sections 1-9** | **5,401** | **1,492** |
| References | 658 | -- |

Main text 5,401 words plus 1,492 words of captions and tables (6,893 together), inside the 5,500-6,500 target only if
captions are counted with the text; the body alone is about 100 words under. Abstract 222 words (outline: ~200).
Appendix: about 10,800 words including tables.

## Every number and its source

Status in brackets: R = registered decision or declared secondary; A = analysis (post hoc, labelled in the text); D =
descriptive / structural.

**Abstract, Sec. 1 (contributions repeat the numbers below).**
- retrace 0.934 at T = 128 [A]: `docs/audits/2026-10-16/paper2x2_T128_floors_out.txt` (MEAN 0.9336).
- retrace 0.843, blank 0.506, n-gram 0.575 at T = 1024 [A]: `docs/audits/2026-10-16/sign_floors_out.txt` (0.8429, 0.5059, 0.5749).
- path 0.998, 8/8; RoPE 0.731, 0/8; p 0.0002 [R]: `SIGN_MATCHED_RESULTS.md`.
- index depth to ~0.94 on short gaps; long-gap / wrap-only 0.46-0.51 [R strata, floors A]: `docs/paper/figs/fig_factorial_strata_out.txt` (4L 0.943, loop 0.938; 0.458 / 0.507 / 0.457 / 0.509).
- Dyck +0.353 / +0.130 / +0.045 / +0.024 [R secondary S1]: `DYCK_MDEPTH_RESULTS.md`.
- ring three index layers 0.998-0.999 vs 1.000 (fresh seeds) [R secondary]: `docs/paper/figs/fig_exchange_rate_out.txt`, `docs/audits/2026-09-27/fresh_seeds.txt`.
- torus four index layers 0.860 vs 1.000 [R]: `FACTORIAL_RESULTS.md`.
- text world 0.979 vs 0.505, 6/6 vs 0/6, p 0.0022 [R, fresh seeds]: `TEXTWORLD_RESULTS.md` CORRECTED block; `docs/audits/2026-09-27/fresh_seeds.txt`.
- rank 2 per head 0-2/8, rank 3 6/8, rank 4 8/8 [R]: `RANK_SEP_RESULTS.md`, `RANK3_RESULTS.md`, `RANK_MI_RESULTS.md`.
- 1800 epochs 0/8 vs 7/8 [R]: `LOOP_RANK_E1800_P1_RESULTS.md`.
- rank-2 projection 0.9955, held [R]: `RANK_PROJ_RESULTS.md`.
- |.| -0.177, 0/8 vs 8/8 torus [R]: `SIGN_MATCHED_RESULTS.md`.
- best of four learning rates 2/8 vs 8/8, p 0.007 (ring) [R]: `LRSWEEP_RESULTS.md`.
- weight decay 0 and 2-layer d 256 [R]: `ROBUST_RESULTS.md`.

**Sec. 2.**
- MapFormer Table 2 (rank 2 solves 2D, rank 1 0.66 in 2D; RoPE / TAPE / CoPE up to 4 layers short; PathAtt short in 2D only; CoPE attributed to the gate sign): `papers/txt/mapformer.txt` (Table 2 at l.436-572; text l.584-586, l.1701-1703); Figs. 3d, 3f, 9b at l.620-657, l.2110-2155.
- reproduction 0.989 (MapWM) / 0.987 (MapEM) vs 1.00 (v4) and 0.99 / 1.0 (v1) [D]: `PAPER_TASK_ACCURACY.md`; CLAUDE.md invariants.
- Sarrof Thm. 2, Grazzi Thms. 1 and 3, Mamba-3 Sec. 3.2 and Prop. 3: `docs/paper/review_draft5/PRIOR_ART.md`, `THEORY_CHECK.md` Sec. 14.
- Shakerinava, Helfrich, Amid-Warmuth Prop. 2, Woodworth, LRU, Liu, Hahn-Rofin, PaTH, Wang-Wang, Tran: `PRIOR_ART.md`; `DRAFT6_workshop.md` related work.
- Proposition 1, softplus construction, Lemma 1 (with the "no map function" fix; flip carried by omega in the |.| class): `APPENDIX_theory_v3.md` P1(a), P1(f), T1; `THEORY_CHECK.md` items 3, 5.

**Sec. 3.**
- frequency ladder, dropout 0.1, FFN 4 d_model, ~204k parameters: `APPENDIX_draft6.md` A1; `FACTORIAL_RESULTS.md`.
- AdamW lr 1e-3, wd 0.05, 5% warm-up, cosine to 10%, 98 batches, batch 16 / 128; 100 / 400 eval walks; run classes: `APPENDIX_draft6.md` A3-A4.
- Dyck floor 0.594, chance 0.5: `DYCK_MDEPTH_RESULTS.md` (0.5935).
- 12 of 21 main-text batches with an independent code check; 20 of 21 reproduced: counted from `APPENDIX_draft6.md` A5, `docs/paper/review_draft5/REPRO_AUDIT.md`, `REPRO_AUDIT_nonsign.md` and the prereg amendments (LRSWEEP: code check yes, `docs/audits/2026-10-16/lrsweep_audit.md`; not in either reproduction audit). The 21: PAPER2X2, SIGN_MATCHED, GENERALITY, FACTORIAL, DYCK_MDEPTH, CANCEL, RANK_MI, RANK_SEP, RANK3, LOOP_RANK_E1800 (part 1), RANK_PROJ, RANK_ND, RANK_WRAP, RANK_TESTS, RING_DIAL, WRAP_EXISTENCE, LRSWEEP, ROBUST, ACCOUNTS, MECHANISM, TEXTWORLD. No code check: PAPER2X2, SIGN_MATCHED, DYCK_MDEPTH, CANCEL, RANK_MI, RANK_SEP, RANK3, E1800, RANK_PROJ.

**Sec. 4 and Table 1.**
- T = 128: blank 0.507, n-gram 0.598, retrace 0.934, wrap-only 0.000 [A]: `paper2x2_T128_floors_out.txt`. Index 0.805, path r2 0.971, path r4 1.000, position effect +0.243 [R]: `PAPER2X2_RESULTS.md` (H1).
- T = 1024: blank 0.506, n-gram 0.575, retrace 0.843, wrap-only 0.079 [A]: `sign_floors_out.txt`. Index 0.731, path r4 0.998 [R]: `SIGN_MATCHED_RESULTS.md`. Path r2 0.894 [R]: `RANK_MI_RESULTS.md` (same evaluation stream, seeds 0-7, per `docs/paper/figs/FIGS_NOTES.md`).
- 0.998 +/- 0.004 vs 0.731 +/- 0.008, +0.267, p 0.0002 [R]: `SIGN_MATCHED_RESULTS.md`.
- index below retrace on every run, 0.72-0.74 [A]: `APPENDIX_draft6.md` B; `SIGN_MATCHED_ANALYSIS` per seed (C1) vs `sign_floors_out.txt` per seed.
- 1.000 vs 0.722 [R]: `GENERALITY_RESULTS.md` (B/C table); 1.000 vs 0.719 [R]: `FACTORIAL_RESULTS.md`.
- strata 1.000 / 0.995 / 0.982 and 0.784 / 0.462 / 0.471 [R secondary]; floors 0.506 / 0.500 / 0.507, retrace 0.911 [A]: `docs/paper/figs/fig_strata_modes_out.txt`.

**Sec. 5.**
- Dyck 0.996, 2-4L 1.000; index 0.648 / 0.889 / 0.964 / 0.982 [R]: `fig_exchange_rate_out.txt`, `DYCK_MDEPTH_RESULTS.md` arm table.
- sign-flip p 0.0078, 8/8; 3x 0.997-0.998; mixture +0.110 (8/8); base 32 vs 10000 +0.001 / -0.001 [R]: `DYCK_MDEPTH_RESULTS.md`.
- ring fresh seeds 0.710 / 0.949 / 0.998; 0.677 / 0.962 / 0.998; 0.824 / 0.987 / 0.999 [R secondary]: `fig_exchange_rate_out.txt` (0.6775 printed there; 0.677 in `fresh_seeds.txt`, used).
- two-layer gap 0.013-0.051, three-layer within 0.002 [R secondary, fresh seeds]: `fresh_seeds.txt` (G2 0.0512 / 0.0383 / 0.0133; G3 0.0017 / 0.0017 / 0.0011).
- p_plus 1.0 one index layer 1.000; path solves every run (32/32) [R]: `CANCEL_RESULTS.md`.
- "one segment in ten runs backwards" at 0.9: `CANCEL_RESULTS.md`.
- torus 1.000 vs 0.719 / 0.860 / 0.864; p 0.029; +0.140 / +0.145; 203,925 vs 798,741 parameters; own map 0.897 vs 0.860 [R, secondary for the memorisation check]: `FACTORIAL_RESULTS.md`.
- strata 0.769 -> 0.943 / 0.938; long 0.458 / 0.507; wrap 0.457 / 0.509 [R secondary]; floors 0.515 / 0.517, retrace 0.912 [A]: `fig_factorial_strata_out.txt`.
- Fig. 1 caption numbers: `fig_exchange_rate_out.txt` (floors blank 0.5094, retrace 0.8441); caption adapted from `FIGS_NOTES.md`.

**Sec. 6.1 and Table 2.**
- six arms, SOLVED and accuracy [R]: `RANK_SEP_RESULTS.md`, `RANK3_RESULTS.md`, `fig_rank_solved_out.txt`.
- per-head r2 1/6 on fresh seeds (seeds 0-1 a pilot) [R arm, pilot flag from audit]: `docs/paper/review_draft5/REPRO_AUDIT_nonsign.md` integrity note 1; per-seed classes `RANK_MI_ANALYSIS.txt` (s2 SOLVED, s3-s7 not).
- per-head rank fires, Fisher and permutation p 0.0070 [R]: `RANK_SEP_RESULTS.md`.
- rank 3 vs rank 2 +0.102, p 0.027 [R]: `RANK3_RESULTS.md`.
- initial angle scale (block-diagonal 0.208 vs ~0.33) [D]: `RANK_SEP_RESULTS.md`.
- rank-2 runs flatten at 0.2-0.5; medians cross at 478-532 [R classes, plot A]: `FIGS_NOTES.md`, `fig_loss_curves_rank_out.txt`.
- 1800 epochs 0/8 (0.908) vs 7/8; Fisher p 0.0014; +0.086, p 0.0003 [R]: `LOOP_RANK_E1800_P1_RESULTS.md`.
- 0.9955 frozen, >= 0.981 (min 0.9806); 7/8 vs 7/8 vs 1/8; p 0.0009 [R]: `RANK_PROJ_RESULTS.md`, `RANK_PROJ_ANALYSIS.txt`.
- restart losses 0.37-0.46 (control 0.26-0.30); back at 553-654 (control 497-669) [R output]: `RANK_PROJ_RESULTS.md`.
- 2D 1/8 vs 8/8, p 0.0014; 3D (10/side) rank 3 1/8, rank 4 4/8; 3D (18/side) rank 4 8/8, 1.000 [R]: `RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md`.
- 1D rank 1 7/8 vs rank 2 8/8 [R]: `RANK_TESTS_RESULTS.md`.

**Sec. 6.2 and Table 3.**
- |.| 0.821 vs 0.998, -0.177, p 0.0002, 0/8 vs 8/8; softplus 0.781, 1/8, -0.218, p 0.0003 [R]: `SIGN_MATCHED_RESULTS.md`.
- ring 2/8 and 1/8 vs 8/8, Fisher p 0.007 / 0.0014; ring full p_plus 0.5 accuracy 0.973 [R]: `RING_DIAL_RESULTS.md`.
- obs.-step-free ring 0/8 vs 8/8 [R]: `WRAP_EXISTENCE_RESULTS.md`; torus 0/8 vs 8/8, 0.880 vs 1.000 [R]: `GENERALITY_RESULTS.md`.
- epochs 480 / 57 / 309 / 183; runs flatten at 0.3-0.5 [R classes, plot A]: `fig_loss_curves_out.txt`, `FIGS_NOTES.md`.
- best of four rates: |.| 3e-3 2/8 (0.963) vs signed 1e-3 8/8 (1.000); p 0.0070 / 0.0057; per rate 8, 8, 4, 1 and 1, 2, 1, 0 [R]: `LRSWEEP_RESULTS.md`.
- weight decay 0: 0.873, 0/8 vs 1.000, 8/8, -0.127; lr / 3: 0.841, 0/8 vs 0.950, 6/8, Fisher p 0.0070; default ring 0.911, 1/8; torus 2L d256 0.967, 2/8 vs 0.999, 8/8, Fisher p 0.0070, perm p 0.0005; cancelling share 0.04-0.13 vs 0.43-1.00 [R]: `ROBUST_RESULTS.md`.
- construction: angle error <= 1.4e-6 rad; frozen = source 8/8; weights within ~5x; omega' = max(|omega|, 1); orthogonal move latents [R E1; D]: `WRAP_EXISTENCE_RESULTS.md`, `GENERALITY_RESULTS.md`, `APPENDIX_draft6.md` D2.
- softplus logits 2.8e-11 [A, theory check]: `APPENDIX_theory_v3.md` P1(f); `docs/paper/review_draft5/theory_check_extra.py` (2).
- ring held 8/8, never lost [R E2]: `WRAP_EXISTENCE_RESULTS.md`; 98% usage-weighted [R, ACCOUNTS T1]: `ACCOUNTS_RESULTS.md`.
- torus lr 1e-4; peak 0.11-0.21; re-entry epochs 119-204; 8/8 SOLVED, 1.000 [R E2]: `GENERALITY_RESULTS.md`.
- signed from the |.| start 7/8 vs |.| 0/8, Fisher p 0.0014 [R M5]: `MECHANISM_RESULTS.md`.
- 98-99% opposite-signed channels [A]: `MECHANISM_RESULTS.md` post hoc (`docs/audits/2026-10-14/mech_escape_out.txt`).
- frozen latents 1/8 vs 7/8, Fisher p 0.010; 82% vs 15% opposite-signed [R T2]: `ACCOUNTS_RESULTS.md`.
- |.| from a start past pi 8/8 (ring) [R M2]: `MECHANISM_RESULTS.md`; torus no rescue [R TS1]: `TORUS_START_RESULTS.md`.
- CARoPE 0.667 (0/8) vs RoPE 0.722, p 0.040; CoPE 0.887, 0/8 vs 1.000, p 0.0002; +0.157 [R]: `GENERALITY_RESULTS.md`.
- CoPE on the line 1/16 vs signed 8/8 [R T4]: `ACCOUNTS_RESULTS.md`.
- Figs. 4-6 captions: `fig_existence_out.txt`, `fig_loss_curves_out.txt`, `fig_mechanism` caption in `FIGS_NOTES.md` (6% -> 89%; 4%; 5%; 7% -> 23%).

**Sec. 6.3.**
- strata 0.784 / 0.462 / 0.471; 0.950 / 0.671 / 0.557; 0.870 / 0.630 / 0.515; 1.000 / 0.995 / 0.982; 1.000 / 0.995 / 0.977 [R secondaries]: `fig_strata_modes_out.txt`.
- usage-weighted cancelling share 0.70-0.93 vs 0.02-0.10 [A]: `FIGS_NOTES.md` (fig_wrap_share table).

**Sec. 7.**
- 0.979 +/- 0.048 (6/6), 0.505 (0/6), +0.474, p 0.0022, 2L 0.793; floors 0.505 / 0.597 [R, fresh seeds]: `fresh_seeds.txt`, `TEXTWORLD_RESULTS.md`.
- RoPE 1L at the constant floor and below reversal-copy on all 8 seeds; 2L STALLED 8/8 [R]: `TEXTWORLD_RESULTS.md`.
- cancel to 0.002-0.012; direction steps zeroed 0.39-0.51; synonyms 1.000; clock in 31-38 of 64 channels; cosine +0.999 to +1.000; zeroing non-direction steps costs 0.33-0.69 vs 0.05-0.30 [R secondaries; channel drift from the CORRECTED block]: `TEXTWORLD_RESULTS.md`, `fig_textworld_out.txt`.

**Sec. 8.**
- 0.974-0.989 (identity removed), 0.988-1.000 (shared kernel x gain), index below floor [A]: `docs/WHAT_WHERE_CHECKS.md` Sec. 1.
- R^2 0.994-0.999 (TinyStories and code, MapWM, n = 3) [A]: `docs/audits/2026-10-12/alpha_language_out.txt`.
- 1.5053 vs 1.5050 nats per token [pilot, n = 3]: `TINYSTORIES_PILOT_RESULTS.md`.

**Sec. 9.**
- accuracy gap a quarter at 2 layers d 256 (-0.032 vs -0.120, out of batch); 6/8 |.| runs descending: `ROBUST_RESULTS.md` item 4.

**Appendix** numbers are copied with their source named inline. Two appendix readouts are new arithmetic on committed
per-seed outputs (labelled analysis): the rank SOLVED counts at loss 0.02 and 0.1 (C8; from the final-5% losses printed in
`RANK_SEP_ANALYSIS.txt`, `RANK3_ANALYSIS.txt`, `RANK_MI_ANALYSIS.txt`), and the per-seed accuracies of the block-diagonal,
per-head r3 and per-head r4 arms read from `RANK_SEP.json` / `RANK3.json` (means reproduce 0.948 / 0.987 / 0.999).

## Deviations from the outline, and why

1. **Proposition numbering.** The outline names CARoPE's result "Proposition 6" (its label in the theory appendix). The main
   text has only Proposition 1 and Lemma 1 before it, so it is called Proposition 2; App. E maps every label to v3
   (Proposition 1 = P1(a) + P1(f), Lemma 1 = T1, Proposition 2 = P6, Propositions 3-4 = P3-P4).
2. **Table numbering.** The outline puts an unnumbered rank table in 6.1 and calls the recipe table "Table 2". The rank table
   is Table 2 and the recipe table Table 3.
3. **Figure 4 first mention.** The outline's Sec. 6 opening refers to Fig. 4 before Fig. 3. To keep figures in order of first
   mention, the opening paragraph names no figure; Fig. 4 is first cited in 6.1 (existence at rank 2), after Fig. 3.
4. **Fig. 2 vs Fig. 1 order.** Sec. 4 cites Fig. 2c; Fig. 1c is cited first in the same section (the n = 4 torus contrast,
   "drawn with its own floors in Fig. 1c"), so the order holds.
5. **Abstract and contribution 3: "|.|" not "non-negative parameterisations".** The outline says non-negative
   parameterisations "keep or re-find a constructed solution". Only the |.| construction was trained (the softplus
   construction is built, not tested under training: `APPENDIX_theory_v3.md` P1(f)), so the text says "a non-negative (|.|)
   step".
6. **Abstract, learning-rate claim scoped to the ring.** The outline abstract says the gap "persists at each step's best
   learning rate"; the sweep was on the ring only (`CLAIM_MAP_clock_map.md` LRSWEEP update: "(ring)"), so "on a ring, at
   each step's best of four learning rates".
7. **Protocol counts.** The outline says "(most batches)" had an independent code check. Counted: 12 of 21 main-text
   batches (the nine rank / Dyck / PAPER2X2 / CANCEL / SIGN_MATCHED batches had none); the text gives the count. The check
   happened before launch for every checked batch except TEXTWORLD (while training, before results), so the text says
   "before any result was read".
8. **Reproduction claim (Sec. 2.2).** The outline says the implementation "reproduces their Table 2 2D accuracy". Ours is
   0.989 / 0.987 against v4's 1.00 / 1.00 (it matches v1's 0.99 / 1.0); the text gives the numbers.
9. **MapFormer baselines (Sec. 1).** The outline says MapFormer beats "index, TAPE, CoPE and PathAtt baselines ... on 1D and
   2D". PathAtt solves 1D (0.99 vs MapWM 1.00) and fails only in 2D (`papers/txt/mapformer.txt` Table 2), so the text says
   "(and PathAtt in 2D)".
10. **Dyck position effect defined.** The outline quotes "+0.353 / ..." as the position effect without saying it averages two
    encoding pairs. The text defines it ((MapWM - RoPE) + (MapPoPE - PoPE)) / 2 and adds the PoPE reference
    (Gopalakrishnan et al. 2025), which the Dyck arms require.
11. **Sec. 6.3 cancellation claim narrowed.** The outline says "the failing codes do not cancel opposite moves". The
    cancelling share was measured for the sign arms only (rank-2 opposition scores exist but are not wrap-aware, rule 8), so
    the text says "the non-negative codes"; the Discussion's "both failures leave the same trace" is stated for the strata,
    with non-cancellation shown for the non-negative step.
12. **Sec. 6.3 range.** The outline says solving models are "at 0.98-1.000 in every stratum"; rank 4's wrap-only stratum is
    0.977, so "0.977-1.000".
13. **Ring start past pi in the main text.** The outline moves the start table to App. H. Fig. 6 (main text) shows the
    past-pi start in panel d, so one sentence in 6.2 reports the ring result (8/8 by wrapping) and points to App. H for the
    torus, where the start does not help.
14. **Added detail to reach length.** To bring the body to ~5,400 words, Secs. 3-8 gained sourced material not in the
    outline: the recipe paragraph (Sec. 3), why retrace is strong on short walks (Sec. 4), the Dyck frequency-ladder check,
    the loop's parameter count and the 4-layer memorisation check (Sec. 5), the initial-angle-scale control, rank training
    curves and RANK_PROJ restart losses (6.1), the per-rate learning-rate counts, ROBUST's cancelling shares, the
    construction recipe, the torus re-entry epochs, ACCOUNTS T2's 82% vs 15%, CoPE on the line (6.2), the text-world task
    description and clock-seed details (Sec. 7), and a what/where paragraph from App. K (Sec. 8).
15. **Limitations bullet 2 extended.** `CLAIM_MAP_clock_map.md` (ROBUST update) requires the paper to say the cost "shrinks in
    accuracy at scale"; the bullet states the quarter-size accuracy gap and the descending |.| runs. The lr x3 cell (4/8 vs
    8/8, unmeasured) is in App. H2 with the fresh-seed 3e-3 result (2/8 vs 8/8) beside it, not in the main text; the claim
    map's preferred wording ("shrinks at lr x3") appears only there. **Flag:** the 6.2 paragraph heading "The cost holds
    across recipes and scale" is the outline's; the claim map advises against "robust to the recipe". If that heading reads
    too strongly, "The cost holds without weight decay, at a third of the learning rate, at the best of four rates and at
    2 layers d 256" is the claim-map wording.
16. **Rank heading softened.** The outline's "Claim 6.1.2: not a budget effect at 2x" is written "Twice the budget does not
    rescue rank 2", following the 2026-09-30 audit wording in `LOOP_RANK_E1800_P1_RESULTS.md` (5/8 rank-2 runs still
    descending).
17. **Appendix K leak paragraph.** The outline lists ActOnly / NormStep convergence (16/16 vs 0/8; fresh 8/8 vs 0/8). The
    fresh-seed accuracies (0.9940 vs 0.9895) are added beside the SOLVED counts because the first batch's accuracy gain did
    not replicate (`GAIN_PHASE_RESULTS.md` CORRECTED block).
18. **Appendix L wording.** The outline says the gain is "not from its doubled frequency count (+0.0002)". The COUNT
    contrast sits at a ceiling (0.9995-0.9997) and its CI is withdrawn as a precision claim (`MAPPOPE_PAIR_RESULTS.md`
    CORRECTED block), so the text says the count adds +0.0002 "at a ceiling where both arms sit at 0.9995-0.9997".

## Checks done (end-to-end read, rule 28)

- Every number in the main text re-read against the source listed above after the final edit.
- Cross-references: Figs. 1-7 each cited before or at their placement, in order of first mention; Tables 1-3 cited; App. A-K
  cited from the main text (App. L via "App. F, L" in Sec. 3); appendix figures F1, G1, H1, H2 cited in their sections.
- Terms defined before use: SOLVED / STALLED / DESCENDING, retrace, strata (short-gap, long-gap, wrap-only), per-head rank,
  signed / |.| / softplus, observation-step-free, p_plus, inverse pair and cancelling (App. E; in the main text "cancel
  through zero" is introduced with Lemma 1), "(analysis)".
- Withdrawn list (CLAUDE.md) and the claim map's "do not claim" lists: grep for +0.461, +0.168, "periodic code", "state
  changes", "two defects", Match-Query, "robust to", "any learning rate", "unstable", "fragile", "near the wrap",
  "follows the relative sign", "without an inverse pair", "first to", "exactly when", "counting tokens": no hit in a claim.
  The main text makes no representational claim for the sign result, no design rule from the start lever, no "cost follows
  the relative sign", and no claim that rank = D is hard in general (1D stated).

## Could not be sourced or not verified

- References marked * were not re-checked against metadata in this session: Hewitt et al. 2020 (EMNLP), Eldan and Li 2023
  (TinyStories), Su et al. 2024 (RoFormer, Neurocomputing volume and article number), Gopalakrishnan et al. 2025 (PoPE; venue
  not checked; arXiv id and authors from `papers/txt/pope.txt`), plus the venue-only stars inherited from draft 6.
- MapFormer's "generalises out of distribution to sparser and denser walks" rests on its Table 2 OOD columns (read in
  `papers/txt/mapformer.txt`); not independently reproduced here.
- Nothing else in the main text lacks a source.
