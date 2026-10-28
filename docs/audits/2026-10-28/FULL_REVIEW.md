# Full adversarial review, 2026-10-28: draft 2 (main), draft 6 (sign companion), claim map, and the results of 2026-10-14..28

Read-only. Nothing edited or committed; no GPU job run. Read in full: CLAUDE.md; `MAIN_DRAFT2.md` (every line), `MAIN_DRAFT2_notes.md`,
`MAIN_APPENDIX2.md` (A3, K5, F8, H14 and the section list), `DRAFT6_workshop.md`, `CLAIM_MAP_clock_map.md`,
`BIG_PICTURE_2026-10-24.md`, `docs/audits/2026-10-26/CRITICAL_REVIEW.md`, `project_state.md`; the 16 results files named in the
brief plus `MEMORY_MAZE_GATE_V3.md`; the 12 `*_VERDICTS.json` of the new batches; and raw evaluation files (below). Line numbers refer
to the files as of commit `fd256da` (working tree, 2026-10-28). The novelty section (Part 3) was produced by a web-search agent
whose report I relay with its caveats; every paper it names must be read first-hand before it is cited (CLAUDE.md: read, don't
re-search).

Severity: BLOCKER = fix before any submission; MAJOR = a reviewer will raise it and a sentence the paper leans on changes;
MINOR = wording, labelling, housekeeping.

---

## 1. Accuracy

### 1.1 Trace of the draft-2 abstract, contributions, main-text tables and headline sentences

Every number below was traced to its results file and, where one exists, to the registered `*_ANALYSIS.txt` / `*_VERDICTS.json`.
"OK" = matches to the printed precision and carries the registered label.

| draft 2 location | number(s) | source checked | status |
|---|---|---|---|
| Abstract l.11-12; contr. 1; Sec. 5 l.248-249; Table 2 | retrace 0.843 / 0.934; path 0.998 +/- 0.004 (8/8) vs RoPE 0.731 +/- 0.008 (0/8), +0.267, p 0.0002 | `SIGN_MATCHED_RESULTS.md` l.19-26; floors `sign_floors_out.txt` (per notes) | OK |
| Abstract l.12-13; contr. 2; Sec. 6 | Dyck +0.353 / +0.130 / +0.045 / +0.024; index 0.648 / 0.889 / 0.964 / 0.982; ring 3 layers; torus > 4 | `DYCK_MDEPTH_RESULTS.md` l.46-49; `fresh_seeds.txt`; `FACTORIAL*_VERDICTS.json` | OK |
| Sec. 6 l.270-275; Fig. 3 caption | fresh 0.999 (4/4) vs 0.729 / 0.830 / 0.859 (0/4), p 0.029; pooled 0.999 vs 0.724 / 0.845 / 0.862; 4L - retrace -0.016 / -0.000; 0.913 vs 0.911; long gap 0.41 < 0.51 | `FACTORIAL_N8_VERDICTS.json` (per-seed acc re-averaged: L4 0.8300); `FACTORIAL_N8_RESULTS.md` l.44-61, 71-76 | OK; "-0.000 over both batches" is a descriptive cross-batch mean (file says so; draft says "over both batches" without "descriptive") |
| Abstract l.13-14; contr. 2; Sec. 6 l.277-285 | RNN 0.57-0.65 vs 0.9998; 0/8 vs 8/8; d +0.35..+0.43, p 0.00016 x 6; own map 0.93-0.98; below RoPE 0.724; short gap 0.62-0.70 vs 0.910 | `RNN_TORUS_VERDICTS.json` (Path 0.99976, RoPE 0.7243 re-averaged from `RNN_TORUS_anchor.json`); `RNN_TORUS_RESULTS.md` l.38-62, 70-88 | OK; "fall below one index layer" is declared secondary S1 (label RNN BELOW INDEX 1L), RoPE_L1 read in eval mode (lower bound on the gap) -- say "declared secondary" |
| Abstract l.16-18; contr. 3; Sec. 7.2 Table 3 | rank 2 0/8, 2/8, 2/8 (0.894 / 0.885 / 0.948); rank 3 6/8 (0.987); rank 4 8/8, 8/8 (0.998 / 0.999); D - C_bd p 0.0070 | `RANK_SEP` / `RANK_MI` / `RANK3` (per notes, CLAUDE.md rank block) | OK on counts; **rank 3 vs rank 2 is perm p 0.027, Holm 0.054, and vs rank 4 UNMEASURED (CLAUDE.md) -- neither appears in the main text** (see M6) |
| Abstract l.17; Sec. 7.2 l.356-357 | twice the budget: rank 2 0/8 (0.908) vs rank 4 7/8, Fisher 0.0014 | `LOOP_RANK_E1800_P1_RESULTS.md` (per notes) | Sec. 7.2 OK; **abstract says "against 8 of 8 at rank 4, also at twice the budget" -- at 1800 epochs rank 4 was 7/8** (m7) |
| Sec. 7.2 l.358-365 | RANKRECIPE 0/24; best 3e-3 0.910 vs 1.000, decision p 0.0002 both routes; 12/24 descending (Limitations) | `RANKRECIPE_VERDICTS.json` (0.9095 / 0.9999, p 0.0002); `RANKRECIPE_RESULTS.md` | OK |
| Sec. 7.2 l.364-365; Limitations l.718 | rank 4 with decay 8/8 vs 1/8, p 0.0014; 5/7 descending | `RANKWD_VERDICTS.json`; `RANKWD_RESULTS.md` l.10, 29-30, 76 | OK |
| Sec. 7.2 l.379-391 | RANKCLOCK: r2 1/8, r4 8/8; DF 8/8 at 1.000; AO 1/8; DF vs AO p 0.0014; slotxy 3/8 vs 1/8 Fisher 0.57; 4/5 unsolved descending | `RANKCLOCK_VERDICTS.json`; `RANKCLOCK_RESULTS.md` l.10-36, 45-47, 57-58 | OK |
| Sec. 7.2 l.367-369 | rank-2 projection 0.9955 frozen; held 7/8 vs control 7/8 vs continued 1/8, p 0.0009 | `RANK_PROJ_RESULTS.md` (per notes) | not re-traced (pre-window) |
| Sec. 7.3 l.411-416; Table 4 rows 1-2 | \|.\| 0.821 0/8 vs 0.998 8/8, -0.177, p 0.0002; softplus 0.781 1/8, -0.218, p 0.0003; obs-free 0.880 vs 1.000 | `SIGN_MATCHED_RESULTS.md` l.19-27; `GENERALITY` (per notes; TORUS_LRSWEEP l.99 quotes it) | OK |
| Table 4 row 3; Sec. 7.3 l.418-421 | torus best of three: \|.\| 3e-3 0/8 0.890 vs signed 1e-3 8/8 1.000; p 0.0002 / 0.0095; signed 8, 0, 0 of 8 | `TORUS_LRSWEEP_VERDICTS.json`; cell means re-averaged from `runs/torus_lrsweep/eval/*.json` (0.8903, 1.000) | OK |
| Table 4 rows 4-8 | ROBUST 2L d256 0.967 2/8 vs 0.999 8/8 (p 0.0005 / 0.0070); ring default 0.911 1/8; wd 0 0.873 0/8; lr/3 0.841 0/8 vs 0.950 6/8 Fisher 0.0070; LRSWEEP 0.963 2/8 vs 1.000 8/8, p 0.0070 / 0.0057 | `ROBUST_RESULTS.md` l.17-20; `LRSWEEP` (per claim map) | OK; **the lr/3 row omits the accuracy p 0.089 (unmeasured on that route)** (m8) |
| Table 4 rows 9-10; Sec. 7.3 l.490-496 | SSM 3e-3 \|.\| 0.821 0/8 vs 1.000 8/8; 1e-3 0.756 0/8 vs 0.994 7/8 (Fisher 0.0014, perm 0.0002); softplus 1/8, 0/8; exp 6/8 vs 7/8 Fisher 1.0; 90 of 95; 13 of 32; 9 of 32 | `G3_SSM_VERDICTS.json`, `G3_SSM_A2_VERDICTS.json`; `G3_SSM_RESULTS.md` l.37-43; `G3_SSM_A2_RESULTS.md` l.46-52, 94-103 | OK; **"The exp step diverged at 3e-3" (l.493) -- the file says 8/8 RUNAWAY, "none diverged" (non-finite)**; same in draft 6 l.137 (m9) |
| Sec. 7.3 l.497-502; Table 5 last row | EXPTORUS exp 0/8 (0.916), softplus 0/8 (0.908) vs signed 8/8 (0.993), Holm 0.0003; 95 of 107; 34% vs 41%; 6/8 and 5/8 descending; partly pilot-informed | `EXPTORUS_VERDICTS.json`; cell means re-averaged from `runs/exptorus/eval/EXPTORUS_lr1e-03_EXP.json` (0.9159), `_SO.json` (0.9926), `lr3e-03_SPL.json` (0.9078) | OK |
| Sec. 7.3 l.446-453; Table 5 | MECHANISM 8/8 / 0/8; 7/8 / 0/8 (p 0.0014); 2/8; 8/8 vs 0/8 (p 0.0002); TORUS_START 1/8, 0/8, 2/8, 5/8 descending; GAUGE 5/8 / 2/8 UNMEASURED, RoPE 0.649, 0.92 / 0.01, Fisher 1.00 | `MECHANISM`, `TORUS_START`, `GAUGE` results (per claim map) | OK |
| Sec. 7.3 l.469-474 | frozen latents 1/8 vs 7/8 (p 0.010), clock 1.18; hybrid 8/8 both rates vs \|.\| 0/8, 2/8 (p 0.0002, 0.007) | `ACCOUNTS` T2; `SIGNROUTE_RESULTS.md` l.19-22 | OK |
| Sec. 7.3 l.484-488 | SIGNSEP W 0/8 vs S 8/8 (p 0.0002), P 7/8, F 4/8; UNMEASURED | `SIGNSEP_VERDICTS.json`; `SIGNSEP_RESULTS.md` l.20-28 | OK |
| Sec. 7.3 l.504-510 | CARoPE 0.667 vs 0.722 (p 0.040); CoPE 0.887 (p 0.0002); SRoPE +0.236, 0/8 all descending, 0.86-0.96 vs 0.027, ST 0.769 0/8 | `GENERALITY`; `SROPE_TORUS_RESULTS.md` l.16-44 | OK |
| Sec. 7.4 l.529-545; contr. 4 | loop / depth 0.973 / 0.990 (p 0.0034 / 0.0012), 2/8, 5/8; PAPER2X2 0.999 / 0.971 / 0.679 / 0.805, +0.154; DELTA_FREE +0.122 (p 0.0078), R3+ +0.180, path -0.018 (p 0.31); SCORE_RANK 3/12 | `LOOP_RANK`, `PAPER2X2_RESULTS.md` l.9-14, `DELTA_FREE_RESULTS.md` l.15-25, `SCORE_RANK` | OK |
| Table 2, T = 128 "path, rank 4" 1.000 | | `PAPER2X2_RESULTS.md` l.13 (`Vanilla_r4` 1.000 +/- 0.000) | OK |
| Sec. 8 l.562-567; Table 6; contr. 5 | fresh 0.979 +/- 0.048 (6/6) vs 0.505 (0/6), +0.474, p 0.0022; RoPE 2L 0.793; 13.4% / 856 / 6,381; path 0.956 all seeds (0.978 fresh; re-scored) vs RoPE 2L 0.411, 1L 0.454, constant 0.506; 6.6% / 423: 0.925 vs 0.129 vs 0.088 | `fresh_seeds.txt` l.2-5; `tw_strata_200walks_out.txt` l.1-4, 43-61 | OK as numbers; **contribution 5 quotes the pilot-inclusive, post hoc re-scored 0.956 inside a clause labelled "(registered)"** (M4); the all-seed path mean includes s0 (a pilot seed) at 0.786 |
| Sec. 8 l.608-617; Table 7 | 0.995 (8/8, >= ceiling 0.993) vs 0.701 (0/8), +0.294 Holm 0.0003; 0.994 / 0.994; probe 0.985 / 0.980 / 0.965; 0.668 (5/8 collapsed); gain 1 0.994 8/8; P1' unmeasured | `IMAGE_TORUS_VERDICTS.json`; per-seed `test\|1024` in `runs/image_torus/*/*/eval_image.json` equal to the file's table | OK |
| Sec. 8 l.630-640; Table (none) | Memory Maze 0.33 vs 0.12, d +0.21, p 0.00016; GRU 0.16; P_raw 0.13; R4 0.39 Holm 0.07; P_odo 0.54 vs 0.93; floor 0.05; 2.95x | `MEMORY_MAZE_VERDICTS.json`; **re-derived from the 64 `results.json` at `/home/prashr/mm_data/batch_runs` (read-only): P_E 0.3309, R1 0.1175, R4 0.3894, GRU 0.1558, P_odo 0.5387, GPS 0.2974, P_raw 0.1289, P_E_frozen 0.2416; held 0.3823 / 0.1332; exact permutation over 12,870 splits P_E - R1 0.000155, P_E - R4 0.0715; 5,132 scored re-entries** | OK; main text omits that P_E is bimodal (sd 0.080; 5 seeds 0.27-0.28, 3 at 0.37-0.46) and that the GRU is 8/8 still descending (both in App. K5) (m10) |
| Sec. 9 | what/where -0.032 (Holm 0.08), 0.25-0.31; 0.064-0.079 / 0.06-0.23 / 21.8-22.1 / 9.9-12.3; 1.5053 vs 1.5050; +0.0056 bpc 3/3; +0.050 -> +0.488 not registered | per notes (audited generators) | not re-traced |

Spot checks to raw files (14, all equal to the printed digits): Memory Maze 8 arm means on eval and held plus two exact permutation
p values; EXPTORUS exp 1e-3, signed 1e-3, softplus 3e-3; FACTORIAL_N8 RoPE_L4; RNN_TORUS Path_L1 and RoPE_L1; TORUS_LRSWEEP AO 3e-3
and SO 1e-3; IMAGE_TORUS I_ActOnly, I_RoPE, I_MapWM, I_MapWM_g1 per seed. No mismatch found anywhere in the trace. The new batches'
results files each re-derive their own numbers independently (verify_numbers scripts under `docs/audits/2026-10-27/` and
`2026-10-28/`), which this review confirms for the sample above.

### 1.2 Accuracy findings

No number in the main text is wrong. The findings are labelling and scope:

- **M4 (MAJOR) Contribution 5, l.67-68.** "One path layer learns navigation told in English (registered), 0.956 against 0.411 for two
  index layers on the revisits retrace cannot reach". 0.956 is (i) an analysis number (`tw_strata_200.py`, post hoc stratum), (ii)
  averaged over all 8 seeds although seeds 0-1 are the pilot (rule 6; the fresh-seed value is 0.978) and (iii) re-scored with the post
  hoc attention-dropout correction, while 0.411 is eval mode (the correction does not move the RoPE arms, so the pair is fair, but the
  reader is not told). Fix: "0.978 against 0.43 on fresh seeds (analysis; re-scored, App. C9)" or keep 0.956 / 0.411 and move
  "(registered)" so it covers only the 0.979 vs 0.505 contrast. Table 6 is correctly labelled; the contribution line is not.
- **M6 (MAJOR) Rank 3, Sec. 7.2 l.339, Table 3, Fig. 5a.** Rank 3's 6/8 (0.987) is presented as a rung of a monotone ladder. CLAUDE.md
  carries the test: vs rank 2 perm p 0.027, Holm 0.054; vs rank 4 UNMEASURED. The main text has no p for rank 3 anywhere. Fix: add
  "(vs rank 2 Holm p 0.054; not separated from rank 4)" in the Table 3 row or the sentence; otherwise a reviewer reads rank 3 as a
  measured intermediate.
- **m7 (MINOR) Abstract l.17.** "against 8 of 8 at rank 4, also at twice the budget" -- the 1800-epoch rank-4 control was 7/8
  (Sec. 7.2 l.357 has it right). Fix: "8 of 8 at rank 4 (7 of 8 at twice the budget)".
- **m8 (MINOR) Table 4, ring lr/3 row (l.433).** Cell "Fisher 0.0070" hides that the accuracy route was unmeasured (p 0.089,
  `ROBUST_RESULTS.md` l.20). Add "(accuracy p 0.089)" as the other rows give both routes.
- **m9 (MINOR) Sec. 7.3 l.493 and draft 6 l.137.** "The exp step diverged at 3e-3" -- the registered class is RUNAWAY (loss above the
  memoryless level or step > 1e3); `G3_SSM_RESULTS.md` l.45 says "none diverged". Write "ran away (8/8 RUNAWAY)".
- **m10 (MINOR) Sec. 8 l.635-639 (Memory Maze).** Give P_E's spread (0.33 +/- 0.08, bimodal: five seeds 0.27-0.28, three 0.37-0.46;
  the low mode alone is above every R1 seed) and say the GRU, like R4, is 8/8 still descending. Both facts are in App. K5 and in
  `MEMORY_MAZE_RESULTS.md` l.99, 161-165; the main text's "above a GRU ... (0.16)" reads as settled.
- **m11 (MINOR) Sec. 6 l.282.** "fall below one index layer (0.724)" is declared secondary S1 of RNN_TORUS, with RoPE_L1 in eval mode
  (lower bound on the gap, `RNN_TORUS_RESULTS.md` l.178-179). The sentence ends "(registered)"; say "declared secondary" for S1-S3.
- **m12 (MINOR) Sec. 5 l.250.** "Two further registered batches give 1.000 against 0.722 (n = 8) and 1.000 against 0.719 (n = 4)" is
  stale: FACTORIAL_N8 (0.999 vs 0.729), SROPE_TORUS (1.000 vs 0.725) and RNN_TORUS (0.9998 vs 0.724) add three more in-batch anchor
  pairs. Say "five further batches (0.9998-1.000 against 0.719-0.729)"; it strengthens the headline for free.
- **m13 (MINOR) Sec. 1 l.39-40.** "Every model we see fail keeps short-gap revisits and loses long-gap and wrap-only ones" -- no longer
  true of every failing model: the LSTM / GRU cells are below retrace even on short gaps (0.62-0.70) and the collapsed image MapWM
  seeds are at the blank floor in every stratum. Scope to "every one-layer attention model with a phase that fails".
- **m14 (MINOR) Sec. 6 l.274.** "-0.000 over both batches" is a cross-batch descriptive mean (`FACTORIAL_N8_RESULTS.md` l.44-48,
  "derived, descriptive"); the fresh-seed value is -0.016 (1/4 above). Say "fresh -0.016, pooled -0.000 (descriptive)".
- Budget-scoped readings stated with their scope: checked for every failing cell quoted in the main text (rank 2 recipes, rank 4
  wd 0, |.| best rates, exp / softplus, SRoPE, clock slots, R4 / GRU on Memory Maze). All carry "within 900 epochs", "still
  descending" or "budget-scoped" in the sentence or the Limitations bullet. Pilot-informed results: IMAGE_TORUS P1 / P2 / G and
  MEMORY_MAZE primary are disclosed where quoted (Sec. 8, Limitations, App. K4-K5). UNMEASURED readings quoted as findings: none
  (GAUGE, SIGNSEP, RANKCLOCK P, Memory Maze S1, P1' are each called unmeasured / partial / half-read). Withdrawn results: no
  withdrawn number or sentence from CLAUDE.md's list appears in draft 2 or draft 6 (checked the list item by item; the notes'
  compliance list is accurate).

---

## 2. Consistency

### 2.1 Draft 2 vs draft 6 vs CLAUDE.md vs claim map vs results files

- **Sign story after EXPTORUS -- consistent.** Draft 2 abstract l.19-24, Sec. 7.3 l.490-502, Sec. 9 l.656-658, Limitations l.708-713;
  draft 6 l.134-143, l.252-253, l.262-268; CLAUDE.md sign row; claim map 2026-10-26 update and do-not-claim l.282 all say the same
  thing: |.| and softplus pay everywhere; exp pays in attention on ring and torus and escapes only in the SSM on the ring at 1e-3;
  mixer / task / start not separated. No contradiction.
- **Draft 6 is stale in two places.** l.264 "no recurrent baseline" (RNN_TORUS and the Memory Maze GRU now exist; for the sign
  question there is still no recurrent sign test, which is what the sentence should say); l.137 "diverged" (m9). Draft 6's title and
  abstract still name only |s| and softplus(s) while its Sec. 4.1 and Limitations now say exp pays in attention too -- acceptable (the
  SSM exception stops a blanket title), but the abstract's "the cost holds ..." sentence could add "and for exp(s) in attention".
- **CLAUDE.md.** Every new batch has a row or a block (RNN_TORUS l.69, IMAGE_TORUS l.71, MEMORY_MAZE l.72, FACTORIAL_N8 in l.68, G3 /
  LRSWEEP / SIGNSEP / EXPTORUS inside the sign row l.77, RANKWD l.116, RANKRECIPE l.122). Two of the 2026-10-26 review's items remain
  open there: (i) the sign row (l.77) is one cell of several thousand characters and is now unreadable as a table row (B MINOR); (ii)
  "rank 4 NEEDS AdamW decay 0.05" (l.116) and "RANK D+1 NEEDED SURVIVES A THREE-RECIPE SEARCH" (l.122) keep the unscoped verbs that
  B-M6 asked to scope ("within 900 epochs"); the results-file headings `RANKRECIPE_RESULTS.md` l.11 and `RANKWD_RESULTS.md` l.10 too.
  Draft 2 itself was fixed (l.356, 364).
- **Claim map vs draft 2.** C6's Match-Query entry (0.730 vs 0.154, n = 5, not pre-registered) is correctly absent from draft 2.
  The claim map's C8 (natural text) is in draft 2 only as the n = 3 pilot sentence (Sec. 9 l.676-679) with "n = 3 pilots" stated:
  consistent. The claim map's "Do not claim" list and draft 2: no violation found ("rank = D is hard because of the clock" is scoped
  to the torus; "exp succeeds because of its start" is written "consistent with"; "navigates from pixels" is not claimed for Memory
  Maze; "one path layer ~ three attention layers" is said not to hold there).
- **Results-file internal consistency.** `G3_SSM_RESULTS.md` carries an UPDATE banner pointing to A2 (good). `IMAGE_TORUS_RESULTS.md`
  l.196 still says the image survey "does not exist" (it does: `docs/IMAGE_TASKS_SURVEY_2026-10-25.md`) -- open since the last review.
  `project_state.md` has 25 "## Running" / "## Last results" headers against its own "live state only" rule -- open since the last
  review. `MAIN_DRAFT2_notes.md` item 7: `fig_summary` panel (b)'s title still says "training keeps it" for the torus |.| bar, which is
  re-found, not kept -- the caption is right, the figure is not regenerated.
- **Figure order.** Fig. 2 is cited (l.40) before Fig. 1 (l.47). Harmless, but a copy-editor will flag it.
- **Memory Maze in the front matter.** The abstract and contributions do not mention Memory Maze at all, while Sec. 8, CLAUDE.md and
  the claim map carry it as a registered PATH WINS. Either is defensible; but contribution 2's exchange rate ("about three ... more
  than four") now has a registered counter-example (R4 above P_E on average, unmeasured) that only Sec. 8 and Limitations state. One
  clause in contribution 2 ("on Memory Maze four index layers are not separated from one path layer") would pre-empt the reviewer.

### 2.2 Status of the earlier reviews' findings

| finding (2026-10-24 BIG_PICTURE / 2026-10-26 CRITICAL_REVIEW) | status |
|---|---|
| B-B1 GAUGE missing from the main text | addressed (Sec. 7.3 l.476-482, Table 1 row, Limitations) |
| B-B2 sign framing; run exp in attention on the torus | addressed (renamed; EXPTORUS run and folded: BOTH PAY) |
| B-M1 triage / LSTM before Memory Maze | addressed (RNN_TORUS ran 2026-10-26; Memory Maze after a passed v3 gate) |
| A-B1..A-B3, A-M1..A-M5 (Memory Maze design) | addressed in the registration: learned (action, heading) step map as the primary arm, given odometry as the ceiling, same-pose-bin frames out of the negatives, pose-bin readout, extent gate (median 146 open cells), autoencoder not time-contrastive, GRU and GPS-PE arms, rule 14 (weakly). The review's "P alone is the GPS result" objection is answered by P_E (learned map) vs P_odo (given): S3 shows the learned step costs 0.21 |
| B-M2 IMAGE_TORUS primary swap stated | addressed (l.615-617, Limitations l.724-727) |
| B-M3 "leak" wording | addressed (l.660-664) |
| B-M4 pixels oversold | addressed (one paragraph, "survives a pixel front end") |
| B-M5 RANKRECIPE "search favours rank 2" | addressed in draft 2 (l.360-364); `RANKRECIPE_RESULTS.md` l.13-14 still says "The search favours rank 2 ... conservative" |
| B-M6 "survives / needed" headings | draft 2 fixed; CLAUDE.md l.116 / l.122 and the two results-file headings not |
| B-M7 Larson / Howe / Heo | addressed (Sec. 7.2, Sec. 9, references) |
| B-M8 length | addressed (18.0k -> 11.6k), then grew back to 12.3k words with RNN, EXPTORUS and Memory Maze; still acceptable for TMLR |
| B-M9 recurrent baseline | addressed (RNN_TORUS) |
| MINORs: project_state collapse; IMAGE_TORUS l.196; CLAUDE.md sign row split; RANKRECIPE Checks section | open, open, open, done |
| BIG_PICTURE A4 "theory cut to proved statements + registered causal tests" | partly: Props. 5-6, Conjecture 1, Remark 2 condensed to one paragraph; Sec. 4 still has eight propositions, a lemma, a conjecture and two remarks (1,342 words + Table 1) |

---

## 3. Novelty (from the web-search agent's report, 2026-10-28; verify each first-hand before citing)

Searches covered MapFormer citers (Semantic Scholar: still 0 indexed), Shakerinava (13 citers), Howe (1), Larson (2), PoPE (12),
Memory Maze 2210.13383 (58, all scanned) and ~16 keyword searches. Nothing scoops a contribution outright. What was found beyond
`docs/lit/`, `papers/INDEX.md` and BIG_PICTURE Sec. 2:

| # | contribution | verdict | nearest prior / concurrent work | wording the paper should use |
|---|---|---|---|---|
| 1 | map-free floors, strata, one layer 0.998 vs 0.731 | **novel (the floor critique); incremental in kind** | MapFormer Table 2; in-context map formation with index positions: Dedieu et al. 2024 (2401.05946, in `docs/lit`, NOT cited in draft 2) and Fang & Rajan 2025 (2506.19686, "From Memories to Maps", transformers build gridworld maps in context) | "In-context map formation in transformers with index positions is known (Dedieu et al. 2024; Fang and Rajan 2025); we measure what a path-integrated phase adds at a length where index attention falls to the map-free floor" |
| 2 | exchange rate; local rules; RNNs memorise and do not transfer | **novel as a measurement; budget-scoped** | Liu 2023 / Heo 2026 (cited); nothing found on RNN transfer to redrawn maps; Wang et al. 2509.22353 (ICLR 2026, in-context learning in world models: environment recognition vs learning) is the framing citation for "learning a new map from context" | keep "within our budgets; torus at n = 4, replicated" |
| 3a | per-head rank D vs D + 1; clock forbidden -> rank 2 solves | **partially anticipated and in tension** | **Sarkar, Mukherjee, Sato, arXiv 2610.34945 (26 Oct 2026)**: attention with per-entity orthogonal frames; minimal faithful width is necessary and at that width exactness forces a faithful representation; empirically the minimal width (d = 3) learns MORE reliably than d = 6 (17/17 vs 7/18) on A4 / D4 -- the opposite direction to ours; cites Shakerinava, Howe and Larson. **Larson's second paper 2609.17594** (rank caps give a sharp transition at the number of bindings; effective rank tracks it). Neither mentions RoPE, path integration, navigation or a clock | "Concurrent work on group composition finds that the minimal faithful dimension is what gradient descent recruits (Larson 2026a, b) and learns most reliably (Sarkar et al. 2026); our tori show the reverse at rank D vs D + 1 under a sparse revisit loss with a shared readout and weight decay, and we name the clock as the reason (registered: forbidding it lets rank D solve)". This must be reconciled, not only cited |
| 3b | sign of the step as a learnability cost; SSM replication; past-pi wrapping; signed increment under non-negative readout | **novel; the genre has precedents** | Shakerinava 2026, Howe 2026 (cited); **Shin et al. 2608.30692** (video world models on Memory Maze: tracking hidden state needs negative transition eigenvalues -- the representational argument, no positional encoding); **Jia 2610.00329** (non-commutative Lie-algebraic SSMs; a local convergence condition theta_max < pi, which touches the past-pi story from the other side) | "a constraint that removes no function still costs training (cf. Shakerinava et al. 2026 for non-Abelian SSMs); we isolate it to the sign of one rotary step. Shin et al. (2026) reach the sign conclusion on Memory Maze from representability; Jia (2026) gives a theta < pi convergence condition where our |.| step learns by passing pi" |
| 3 framing | clock / map decomposition; Howard's three cases | **framing, not a result** (BIG_PICTURE already says so) | note: 2610.34945's first author is the SITH-RNN Sarkar of the Howard lab -- the Howard framing is no longer ours to introduce | state once in Background; do not list among contributions |
| 4 | readers vs codes (proved); PoPE peak pinning; delta freedom +0.122 | **novel, narrow** | no PoPE follow-up on the offset or kernel peak among its 12 citers | keep as an appendix-level result |
| 5 words | navigation told in English | **novel demonstration; adjacent LLM line uncited** | Martorell 2502.16690 (LLMs on text grid worlds), Kim et al. 2406.15275 (textualised Gridworld; cited once in `docs/theory` only), 2505.20874, MazeEval 2507.20395 | "Pretrained LLMs track position poorly from text (Kim et al. 2024; Martorell 2025); we show the step map a one-layer model needs to do it" |
| 5 pixels | survives a pixel front end | **partially anticipated** | **gradCSCG, Nikzad et al. 2607.12382**: MNIST-per-location image sequences in aliased grid worlds, VQ-VAE front end, differentiable clone-structured graph | "the MNIST-on-a-grid design is also used by gradCSCG (Nikzad et al. 2026); ours differs in the model and in evaluating on redrawn maps" |
| 5 Memory Maze | first-person 3D with heading given | **novel in the learned-step sense; "loop closure under pose-conditioned rotary" is a named problem** | GTA, PRoPE 2507.10496, ViewRope 2602.07854 (loop-closure fidelity metric), EgoGenesis 2607.28243, MeRoPE 2609.01252, SolarWM 2609.02886 (all take GIVEN camera pose); Laird & Clark 2512.06983 (TMLR 2026; 20 memory mechanisms on Memory Maze, no pose PE); StateLinFormer 2603.23571 | "Pose-conditioned rotary encodings with loop-closure metrics exist for video models given camera pose (ViewRope, PRoPE); ours learns the step from actions and heading and is scored only at blind re-entries; our P_odo arm is in their class" |

Searches that returned nothing: map-free / retrace baselines or critiques of MapFormer-style benchmarks; a clock-vs-displacement
decomposition of a content-dependent phase; non-negative / softplus rotary-step learnability beyond Shakerinava / Howe; Howard 2014
applied to transformers beyond `LIT_HOWARD_TRANSFORMER.md`; ICLR 2027 OpenReview (not indexed).

Required citation additions in Sec. 9 (CPU): Dedieu 2024, Fang & Rajan 2025, Sarkar-Mukherjee-Sato 2026, Larson 2609.17594, Shin
2026, Nikzad 2026, Laird & Clark 2026, one LLM-text-grid paper, ViewRope / PRoPE by name. Each needs a first-hand read; the agent
fetched abstracts and full texts for 15 of them but this review did not.

---

## 4. Overall

### 4.1 The three strongest results

1. **One path-integrated layer learns the 1024-move torus map and nothing else does, measured against the right floor.** 0.998 /
   1.000 vs index RoPE 0.72-0.73 in six registered batches; four index layers and a loop x4 at the retrace floor (n = 8 over two
   batches); LSTM / GRU at 0.57-0.65 on new maps; the same model to the perception ceiling from pixels (0.995) and in English (0.979
   fresh). The floor critique (retrace 0.934 on MapFormer's own length) is the sharpest new sentence in the paper.
2. **The rank-2 obstacle on the torus is the clock, registered and causal** (drift-forbidden rank 2 8/8 at 1.000 vs actions-only
   1/8, p 0.0014), with rank 2 failing at 0/24 across three recipes and at 1800 epochs, and the rank-2 solution held when installed.
3. **The barrier-parameterisation cost at matched length, at best rates, on two tasks, in attention and in an SSM, with the solution
   constructed, installed and kept or re-found** (SIGN_MATCHED, LRSWEEP, TORUS_LRSWEEP, G3 A / A2, WRAP_EXISTENCE, GENERALITY), plus
   the two clean causal cells (frozen latents 1/8 vs 7/8; hybrid 16/16) and EXPTORUS closing the "exp was never run" gap.

### 4.2 The three weakest claims

1. **The mechanism half of the sign story.** The route account's one registered prediction (GAUGE) was not confirmed; the
   route-vs-barrier 2x2 is UNMEASURED; the start lever is ring-only; exp's SSM escape is unexplained. Sec. 4 still spends Prop. 4(b),
   Lemma 1 and part of the Introduction (l.42-47) on an account whose tests did not land. The paper is honest about it, at length.
2. **Rank D + 1 as a rule.** It rests on n_heads = 2, a shared latent (MapFormer's is per head), weight decay 0.05 on all parameters
   (rank 4 fails without it, 1/8), one length, and now two concurrent papers (Larson; Sarkar et al.) reporting the opposite direction
   on group composition. The registered mechanism test (clock exits) was PARTIAL. The claim must be stated as task-conditional.
3. **Memory Maze.** Heading given; loss and budget chosen on pilots; P_E bimodal; four index layers above one path layer on average
   (unmeasured, still descending); given odometry capped at 0.54 by the one-layer readout; the standard probe uninformative. It is a
   registered positive and the disclosures are complete, but it will be read as "a given-compass phase beats one index layer", and
   the paper's own exchange rate fails there.

### 4.3 What a hostile TMLR reviewer writes first

1. "Everything is one or two layers, d 128, on walks of 2,048 tokens; the one external benchmark (Memory Maze) is given the compass,
   and there four index layers beat your one path layer. What does this say about models people use?"
2. "Every negative is 'within 900 epochs' and your tables show runs still descending in most failing cells (12/24 rank-2 recipe runs,
   5/8 |.| at its best rate, 6/8 exp, 8/8 R4 and GRU on Memory Maze)."
3. "Your rank result contradicts Larson 2026 and Sarkar et al. 2026, who find the minimal dimension is what gradient descent learns
   and learns best; your explanation (the clock) was tested and came out PARTIAL."
4. "The theory came after the data; its one registered mechanism prediction was PARTIAL, the 2x2 was unmeasured, the +1/+2 test
   disconfirmed Lemma 1's reading. Cut Sec. 4 to Props. 1-4 and 7."
5. "Three of your five contributions' headline numbers are post hoc strata (0.956 / 0.411; 0.913 vs 0.911; the Memory Maze
   non-local stratum was chosen after a gate), and the text-world number in the contributions list includes pilot seeds."
6. "Rank D + 1 depends on weight decay on all parameters; the recipe search gave rank 2 two recipes at which your own rank-4 model
   fails."

### 4.4 Must be removed or reworded before submission

Nothing must be removed. Reword: M4 (contribution 5's 0.956), M6 (rank 3's Holm), m7, m9, m13; add the Sarkar / Larson reconciliation
sentence (3a) and the Dedieu / Fang citation (1); scope contribution 2 with the Memory Maze clause. Draft 6: l.137, l.264.

### 4.5 Ranked findings with fixes

| rank | id | where | finding | fix |
|---|---|---|---|---|
| BLOCKER | -- | -- | none | -- |
| MAJOR | M1 | `MAIN_DRAFT2.md` Sec. 7.2 l.372-377, Sec. 9 l.686; references | Concurrent work in the opposite direction (Sarkar-Mukherjee-Sato 2610.34945, 26 Oct 2026; Larson 2609.17594) is not cited; the rank claim is stated as a finding about rank, not about this loss / readout / recipe | read both first-hand; add the reconciliation sentence of Part 3 row 3a to Sec. 7.2 and Sec. 9; state "rank D + 1, under a sparse revisit loss, a shared readout and decay 0.05" wherever the rule is given (abstract l.16-18, contr. 3, Sec. 9 design guidance l.658) |
| MAJOR | M2 | Sec. 9 l.682-695 | In-context map formation by transformers with index positions (Dedieu et al. 2024, in `docs/lit`; Fang & Rajan 2025) is uncited; a reviewer reads contribution 1 as claiming the phenomenon | one sentence + two references (Part 3 row 1) |
| MAJOR | M3 | Sec. 8 l.630-640; contr. 2 l.53-56; Limitations l.728-732 | Memory Maze: P_E bimodality and the GRU's 8/8 descending are in App. K5 only; contribution 2's exchange rate has no Memory Maze clause although Sec. 8 says the rate "is not shown here" | add the spread and the GRU qualifier to Sec. 8; add "(not on Memory Maze, where four index layers are unmeasured against one path layer)" to contribution 2 |
| MAJOR | M4 | contr. 5 l.67-68 | 0.956 / 0.411 is post hoc, re-scored, and includes two pilot seeds inside a clause marked "(registered)" | see 1.2 M4 |
| MAJOR | M5 | Sec. 4 (l.151-232), Sec. 1 l.42-47 | Theory load unchanged in kind since BIG_PICTURE A4: eight propositions, a lemma, a conjecture, two remarks; the route account (Prop. 4b, Lemma 1 reading) is still the Introduction's explanation of the |.| failure although its registered tests did not confirm it | move Props. 5-8, Lemma 1's learnability reading and Remark 2 to App. E; keep Props. 1-4 and 7 in Sec. 4; rewrite Sec. 1 l.42-47 as "whether training removes the clock" without the route inventory |
| MAJOR | M6 | Sec. 7.2 l.339, Table 3 | rank 3's 6/8 shown without its test (Holm 0.054 vs rank 2; unmeasured vs rank 4) | add both to Table 3 or the sentence |
| MINOR | m7 | abstract l.17 | "8 of 8 at rank 4, also at twice the budget" (7/8) | "(7 of 8 at twice the budget)" |
| MINOR | m8 | Table 4 l.433 | lr/3 row omits accuracy p 0.089 | add it |
| MINOR | m9 | Sec. 7.3 l.493; `DRAFT6_workshop.md` l.137 | "diverged" for RUNAWAY | "ran away (8/8 RUNAWAY)" |
| MINOR | m10 | Sec. 8 l.635-639 | P_E spread; GRU descending | see M3 |
| MINOR | m11 | Sec. 6 l.282-285 | S1-S3 are declared secondaries; RoPE_L1 in eval mode | label |
| MINOR | m12 | Sec. 5 l.250 | "Two further registered batches" is stale (five) | update |
| MINOR | m13 | Sec. 1 l.39-40 | "Every model we see fail keeps short-gap revisits" is false for the RNNs and the collapsed image seeds | scope to one-layer attention models with a phase |
| MINOR | m14 | Sec. 6 l.274 | "-0.000 over both batches" is a descriptive pooled mean; fresh -0.016 | label |
| MINOR | m15 | `DRAFT6_workshop.md` l.264 | "no recurrent baseline" is stale project-wide | "no recurrent sign test" |
| MINOR | m16 | Sec. 9 l.682-695; references | Shin 2608.30692, Jia 2610.00329, Nikzad 2607.12382, Laird & Clark 2512.06983, ViewRope / PRoPE, one LLM-text-grid paper absent | one clause each (Part 3) |
| MINOR | m17 | `CLAUDE.md` l.77, l.116, l.122; `RANKRECIPE_RESULTS.md` l.11, 13-14; `RANKWD_RESULTS.md` l.10 | sign row one unreadable cell; "NEEDS" / "SURVIVES" unscoped; "the search favours rank 2" | split the sign row into a short row + pointer to the claim map; "within 900 epochs" in the three headings; reword l.13-14 as draft 2 l.360-364 does |
| MINOR | m18 | `IMAGE_TORUS_RESULTS.md` l.196; `.claude-memory/project_state.md` (25 Running / Last-results headers); `figs/fig_summary.png` panel (b) title | open since 2026-10-26 | fix the line; collapse the state file; regenerate the figure or note in the caption that the panel title is loose |
| MINOR | m19 | Sec. 1 l.40 vs l.47 | Fig. 2 cited before Fig. 1 | swap or renumber |

### 4.6 Minimal remaining work

CPU only, about one day: M1-M6 and the MINORs above (draft 2, draft 6, CLAUDE.md, two results-file headings, project_state), after
reading the nine new papers first-hand (the two that can change a sentence are 2610.34945 and 2609.17594; the rest are citations).
Then the rule-28 end-to-end read and the PDF rebuild.

No GPU run is required for submission. Optional, only if a reviewer asks (costs from the results files and BIG_PICTURE):
- Memory Maze R4 and GRU to 1200 epochs (both 8/8 descending at 600; ~16 runs x 1-2 A40-h = 20-30 A40-h) -- would turn S1 from
  UNMEASURED into a reading either way, and it is the one place the exchange rate is contradicted.
- A 3x-budget torus batch for rank 2 and |.| (~1 GPU-day) against attack 2.
- Memory Maze without the compass (a learned heading from frames), the arm that would license "navigates from pixels" (new design;
  weeks). Not for this paper.
