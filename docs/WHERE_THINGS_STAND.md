# Where things stand -- 2026-10-29

Orientation for a fresh session, one page. Read `.claude-memory/project_state.md` first (what is running, what the
user must decide), then this. `CLAUDE.md` is the authority on conventions, citable results and withdrawals; this file is
the shape of the project around them. Every number below is copied from CLAUDE.md's rows or the results file named
beside it; a CORRECTED or AUDIT block at the top of a results file supersedes its body. Rewritten 2026-10-29 (cleanup
C6) from CLAUDE.md, `project_state.md`, `docs/paper/BIG_PICTURE_2026-10-24.md`, `docs/audits/2026-10-28/FULL_REVIEW.md`
and the registered results of 2026-10-14..28; the 2026-10-03 version is in git history.

Status words: **REG** = pre-registered, one batch, 8 seeds, verdict as registered (with its amendments);
**pilot-informed** = a registered batch whose setting was chosen after a pilot (said in its row); **post hoc** =
eval-only analysis of stored checkpoints; **pilot** = n = 1-3, not a result. "Within budget" qualifies every failure:
most failing cells still had runs descending.

## The thesis

Began as a reproduction of MapFormer (Rambaud et al., arXiv:2511.19279): a transformer whose rotary angle is a
path-integrated, content-dependent phase `theta = omega * cumsum(W_out W_in x_t)` instead of the token index. The claim
the evidence supports today (BIG_PICTURE, updated by the batches since): **on navigation whose revisits cannot be served
by a local rule, one attention layer whose rotary phase path-integrates a learned step learns the map that index
attention, depth, a loop and recurrent baselines do not learn; whether training finds it is decided by two choices of
the step map -- the sign of the increment and the per-head rank above the map's dimension -- and both are costs of
search, not of representation.** The constrained classes represent the map exactly and keep an installed solution, yet
rarely find it from scratch. Earlier lines (InEKF / Level 1.5, predictive coding, grid cells, hierarchy) ended as
negatives or retractions; most out-of-distribution effects closed under matched-length controls ("robustness is not
capability"). The taxonomy and the content-dependent rotation are published (GRAPE, Mamba-3, Selective RoPE); what is
ours is empirical.

## What survives (numbers in CLAUDE.md and the files)

| result | numbers | status | file |
|---|---|---|---|
| One path layer learns the T=1024 torus map; index attention does not | path 0.998 (8/8) vs RoPE 0.731 (0/8), p 0.0002; map-free retrace floor 0.843 (0.934 at MapFormer's T=128) | REG | `SIGN_MATCHED_RESULTS.md` |
| ...not even at 4 layers or looped x4 | path 0.999-1.000 vs index 1L 0.72-0.73, 4L 0.83-0.86, loop x4 0.86 (0/8 each over two batches, p 0.029 per batch); 4 index layers sit at the retrace floor | REG (n = 4 + 4 fresh) | `FACTORIAL_RESULTS.md`, `FACTORIAL_N8_RESULTS.md` |
| ...nor LSTM / GRU | path 0.9998 (8/8) vs six RNN cells 0.57-0.65 (0/8), p 0.00016; RNNs memorise the training map (0.93-0.98) and do not transfer | REG | `RNN_TORUS_RESULTS.md` |
| Same model from pixels | action-driven step 0.9946 (8/8, at the perception ceiling) vs RoPE 0.701; an image-driven step at token scale stops training (5/8 collapsed) | REG, partly pilot-informed | `IMAGE_TORUS_RESULTS.md` |
| First-person 3D (Memory Maze, heading given) | path 0.331 vs RoPE 1L 0.118 (+0.213, p 0.00016; floor 0.047); vs RoPE 4L 0.389 UNMEASURED (R4 still descending) | REG, loss pilot-informed | `MEMORY_MAZE_RESULTS.md` |
| Navigation told in English | path 0.969 (0.993 re-scored) vs RoPE 1L 0.505, 7/8 vs 0/8; fresh seeds 6/6 vs 0/6 | REG | `TEXTWORLD_RESULTS.md` |
| Depth exchange rate | Dyck at matched depth: +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers (~3 index layers); biased ring: 3 layers (secondary); torus: more than 4 | REG | `DYCK_MDEPTH_RESULTS.md`, `CANCEL_RESULTS.md`, `FACTORIAL*` |
| Sign: a learnability cost of `|.|` and softplus increments | torus -0.177, 0/8 vs 8/8; at each arm's best rate: ring 2/8 vs 8/8, torus 0/8 vs 8/8; exact non-negative solution held (ring) / re-found (torus); hybrid signed increment + non-negative readout 16/16; in a diagonal complex SSM 0/8 vs 8/8; exp: no cost in the SSM at lr 1e-3, pays in attention on the torus (EXPTORUS) | REG (more than ten batches); route-vs-barrier mechanism UNMEASURED (SIGNSEP) | CLAUDE.md sign row |
| Rank: per-head rank D+1, with weight decay | per-head rank 2: 0/8, 2/8, 2/8; rank 3 6/8 (Holm 0.054 vs 2); rank 4 8/8; 2x budget 0/8 vs 7/8; rank-2 recipes 0/24; rank 4 needs AdamW decay 0.05 (8/8 vs 1/8); the rank-2 obstacle is the clock (drift-forbidden rank 2 8/8 vs actions-only 1/8, p 0.0014); clock-exit mechanism PARTIAL; 1D exempt (rank 1 7/8) | REG | CLAUDE.md rank block |
| Published content phases fail the map | Selective RoPE 0.961 0/8 (all descending); CARoPE torus 0.667, below RoPE; CoPE ring 0/8 though it can represent the map | REG | `SROPE_TORUS_RESULTS.md`, `GENERALITY_RESULTS.md` |
| PoPE's score rule, and why | MapPoPE's gain is the score rule (+0.024); PoPE's pinned offset fixes the kernel peak (freeing it +0.122 on the index arm) | REG | `MAPPOPE_PAIR_RESULTS.md`, `DELTA_FREE_RESULTS.md` |
| What/where and the leak | path models do not need the content x position interaction (post hoc); ActOnly / NormStep remove the what-to-where leak, +0.0107 in distribution (accuracy gain did not replicate on fresh seeds; SOLVED did) | REG + post hoc | `LEAK_RESULTS.md`, `docs/WHAT_WHERE_CHECKS.md` |
| Earlier rows still citable | paper torus +0.243 at the converged recipe; Match-Query 0.730 vs 0.154; map-extent threshold; allocentric recoding; EM's recency deficit is search; Dyck / Indirect / Bach replications; hierarchy on text is efficiency only | mixed, see CLAUDE.md | CLAUDE.md citable table |

## Documents

- **Main paper, draft 2**: `docs/paper/MAIN_DRAFT2.md` + `MAIN_APPENDIX2.md` (notes: `MAIN_DRAFT2_notes.md`); PDFs
  `docs/paper/pdf/main_paper_draft2.pdf`, `_appendix.pdf`, `_with_appendix.pdf`. BIG_PICTURE recommends TMLR.
- **Sign companion, draft 6**: `docs/paper/DRAFT6_workshop.md` + `APPENDIX_draft6.md`; PDFs `docs/paper/pdf/4_sign_workshop_paper.pdf`,
  `5_sign_workshop_appendix.pdf`; theory appendix `APPENDIX_theory_v3.md` (`3_theory_appendix.pdf`). BIG_PICTURE recommends
  an ICLR 2027 workshop (non-archival).
- Claim map `docs/paper/CLAIM_MAP_clock_map.md`; advisor read `docs/paper/BIG_PICTURE_2026-10-24.md`; latest review
  `docs/audits/2026-10-28/FULL_REVIEW.md` (no BLOCKER; 6 MAJOR, 13 MINOR). Rebuild PDFs:
  `python3 docs/paper/tools/build_pdfs.py docs/paper/pdf`. Superseded drafts and outlines: `docs/paper/old/`.

## What is open, ranked

1. **CPU, about a day, before any submission** (FULL_REVIEW 4.6): fix M1-M6 and the MINORs in drafts 2 and 6 (cite and
   reconcile Larson 2609.17594 and Sarkar et al. 2610.34945 on rank; cite Dedieu 2024 / Fang & Rajan 2025; mark the
   0.956 / 0.411 text-world stratum post hoc; add rank 3's Holm 0.054; scope contribution 2 with Memory Maze; theory to
   proved statements plus registered tests), read the nine new papers first-hand, a rule-28 end-to-end read, rebuild PDFs.
2. **GPU, only if a reviewer asks**: Memory Maze R4 and GRU to 1200 epochs (both 8/8 descending at 600; ~20-30 A40-h);
   a 3x-budget torus batch for rank 2 and `|.|` (~1 GPU-day); optional, undeclared: IMAGE_TORUS block B, a GPS-PE control,
   many-map RNN training.
3. **Stopped or postponed, do not restart without the user**: TW_AMBIG, TW_LANDMARK (live negative), CTXSTEP, RANK_NOWRAP
   (postponed), RANKDECAY (shelved, gauge flaw).
4. **Cleanup not yet approved** (`docs/cleanup/CLEANUP_PLAN.md`): D1b, D2a / D2b / D3 (checkpoint tiers), B3-B5, A6, the
   topic-folder layout.

## Known stale

- The September `.tex` documents (`positional_review`, `axes_measured`, `mapformer_math`, `report/report`,
  `report/report_short`) stop at 2026-09-30: no rank 3, text world, what/where, leak, sign mechanism, RNN / image / Memory
  Maze. Incomplete, not wrong. Drafts 2 and 6 are the current write-up.
- The shared report (`report/language_summary.html`, v22, 2026-10-11) lacks everything after 2026-10-11. Republish WITH
  `url=` (CLAUDE.md).
- `docs/paper/pdf/7_figures_and_captions.pdf` predates the current `figs/FIGS_NOTES.md`.
- `RESULTS_INDEX.md` hand rows stop at the leak batch; its catalogue is current (2026-10-29). `docs/SESSION_*.md` are
  narratives up to 2026-10-09. Memory files carry dated CORRECTED notes where CLAUDE.md overtook them.
- Moved or archived files: `archive/MOVED.md` (old path -> new path). Archived batches are reproducible from their
  pre-registration commit, not from HEAD (`train.py` / `train_variant.py` differ from 41 recorded md5s).

## The working method, in six lines

1. **Pre-register** the primary readout, branches set against the measured noise floor, and the task's floor
   (the better of an n-gram and a constant, per cell). Report the registered verdict even when it looks silly.
2. **Audit before reading**: an independent, results-blind code-verification agent per batch; its findings go
   into an amendment before any result is read.
3. **One batch, eight seeds**, pilots on seeds outside the batch. Below the MDE the word is "unmeasured".
4. **Matched distribution**: train at the length, depth and setting of the axis the claim is about.
5. **Existence before mechanism**: construct a solution in the class, install it frozen and trainable, then train from scratch.
6. **Convergence before comparison**, and verify what a probe measures by reading its code; check gauges.
