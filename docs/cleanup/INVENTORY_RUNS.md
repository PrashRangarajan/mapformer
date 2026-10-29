# Run-directory inventory for a cleanup (read-only, 2026-10-28)

Inventory only. Nothing was moved, deleted, edited or committed. Host kalman; numbers from `du -sk` and `find`
at 2026-10-28 ~23:00. Sizes are GiB ("GB" below). "Checkpoint" = `*.pt` / `*.pth` / `*.pt.stale` (all gitignored).

**Bottom line.** `runs/` is 30.7 GB, of which 30.6 GB is checkpoints. About 3.8 GB can go with nothing lost (tier 1).
Another 15.8 GB can go if the evals and logs are kept, but that loses named post hoc re-analyses (tier 2). KEEP-CHECKPOINTS
is small: 101 dirs, 6.9 GB. Of that, 2.6 GB is `code2048`; the rest is the registered torus / ring / rank / sign /
EM / Dyck batches behind the current paper and the pilots their analyses read. Keep them whole. Outside the repo,
`/home/prashr/mm_data` (14.8 GB) holds the only local copy of the Memory Maze per-run results (20 MB) and 12.6 GB of re-downloadable zips.

How each directory was classified:
- **Backs which result**: a scan of every `*.md / *.py / *.sh / *.json / *.tex` outside `runs/` (top level, `docs/`,
  `report/`, `.claude-memory/`) for `runs/<dir>`, plus each driver's named outputs.
- **Checkpoints needed**: scripts that both name the dir and call `torch.load` / `load_state_dict`. I read the code
  for the large dirs and for every `.best.pt` vs `.final.pt` question.
- **Status**: CITED = feeds a row of CLAUDE.md "Citable results", a registered results file folded into CLAUDE.md,
  or draft 2 (`docs/paper/MAIN_DRAFT2.md`, `MAIN_APPENDIX2.md`). REP = reported in a results file but not cited.
  ROB = out-of-distribution / robustness only. NEG = listed under "Live negatives". WD = withdrawn, void or
  superseded. PILOT = pilot, smoke or diagnostic.

Classes, as asked: **KEEP-CHECKPOINTS** (cited and re-analysable from the weights). **KEEP-SMALL-ONLY** (keep the
JSON, logs and md5 files; the checkpoints could go). **KEEP-SMALL-ONLY\*** (a split decision: one checkpoint
variant goes, the other stays). **DROP-CANDIDATE** (pilot / smoke / void / withdrawn). **DUPLICATE** (end of section 2).

---

## 1. Disk summary

kalman `/` (md0p1): 1.8 TB, 1.6 TB used, **64 GB free (97%)**. The disk is shared with other lab users; only our
files are counted below.

### 1.1 Repo `/home/prashr/mapformer` (about 34.7 GB)

| folder | size | tracked in git | notes |
|---|---|---|---|
| `runs/` | **30.72 GB** (checkpoints 30.6 GB, 7,267 `.pt`; 18,970 files; 436 dirs + 148 symlinks) | 4,604 files, 11.1 MB (4,394 JSON) | 6,770 per-run `.log` (21 MB) are gitignored and hold the training curves |
| `data/` | 3.46 GB (gitignored) | 0 | `tinystories/` 3.2 GB (raw `TinyStoriesV2-GPT4-train.txt` 2.1 GB + derived `train.bin` 1.0 GB); `image_torus/` 126 MB (MNIST/F-MNIST bank, read by IMAGE_TORUS); `code_*` 107 MB (code corpus, read by the code line); `enwik8` 96 MB + `enwik8.zip` 35 MB (zip duplicates the unzipped file); `Jsb16thSeparated.json` 1.6 MB |
| `.git` | 183 MB (pack 74 MB + loose 106 MB) | -- | `git gc` would pack the loose objects |
| `papers/` | 0.21 GB | 89 files | untracked PDFs under `papers/new_2026-10-1*/` (git status) |
| `paper_figures/` | 53 MB | 167 files (40 MB) | `*.npz` now gitignored, but 4 x 8.4 MB `cnav_*rate_maps*.npz` remain in history |
| `docs/` | 40 MB | 1,675 files | |
| `figures*/` (24 dirs) | ~26 MB | ignored | April 2026 figure output |
| `logs/` | 1.7 MB | 8 files | 393 files, mostly gitignored `.log` |
| `runs_rescore/` | 0.3 MB | 53 files | dropout rescore outputs (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`): **keep** |
| `enwik8_2x2/`, `enwik8_long/` | 0.3 MB | 6 + 24 files | ENWIK8_HIERARCHY (cited): keep |
| `hourglass_enwik8/`, `hourglass_enwik8_long/` | <0.1 MB | 0 | 3 logs + 1 JSON / empty |
| `archive/`, `archive_stale/` | 0.4 MB | 92 files | void / stale results (bannered) |
| repo root loose files | ~29 MB | most | 191 ignored `*.log`; ~145 `.<batch>_done` markers; 4.9 MB `_INIT_GRAD.json` (tracked) |

### 1.2 Largest run directories (checkpoints are about 100% of each)

| dir | GB | | dir | GB |
|---|---|---|---|---|
| `indirect` | 3.02 | | `jsb_len512` | 0.72 |
| `code_ablate` | 2.56 | | `mw_grid_sweep` | 0.61 |
| `code_decay` | 2.56 | | `flipflop` | 0.57 |
| `code` | 2.56 | | `dyck_mdepth` | 0.49 |
| `code2048` | 2.56 | | `tinystories` | 0.40 |
| `indirect_t3` | 1.76 | | `jsb` | 0.36 |
| `indirect_200k` | 1.51 | | `frontier_alg` | 0.34 |

These 14 dirs hold 20.0 GB (65% of `runs/`): the language / PoPE-paper line (code, Indirect, Bach, TinyStories),
MiniWorld, Flip-Flop, Dyck at matched depth and the algorithmic frontier. The registered torus / ring / rank / sign /
EM batches hold 3.1 GB (64 dirs), plus 0.2 GB of pilot / repro dirs that their analyses read (21 dirs).

### 1.3 Outside the repo, our files only

| path | size | content | verdict |
|---|---|---|---|
| `/home/prashr/mm_data/zips/15x15/{eval,train-part0}.zip` | 12.55 GB | official Memory Maze offline data (public Google Drive; ids and resumable fetch in `hyak/mm_download.sh`) | re-downloadable: tier 3 |
| `/home/prashr/mm_data/arrays/` | 2.15 GB | frozen encoder (`encoder.pt`, `ae_ckpt.pt`), embeddings, pose/dir/tvec arrays; read by `train_mm.py` and `docs/audits/2026-10-28/memory_maze/pose_only_probe.py` | keep (derived, but it is the exact input of the registered batch) |
| `/home/prashr/mm_data/batch_runs/` | 20 MB | **the 64 `results.json` + `DONE` + `log.txt` behind `MEMORY_MAZE_RESULTS.md`** (re-derived from here in `docs/audits/2026-10-28/FULL_REVIEW.md`); the cluster copies are on purgeable scratch | **keep; copy into the repo** (section 4) |
| `/home/prashr/mm_data/{pilots,pilots3,smoke_a2}` | 0.08 GB | Memory Maze pilots (pilot tables committed under `docs/audits/2026-10-27/memory_maze_batch/pilots/`) | tier 1 |
| `/tmp/claude-1002/-home-prashr-mapformer/` | 3.2 GB | session scratchpads (verification-agent clones: `imgcheck*`, `rnncheck`, `expcheck`, `torus_lrsweep_smoke`, ...) | tier 1, once no session is active |
| other `/home/prashr` (not this project) | ~245 GB | `helix_transformer/` 150 GB (`train_models/` 150 GB), `.cache/huggingface` 40 GB (pythia-6.9b 13 GB, ...), `miniconda3` 20 GB, `.local` 8.4 GB, `dreamerv3-torch` 6.9 GB, `transformer-mem` 6.3 GB, `.vscode-server` 6.2 GB | outside this inventory's scope; listed because they are 6x the mapformer footprint |

---

## 2. Per-batch table

Columns: size total (checkpoints); the results file it backs; status; which documented re-analysis reads the
checkpoints; number of git-tracked files in the dir; class; tier (section 3). `safe_clear.sh` refuses 187 of the 436
dirs because they have a completion marker (inside the dir, or a repo-level `.<name>_done` / prefix marker). The `git-tracked files` count shows
what survives in git if the dir disappears entirely.

Two facts drive the classes:
1. **Older nav-trainer runs (`train_variant`, MiniWorld, Flip-Flop, Level 1.5, EM) keep `losses` and `results`
   inside the `.pt` dict.** Their JSONs carry only accuracies, and `agg_alias.py` / `agg_visits.py` /
   `agg_miniworld.py` / `agg_refine.py` read the loss curves from the checkpoint. The gitignored `.log` holds
   them too, every 5 epochs. Fifteen KEEP-SMALL-ONLY dirs have no JSON at all and nothing tracked (section 3,
   tier 2 prerequisites).
2. **Most registered torus/ring/rank batches store per-run evals in top-level `<NAME>_*.json`** (tracked) and only
   `.pt` + `.log` per run. Their registered analyses re-score from the checkpoints (strata, censuses, dropout
   rescore, what-where checks), so those checkpoints are the primary data.

### Language / PoPE-paper line (code, Indirect, Bach, TinyStories, addition) -- 23 dirs, 20.07 GB (20.06 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `indirect` | 3.02 (3.02) | INDIRECT_RESULTS (100k; superseded in part; "1/8 at their 100k" cited) | CITED | pope_task_step_table.py (post hoc audit 2026-10-05) | 33 | KEEP-SMALL-ONLY | 2 |
| `code_ablate` | 2.56 (2.56) | ABLATE_RESULTS (PoPE faithful, NoSigma, 80.7% delta in clamp) | CITED | .final.pt: A1-A3 rescore and delta census; .best.pt: no reader (one is the contaminated .stale) | 25 | KEEP-SMALL-ONLY* | 1b best.pt + .stale; 3 final.pt |
| `code_decay` | 2.56 (2.56) | CODE_FULLVAL_RESULTS (C2, envelope); CODE_DECAY_RESULTS (withdrawn OOD) | CITED | .final.pt: full-val rescore; .best.pt: no reader | 26 | KEEP-SMALL-ONLY* | 1b best.pt; 3 final.pt |
| `code` | 2.56 (2.56) | CODE_FULLVAL_RESULTS (c512), ABLATE_RESULTS, CODE_RESULTS (withdrawn) | CITED | .final.pt: full-val rescore, eval_code_delims; .best.pt: probe_code_depth0/accum (CODE_RESULTS, withdrawn OOD) | 25 | KEEP-SMALL-ONLY* | 2 best.pt; 3 final.pt |
| `code2048` | 2.56 (2.56) | CODE_FULLVAL_RESULTS (C1); draft 2 App. J4; CODE_RESULTS (withdrawn OOD) | CITED | .final.pt: full-val rescore; .best.pt: usage_tick.py / tw_alpha_drift.py (draft 2 App. J3-J4), alpha_language.py, code_step_table.py | 26 | KEEP-CHECKPOINTS | - |
| `indirect_t3` | 1.76 (1.76) | T3 per-token phase on Indirect (T3_RESULTS / LOG 2026-09-15..20) | REP | none documented | 16 | KEEP-SMALL-ONLY | 2 |
| `indirect_200k` | 1.51 (1.51) | INDIRECT_RESULTS_200k (cited replication), INDIRECT_OOD | CITED | eval_indirect_ood.py (INDIRECT_OOD, eval-only), pope_task_step_table.py | 17 | KEEP-SMALL-ONLY | 2 |
| `jsb_len512` | 0.72 (0.71) | JSB_LENGTH_RESULTS (OOD; collapse account withdrawn), T1 | ROB | analyze_t1.py, probe_theory_numbers.py | 41 | KEEP-SMALL-ONLY | 2 |
| `tinystories` | 0.40 (0.40) | TINYSTORIES_PILOT_RESULTS (pilot; three claims withdrawn 2026-10-11); draft 2 Sec. 9 / App. J4 | CITED | .best.pt: usage_tick.py, tw_alpha_drift.py (draft 2 App. J4), alpha_language.py, tinystories_probe.py; .final.pt: no reader | 13 | KEEP-SMALL-ONLY* | 2 final.pt; keep best.pt |
| `jsb` | 0.36 (0.36) | JSB_RESULTS (Bach replication, cited) | CITED | probe_theory_numbers.py (THEORY_MAPPOPE, account withdrawn), pope_task_step_table.py | 21 | KEEP-SMALL-ONLY | 2 |
| `jsb_forced` | 0.21 (0.21) | T3_RESULTS / T3GEN_RESULTS (superseded as cross-task claim) | REP | probe_theory_numbers.py | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_t3` | 0.21 (0.21) | T3_RESULTS / T3GEN_RESULTS (superseded as cross-task claim) | REP | probe_theory_numbers.py | 10 | KEEP-SMALL-ONLY | 2 |
| `sameblock` | 0.19 (0.19) | SAMEBLOCK_RESULTS | REP | analyze_sameblock.py (registered analysis loads checkpoints) | 13 | KEEP-SMALL-ONLY | 2 |
| `jsb_cross` | 0.18 (0.18) | DECAY_RESULTS / CROSS line on Bach (LOG 2026-09-15..20) | REP | none documented | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_decay` | 0.18 (0.18) | DECAY_RESULTS / CROSS line on Bach (LOG 2026-09-15..20) | REP | none documented | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_centered` | 0.18 (0.18) | T1_RESULTS (centred accumulator) | REP | analyze_t1.py | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_aug` | 0.18 (0.18) | AUG_RESULTS (cited: Bach overfitting-limited 3x) | CITED | none documented | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_base8192` | 0.18 (0.18) | JSB_LENGTH_RESULTS_BASE (omega-base sweep, OOD) | ROB | none (analyze_jsb_base reads JSON) | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_base512` | 0.18 (0.18) | JSB_LENGTH_RESULTS_BASE (omega-base sweep, OOD) | ROB | none (analyze_jsb_base reads JSON) | 10 | KEEP-SMALL-ONLY | 2 |
| `jsb_base32768` | 0.18 (0.18) | JSB_LENGTH_RESULTS_BASE (omega-base sweep, OOD) | ROB | none (analyze_jsb_base reads JSON) | 10 | KEEP-SMALL-ONLY | 2 |
| `addition_pilot` | 0.09 (0.09) | ADDITION_PILOT*.md, ADDITION_CHO_REPRO.md (line not pursued) | PILOT | none | 21 | DROP-CANDIDATE | 1 |
| `addition_pilot2` | 0.08 (0.08) | ADDITION_PILOT*.md, ADDITION_CHO_REPRO.md (line not pursued) | PILOT | none | 19 | DROP-CANDIDATE | 1 |
| `addition_cho` | 0.03 (0.03) | ADDITION_PILOT*.md, ADDITION_CHO_REPRO.md (line not pursued) | PILOT | none | 3 | DROP-CANDIDATE | 1 |

### Dyck-2 -- 9 dirs, 0.87 GB (0.86 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `dyck_mdepth` | 0.49 (0.48) | DYCK_MDEPTH_RESULTS (cited) | CITED | analyze_dyck_mdepth.py re-scores every checkpoint; dyck_step_table/distance/crosstab; runs_rescore/dyck_mdepth.json | 211 | KEEP-CHECKPOINTS | - |
| `dyck_ladder` | 0.24 (0.24) | DYCK_LADDER_RESULTS (cited; 4L +0.168 withdrawn as capability) | CITED | analyze_dyck_ladder.py / analyze_dyck_mdepth.py | 129 | KEEP-CHECKPOINTS | - |
| `dyck_depth` | 0.06 (0.06) | DYCK_DEPTH_RESULTS (base confound settled) | REP | analyze_dyck_depth.py | 81 | KEEP-CHECKPOINTS | - |
| `dyck_bs128` | 0.05 (0.05) | DYCK_RESULTS_bs128, DYCK_LITERATURE_METRICS (cited) | CITED | eval_dyck_literature.py, probe_dyck_stack/far/alpha | 89 | KEEP-CHECKPOINTS | - |
| `dyck_t2` | 0.01 (0.01) | T2_RESULTS (accumulator+kernel account WITHDRAWN) | WD | probe_dyck_alpha.py | 40 | DROP-CANDIDATE | 1 |
| `dyck_t3` | 0.01 (0.01) | T3GEN G1 (DYCK_T3_RESULTS is an empty artefact) | REP | none | 24 | KEEP-SMALL-ONLY | 2 |
| `dyck_cross` | 0.01 (<0.01) | CROSS_RESULTS (cited: decay envelope) | CITED | none documented | 24 | KEEP-CHECKPOINTS | - |
| `dyck_decay` | <0.01 (<0.01) | CROSS_RESULTS (cited: decay envelope) | CITED | none documented | 16 | KEEP-CHECKPOINTS | - |
| `dyck_lam` | <0.01 (<0.01) | CROSS_RESULTS (cited: decay envelope) | CITED | none documented | 16 | KEEP-CHECKPOINTS | - |

### MiniWorld (map-extent boundary) -- 16 dirs, 2.88 GB (2.88 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `mw_grid_sweep` | 0.61 (0.61) | MINIWORLD_GRID_SWEEP(_HIER), MINIWORLD_HIER_ABLATION | REP | experiment_audit.py uses mw_grid_sweep as its documented example | 51 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `miniworld_oracle` | 0.28 (0.28) | MINIWORLD_FIXED/FRESH/ORACLE_*, MINIWORLD_RESULTS | REP | agg_miniworld.py (losses), eval_ablation_miniworld.py (context destruction) | 24 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `miniworld_fresh` | 0.28 (0.28) | MINIWORLD_FIXED/FRESH/ORACLE_*, MINIWORLD_RESULTS | REP | agg_miniworld.py (losses), eval_ablation_miniworld.py (context destruction) | 24 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `miniworld_fixed` | 0.28 (0.28) | MINIWORLD_FIXED/FRESH/ORACLE_*, MINIWORLD_RESULTS | REP | agg_miniworld.py (losses), eval_ablation_miniworld.py (context destruction) | 24 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `alias_sweep` | 0.28 (0.28) | ALIASING_CONTROLLED, MINIWORLD_ENDPOINTS, CROSSOVER_CONVERGED (cited: map-extent threshold) | CITED | agg_alias.py reads training losses from the .pt (no loss in the JSON) | 23 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `alias_follow` | 0.22 (0.22) | ALIASING_CONTROLLED, MINIWORLD_ENDPOINTS, CROSSOVER_CONVERGED (cited: map-extent threshold) | CITED | agg_alias.py reads training losses from the .pt (no loss in the JSON) | 18 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `mw_hier_ablation` | 0.21 (0.21) | MINIWORLD_GRID_SWEEP(_HIER), MINIWORLD_HIER_ABLATION | REP | none documented (losses in the .pt: P2) | 24 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `mw_grid_sweep_hier` | 0.21 (0.21) | MINIWORLD_GRID_SWEEP(_HIER), MINIWORLD_HIER_ABLATION | REP | none documented (losses in the .pt: P2) | 24 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `visits_test` | 0.15 (0.15) | VISITS_TEST (cited boundary) | CITED | agg_visits.py loads .pt (losses) | 12 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `rope_converge` | 0.08 (0.08) | ALIASING_CONTROLLED, MINIWORLD_ENDPOINTS, CROSSOVER_CONVERGED (cited: map-extent threshold) | CITED | agg_alias.py reads training losses from the .pt (no loss in the JSON) | 7 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `g8_converge` | 0.07 (0.07) | ALIASING_CONTROLLED, MINIWORLD_ENDPOINTS, CROSSOVER_CONVERGED (cited: map-extent threshold) | CITED | agg_alias.py reads training losses from the .pt (no loss in the JSON) | 6 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `miniworld` | 0.05 (0.05) | MINIWORLD_FIXED/FRESH/ORACLE_*, MINIWORLD_RESULTS | REP | agg_miniworld.py (losses), eval_ablation_miniworld.py (context destruction) | 24 | KEEP-SMALL-ONLY | 2 (after loss extraction) |
| `mw_fixed_diag` | 0.05 (0.05) | diagnostics / pilots of the MiniWorld line | PILOT | none | 4 | DROP-CANDIDATE | 1 |
| `mw_grid_pilot` | 0.05 (0.05) | diagnostics / pilots of the MiniWorld line | PILOT | none | 4 | DROP-CANDIDATE | 1 |
| `mw_scale_diag` | 0.02 (0.02) | diagnostics / pilots of the MiniWorld line | PILOT | none | 2 | DROP-CANDIDATE | 1 |
| `mw_fresh_probe` | 0.02 (0.02) | diagnostics / pilots of the MiniWorld line | PILOT | none | 2 | DROP-CANDIDATE | 1 |

### MiniGrid -- 3 dirs, 0.25 GB (0.25 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `minigrid_em_fix` | 0.09 (0.09) | MINIGRID_EM (report.tex), THEORY_KERNEL | REP | none documented | 0 | KEEP-SMALL-ONLY | 2 (after results extraction) |
| `minigrid_em` | 0.09 (0.09) | MINIGRID_EM (report.tex), THEORY_KERNEL | REP | none documented | 0 | KEEP-SMALL-ONLY | 2 (after results extraction) |
| `minigrid_2x2x2_allo` | 0.06 (0.06) | MINIGRID_ALLOCENTRIC_2X2X2 | REP | none documented | 0 | KEEP-SMALL-ONLY | 2 |

### Borrowed benchmarks -- 1 dirs, 0.57 GB (0.57 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `flipflop` | 0.57 (0.57) | FLIPFLOP_RESULTS / GATES (live negative: does not test our axis) | NEG | none | 48 | KEEP-SMALL-ONLY | 2 (after loss extraction) |

### Algorithmic tasks (parity, copy, frontier) -- 6 dirs, 0.76 GB (0.75 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `frontier_alg` | 0.34 (0.34) | FRONTIER_ALGORITHMIC, ALGORITHMIC_RESULTS | REP | none (agg_* read JSON) | 256 | KEEP-SMALL-ONLY | 2 |
| `hier_parity` | 0.14 (0.14) | HIER_PARITY, LOOP_HIER_PARITY (hierarchy unpowered) | REP | none | 64 | KEEP-SMALL-ONLY | 2 |
| `selective` | 0.11 (0.11) | SELECTIVE_ROPE (live negative) | NEG | probe_rank_learned/truncate.py, probe_gate.py, probe_conv_kernel.py | 96 | KEEP-SMALL-ONLY | 2 |
| `loop_hier` | 0.07 (0.07) | HIER_PARITY, LOOP_HIER_PARITY (hierarchy unpowered) | REP | none | 48 | KEEP-SMALL-ONLY | 2 |
| `algorithmic` | 0.06 (0.06) | FRONTIER_ALGORITHMIC, ALGORITHMIC_RESULTS | REP | none (agg_* read JSON) | 80 | KEEP-SMALL-ONLY | 2 |
| `rope_canon` | 0.02 (0.02) | ROPE_CANONICAL | REP | none | 32 | KEEP-SMALL-ONLY | 2 |

### Match-Query loop line -- 3 dirs, 0.08 GB (0.08 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `loop_headroom` | 0.04 (0.04) | LOOP_HEADROOM, REFINE_RESULTS, MQ_RANK_2X2 (cited: loop x4) | CITED | loop_depth_strata.py; agg_refine.py loads .pt | 40 | KEEP-CHECKPOINTS | - |
| `mq_rank` | 0.03 (0.02) | LOOP_HEADROOM, REFINE_RESULTS, MQ_RANK_2X2 (cited: loop x4) | CITED | loop_depth_strata.py; agg_refine.py loads .pt | 32 | KEEP-CHECKPOINTS | - |
| `refine` | 0.01 (0.01) | LOOP_HEADROOM, REFINE_RESULTS, MQ_RANK_2X2 (cited: loop x4) | CITED | loop_depth_strata.py; agg_refine.py loads .pt | 16 | KEEP-CHECKPOINTS | - |

### Compositional Match-Query -- 5 dirs, 0.27 GB (0.27 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `cmq_curric` | 0.07 (0.07) | COMPOSITIONAL_MATCH_QUERY(_CURRIC/_STAB) | REP | none | 30 | KEEP-SMALL-ONLY | 2 |
| `cmq_stab` | 0.07 (0.07) | COMPOSITIONAL_MATCH_QUERY(_CURRIC/_STAB) | REP | none | 30 | KEEP-SMALL-ONLY | 2 |
| `comp_headroom` | 0.06 (0.06) | COMP_HEADROOM, HIER_RECHECK (cited) | CITED | none (agg reads JSON) | 32 | KEEP-CHECKPOINTS | - |
| `hier_recheck` | 0.04 (0.04) | COMP_HEADROOM, HIER_RECHECK (cited) | CITED | none (agg reads JSON) | 16 | KEEP-CHECKPOINTS | - |
| `cmq_sweep` | 0.03 (0.03) | COMPOSITIONAL_MATCH_QUERY(_CURRIC/_STAB) | REP | none | 15 | KEEP-SMALL-ONLY | 2 |

### Level 1.5 / looped negatives -- 7 dirs, 0.27 GB (0.27 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `mq_noise_c2` | 0.06 (0.06) | MQ_NOISE_2X2*, L15_LOOP_2X2, L15_ABLATION, NOISE_REFINE, LOOP_SAMPLED (live negatives) | NEG | ablate_match_noise.py, eval_noise_refine.py | 64 | KEEP-SMALL-ONLY | 2 |
| `mq_noise` | 0.06 (0.06) | MQ_NOISE_2X2*, L15_LOOP_2X2, L15_ABLATION, NOISE_REFINE, LOOP_SAMPLED (live negatives) | NEG | ablate_match_noise.py, eval_noise_refine.py | 64 | KEEP-SMALL-ONLY | 2 |
| `l15_loop_2x2` | 0.05 (0.05) | MQ_NOISE_2X2*, L15_LOOP_2X2, L15_ABLATION, NOISE_REFINE, LOOP_SAMPLED (live negatives) | NEG | ablate_match_noise.py, eval_noise_refine.py | 0 | KEEP-SMALL-ONLY | 2 |
| `noise_refine` | 0.03 (0.03) | MQ_NOISE_2X2*, L15_LOOP_2X2, L15_ABLATION, NOISE_REFINE, LOOP_SAMPLED (live negatives) | NEG | ablate_match_noise.py, eval_noise_refine.py | 0 | KEEP-SMALL-ONLY | 2 |
| `looped_pilot` | 0.03 (0.03) | LOOPED_PILOT / HORIZON_RESULTS (WITHDRAWN: "scale hurts") | WD | none | 0 | DROP-CANDIDATE | 1 |
| `l15_ablation` | 0.03 (0.03) | MQ_NOISE_2X2*, L15_LOOP_2X2, L15_ABLATION, NOISE_REFINE, LOOP_SAMPLED (live negatives) | NEG | ablate_match_noise.py, eval_noise_refine.py | 0 | KEEP-SMALL-ONLY | 2 |
| `loop_sampled` | 0.02 (0.02) | MQ_NOISE_2X2*, L15_LOOP_2X2, L15_ABLATION, NOISE_REFINE, LOOP_SAMPLED (live negatives) | NEG | ablate_match_noise.py, eval_noise_refine.py | 0 | KEEP-SMALL-ONLY | 2 |

### Older torus / recency batches (pre-registration era) -- 13 dirs, 0.31 GB (0.31 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `popewrap` | 0.05 (0.05) | POPE_WRAPPING (robustness only) | ROB | none | 8 | KEEP-SMALL-ONLY | 2 |
| `recipe_power` | 0.04 (0.04) | RECIPE_POWER | REP | none | 0 | KEEP-SMALL-ONLY | 2 |
| `torus_t3` | 0.03 (0.03) | TORUS_T3_RESULTS; RECENCY_T3_RESULTS (VOID for ceiling) | WD | none | 0 | DROP-CANDIDATE | 1 |
| `gated_recency` | 0.03 (0.03) | GATED_RESULTS (live negative), RECENCY_GATE_ABLATION (cited) | CITED | ablate_recency_gate.py, probe_gate_separation.py | 32 | KEEP-CHECKPOINTS | - |
| `forget` | 0.03 (0.02) | FORGET_GATE, FORGET_CONTROL, LAMBDA_TRACE (live negative) | NEG | none | 0 | KEEP-SMALL-ONLY | 2 |
| `gated_torus` | 0.02 (0.02) | GATED_RESULTS (live negative), RECENCY_GATE_ABLATION (cited) | CITED | ablate_recency_gate.py, probe_gate_separation.py | 0 | KEEP-CHECKPOINTS | - |
| `mappope_r4` | 0.02 (0.02) | MAPPOPE_R4_RESULTS (OOD, unmatched init) | ROB | none | 0 | KEEP-SMALL-ONLY | 2 |
| `recency_t3` | 0.02 (0.02) | TORUS_T3_RESULTS; RECENCY_T3_RESULTS (VOID for ceiling) | WD | none | 24 | DROP-CANDIDATE | 1 |
| `forget_ctrl` | 0.02 (0.02) | FORGET_GATE, FORGET_CONTROL, LAMBDA_TRACE (live negative) | NEG | none | 0 | KEEP-SMALL-ONLY | 2 |
| `paper_task_rerun` | 0.02 (0.02) | PAPER_OOD_RERUN / PAPERTASK_PREREG | REP | analyze_papertask.py, probe_collisions.py | 2 | KEEP-SMALL-ONLY | 2 |
| `mappope_recency` | 0.01 (0.01) | MAPPOPE_R4_RESULTS (OOD, unmatched init) | ROB | none | 16 | KEEP-SMALL-ONLY | 2 |
| `em_fig4` | 0.01 (0.01) | PAPER_FIG4_EM; GAIN_GRAIN power check | REP | gain_grain_power.py | 0 | KEEP-SMALL-ONLY | 2 |
| `lamtrace` | 0.01 (0.01) | FORGET_GATE, FORGET_CONTROL, LAMBDA_TRACE (live negative) | NEG | none | 8 | KEEP-SMALL-ONLY | 2 |

### Rank, withdrawn -- 1 dirs, 0.05 GB (0.05 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `dxr` | 0.05 (0.05) | DXR_RANK_THRESHOLD (WITHDRAWN packing account) | WD | none | 0 | DROP-CANDIDATE | 1 |

### Rank, older -- 1 dirs, 0.03 GB (0.03 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `rank_sweep` | 0.03 (0.03) | RANK_SWEEP (r=4 +0.085 OOD-only, correction cited), PAPER_FIG4_REPRO | CITED | probe_paper_fig4.py, probe_drift.py, probe_rank_projection.py | 0 | KEEP-CHECKPOINTS | - |

### EM / recency line -- 7 dirs, 0.22 GB (0.22 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `vocab_em` | 0.10 (0.10) | VOCAB_EM.md | REP | none documented | 0 | KEEP-SMALL-ONLY | 2 (after results extraction) |
| `n5_phase` | 0.05 (0.05) | N5_RESULTS | REP | none documented | 34 | KEEP-SMALL-ONLY | 2 |
| `counter_batch` | 0.02 (0.02) | COUNTER_BATCH / COUNTER_RESULTS | REP | none documented | 21 | KEEP-SMALL-ONLY | 2 |
| `tem_recency_diag` | 0.02 (0.02) | TEM recency diagnostics | PILOT | none | 7 | DROP-CANDIDATE | 1 |
| `tem_recency_pilot` | 0.01 (0.01) | TEM recency diagnostics | PILOT | none | 6 | DROP-CANDIDATE | 1 |
| `recency_budget` | 0.01 (0.01) | RECENCY budget check (RECENCY_RESULTS line) | REP | none documented | 9 | KEEP-SMALL-ONLY | 2 |
| `vocab_em_link` | <0.01 (<0.01) | VOCAB_EM.md | REP | none documented | 0 | KEEP-SMALL-ONLY | 2 (after results extraction) |

### Registered torus / ring / rank / sign / EM batches (the current paper) -- 64 dirs, 3.11 GB (3.06 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `cancel` | 0.17 (0.17) | CANCEL_RESULTS (H3) | CITED | registered analysis and/or documented probes load the checkpoints | 129 | KEEP-CHECKPOINTS | - |
| `dof` | 0.15 (0.15) | SEARCH/SPREAD/DOF; test_guards fixtures | CITED | registered analysis and/or documented probes load the checkpoints | 146 | KEEP-CHECKPOINTS | - |
| `robust` | 0.15 (0.14) | ROBUST_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 65 | KEEP-CHECKPOINTS | - |
| `ssm_sign` | 0.14 (0.14) | G3_SSM_RESULTS, G3_SSM_A2_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `accounts` | 0.10 (0.10) | ACCOUNTS_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 164 | KEEP-CHECKPOINTS | - |
| `generality` | 0.10 (0.10) | GENERALITY_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 49 | KEEP-CHECKPOINTS | - |
| `score_rank` | 0.09 (0.09) | SCORE_RANK_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `loop_rank` | 0.09 (0.09) | LOOP_RANK_RESULTS (H1) | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `gain_grain` | 0.09 (0.09) | GAIN_GRAIN_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `rnn_torus` | 0.09 (0.09) | RNN_TORUS_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `monotone` | 0.09 (0.08) | MONOTONE_RESULTS (clock/map) | CITED | registered analysis and/or documented probes load the checkpoints | 73 | KEEP-CHECKPOINTS | - |
| `rank_wrap` | 0.08 (0.07) | RANK_WRAP_RESULTS; what-where 3D probe | CITED | registered analysis and/or documented probes load the checkpoints | 9 | KEEP-CHECKPOINTS | - |
| `rank_nd` | 0.08 (0.07) | RANK_ND_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `sign_matched` | 0.07 (0.07) | SIGN_MATCHED_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `pairsplit` | 0.07 (0.07) | PAIRSPLIT_RESULTS (cited) | CITED | registered analysis and/or documented probes load the checkpoints | 84 | KEEP-CHECKPOINTS | - |
| `signsep` | 0.06 (0.06) | SIGNSEP_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `rank_tests` | 0.06 (0.06) | RANK_TESTS_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 41 | KEEP-CHECKPOINTS | - |
| `mappope_pair` | 0.06 (0.06) | MAPPOPE_PAIR_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `sign` | 0.06 (0.06) | SIGN_ABLATION, PAPER2X2, MONOTONE | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `exptorus` | 0.05 (0.05) | EXPTORUS_RESULTS (exp census) | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `wrap_mech` | 0.05 (0.05) | MECHANISM_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 65 | KEEP-CHECKPOINTS | - |
| `wrap_existence` | 0.05 (0.05) | WRAP_EXISTENCE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 57 | KEEP-CHECKPOINTS | - |
| `lrsweep` | 0.05 (0.05) | LRSWEEP_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 65 | KEEP-CHECKPOINTS | - |
| `image_torus` | 0.05 (0.05) | IMAGE_TORUS_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `recency` | 0.04 (0.04) | RECENCY_RESULTS (crossover) | CITED | registered analysis and/or documented probes load the checkpoints | 48 | KEEP-CHECKPOINTS | - |
| `wrap_cope` | 0.04 (0.04) | WRAP_COPE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 33 | KEEP-CHECKPOINTS | - |
| `gain_phase` | 0.04 (0.04) | GAIN_PHASE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 33 | KEEP-CHECKPOINTS | - |
| `srope_torus` | 0.04 (0.04) | SROPE_TORUS_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `torus_lrsweep` | 0.04 (0.04) | TORUS_LRSWEEP_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `loop_rank_e1800` | 0.04 (0.04) | LOOP_RANK_E1800_P1_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `ring_dial` | 0.04 (0.04) | RING_DIAL_RESULTS (rule-14 construction) | CITED | registered analysis and/or documented probes load the checkpoints | 49 | KEEP-CHECKPOINTS | - |
| `paper2x2` | 0.04 (0.04) | PAPER2X2_RESULTS; what-where checks (docs/WHAT_WHERE_CHECKS.md) | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `rank_sep` | 0.04 (0.04) | RANK_SEP_RESULTS, RANK3 | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `rank_matched_e900c` | 0.04 (0.04) | RANK_MATCHED_RESULTS (withdrawn claims) | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `clock_speed` | 0.04 (0.04) | CLOCK_SPEED_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 41 | KEEP-CHECKPOINTS | - |
| `clock_side` | 0.04 (0.04) | CLOCK_SIDE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 41 | KEEP-CHECKPOINTS | - |
| `unfreeze` | 0.04 (0.04) | UNFREEZE/NOLEAK; test_guards fixture | CITED | registered analysis and/or documented probes load the checkpoints | 41 | KEEP-CHECKPOINTS | - |
| `rank_mi` | 0.03 (0.03) | RANK_MI_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 11 | KEEP-CHECKPOINTS | - |
| `torus_start` | 0.03 (0.03) | TORUS_START_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 41 | KEEP-CHECKPOINTS | - |
| `leak` | 0.03 (0.03) | LEAK_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 25 | KEEP-CHECKPOINTS | - |
| `delta_free` | 0.03 (0.03) | DELTA_FREE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `gauge` | 0.03 (0.03) | GAUGE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 41 | KEEP-CHECKPOINTS | - |
| `clock_speed_rep` | 0.03 (0.03) | CLOCK_SPEED_REP_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 33 | KEEP-CHECKPOINTS | - |
| `spread` | 0.03 (0.03) | SPREAD_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 36 | KEEP-CHECKPOINTS | - |
| `search` | 0.03 (0.03) | SEARCH_RESULTS (cited) | CITED | registered analysis and/or documented probes load the checkpoints | 42 | KEEP-CHECKPOINTS | - |
| `spread2` | 0.03 (0.03) | SPREAD2_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 34 | KEEP-CHECKPOINTS | - |
| `tw_statechange` | 0.03 (0.03) | TW_STATECHANGE_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 33 | KEEP-CHECKPOINTS | - |
| `tw_normstep` | 0.03 (0.03) | TW_NORMSTEP_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 33 | KEEP-CHECKPOINTS | - |
| `textworld` | 0.03 (0.03) | TEXTWORLD_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 25 | KEEP-CHECKPOINTS | - |
| `factorial_n8` | 0.02 (0.02) | FACTORIAL_N8_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `factorial` | 0.02 (0.02) | FACTORIAL_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `pairorigin` | 0.02 (0.02) | PAIRORIGIN_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 27 | KEEP-CHECKPOINTS | - |
| `recency_em` | 0.02 (0.02) | RECENCY_EM; test_guards fixture | CITED | registered analysis and/or documented probes load the checkpoints | 25 | KEEP-CHECKPOINTS | - |
| `rank3` | 0.02 (0.02) | RANK3_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `rank_proj_train` | 0.02 (0.02) | RANK_PROJ_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `noleak` | 0.01 (0.01) | NOLEAK_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 17 | KEEP-CHECKPOINTS | - |
| `warm` | 0.01 (0.01) | WARM_RESULTS; test_guards fixture | CITED | registered analysis and/or documented probes load the checkpoints | 17 | KEEP-CHECKPOINTS | - |
| `rank_matched_e900` | 0.01 (0.01) | RANK_MATCHED_RESULTS; experiment_audit fixture; symlink target of rank_mi/p0 | CITED | registered analysis and/or documented probes load the checkpoints | 1 | KEEP-CHECKPOINTS | - |
| `ring_budget` | 0.01 (0.01) | RING_BUDGET_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 17 | KEEP-CHECKPOINTS | - |
| `rank_matched` | 0.01 (0.01) | RANK_MATCHED_RESULTS (claims withdrawn; GUARDS fixture) | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `pairconst` | 0.01 (0.01) | PAIRCONST_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 12 | KEEP-CHECKPOINTS | - |
| `rank_proj` | 0.01 (0.01) | RANK_PROJ_RESULTS | CITED | registered analysis and/or documented probes load the checkpoints | 0 | KEEP-CHECKPOINTS | - |
| `clock_side_calib` | 0.01 (0.01) | CLOCK_SIDE calibration (docs/audits/2026-10-12/clock_side_audit.md) | REP | clock_side_calib.sh | 6 | KEEP-CHECKPOINTS | - |
| `spread_cal` | 0.01 (0.01) | SPREAD2 calibration | CITED | registered analysis and/or documented probes load the checkpoints | 7 | KEEP-CHECKPOINTS | - |

### Pilots and repro dirs that a registered analysis, guard or symlink reads -- 21 dirs, 0.21 GB (0.21 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `cancel_pilot` | 0.03 (0.03) | - | PILOT | CANCEL_RESULTS reports the pilot seeds | 17 | KEEP-CHECKPOINTS | - |
| `ssm_sign_pilot` | 0.03 (0.03) | - | PILOT | gate_ssm_sign.py / c3_built.py | 26 | KEEP-CHECKPOINTS | - |
| `rank_nowrap_pilot` | 0.03 (0.03) | - | PILOT | RANK_NOWRAP pilots (docs/audits/2026-10-06 repro); tracked symlinks | 55 | KEEP-CHECKPOINTS | - |
| `score_rank_pilot` | 0.03 (0.03) | - | PILOT | analyze_score_rank.py repro check | 0 | KEEP-CHECKPOINTS | - |
| `ctxstep2_pilot` | 0.02 (0.02) | - | PILOT | swap_ctxstep2.py | 17 | KEEP-CHECKPOINTS | - |
| `rnn_torus_pilot` | 0.02 (0.02) | - | PILOT | analyze_rnn_torus.py pilot() | 0 | KEEP-CHECKPOINTS | - |
| `ctxstep3_pilot` | 0.01 (0.01) | - | PILOT | gate_tw_ambig.py | 13 | KEEP-CHECKPOINTS | - |
| `gain_phase_pilot` | 0.01 (0.01) | - | PILOT | gain_phase_lossgate_validate.py | 25 | KEEP-CHECKPOINTS | - |
| `newobj_pilot` | 0.01 (0.01) | - | PILOT | e0_leak_decomposition.py / e0_step_geometry.py (cited in WHERE_THINGS_STAND) | 7 | KEEP-CHECKPOINTS | - |
| `rank_perhead_pilot` | 0.01 (0.01) | - | PILOT | symlink target of rank_mi/p0 (cited); analyze_rank_perhead.py | 1 | KEEP-CHECKPOINTS | - |
| `gain_grain_pilot` | 0.01 (0.01) | - | PILOT | analyze_gain_grain.py repro check | 2 | KEEP-CHECKPOINTS | - |
| `delta_free_pilot` | <0.01 (<0.01) | - | PILOT | pilot_decay_check.py | 5 | KEEP-CHECKPOINTS | - |
| `wrap_existence_pilot` | <0.01 (<0.01) | - | PILOT | analyze_wrap_existence.py | 6 | KEEP-CHECKPOINTS | - |
| `exptorus_pilot` | <0.01 (<0.01) | - | PILOT | analyze_exptorus.py | 0 | KEEP-CHECKPOINTS | - |
| `rank3_repro` | <0.01 (<0.01) | - | PILOT | analyze_rank3.py | 0 | KEEP-CHECKPOINTS | - |
| `rank_sep_repro` | <0.01 (<0.01) | - | PILOT | analyze_rank_sep.py | 0 | KEEP-CHECKPOINTS | - |
| `loop_rank_repro` | <0.01 (<0.01) | - | PILOT | analyze_loop_rank.py | 0 | KEEP-CHECKPOINTS | - |
| `rank_mi_repro` | <0.01 (<0.01) | - | PILOT | analyze_rank_mi.py | 0 | KEEP-CHECKPOINTS | - |
| `magonly_repro` | <0.01 (<0.01) | - | PILOT | test_guards.py fixture | 3 | KEEP-CHECKPOINTS | - |
| `noleak_repro` | <0.01 (<0.01) | - | PILOT | analyze_noleak.py | 2 | KEEP-CHECKPOINTS | - |
| `p0_extend_repro` | <0.01 (<0.01) | - | PILOT | determinism pair | 2 | KEEP-CHECKPOINTS | - |

### Pilots with no reader -- 22 dirs, 0.16 GB (0.16 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `tw_landmark_pilot` | 0.03 (0.03) | prereg pilot section | PILOT | none | 21 | DROP-CANDIDATE | 1 |
| `hsr_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 9 | DROP-CANDIDATE | 1 |
| `hs_recipe_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 9 | DROP-CANDIDATE | 1 |
| `robust_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 0 | DROP-CANDIDATE | 1 |
| `image_torus_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 0 | DROP-CANDIDATE | 1 |
| `signsep_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 13 | DROP-CANDIDATE | 1 |
| `lrsweep_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 13 | DROP-CANDIDATE | 1 |
| `ctxstep_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 9 | DROP-CANDIDATE | 1 |
| `factorial_n8_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 0 | DROP-CANDIDATE | 1 |
| `torus_exist_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 3 | DROP-CANDIDATE | 1 |
| `rank_tests_pilot` | 0.01 (0.01) | prereg pilot section | PILOT | none | 4 | DROP-CANDIDATE | 1 |
| `image_torus_pilot2` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 0 | DROP-CANDIDATE | 1 |
| `tw_statechange_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 6 | DROP-CANDIDATE | 1 |
| `tw_normstep_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 6 | DROP-CANDIDATE | 1 |
| `textworld_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 5 | DROP-CANDIDATE | 1 |
| `generality_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 2 | DROP-CANDIDATE | 1 |
| `signsep_pilot2` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 5 | DROP-CANDIDATE | 1 |
| `mappope_pair_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 2 | DROP-CANDIDATE | 1 |
| `ring_dial_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 4 | DROP-CANDIDATE | 1 |
| `wrap_cope_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 2 | DROP-CANDIDATE | 1 |
| `gauge_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 3 | DROP-CANDIDATE | 1 |
| `srope_torus_pilot` | <0.01 (<0.01) | prereg pilot section | PILOT | none | 5 | DROP-CANDIDATE | 1 |

### Smoke tests -- 2 dirs, 0.02 GB (0.02 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `exptorus_smoke` | 0.01 (0.01) | smoke tests | PILOT | none | 0 | DROP-CANDIDATE | 1 |
| `_sign_smoke` | <0.01 (<0.01) | smoke tests | PILOT | none | 1 | DROP-CANDIDATE | 1 |

### Misc (no checkpoints) -- 4 dirs, <0.01 GB (0 GB checkpoints)

| dir | size GB (ckpt) | backs | status | checkpoints needed for | git-tracked files | class | tier |
|---|---|---|---|---|---|---|---|
| `toy_wrap` | <0.01 (0) | logs / toy grid (no checkpoints) | REP | none | 145 | KEEP-SMALL-ONLY | - |
| `gated_logs` | <0.01 (0) | logs / toy grid (no checkpoints) | REP | none | 0 | KEEP-SMALL-ONLY | - |
| `mappope_logs` | <0.01 (0) | logs / toy grid (no checkpoints) | REP | none | 0 | KEEP-SMALL-ONLY | - |
| `ctxstep` | <0.01 (0) | logs / toy grid (no checkpoints) | REP | none | 1 | KEEP-SMALL-ONLY | - |

### Legacy, pre-August 2026 (lm200 era, hier-goal, Level 1.5 variants, TEM, DoG/cnav) -- 228 dirs, 0.59 GB (0.59 GB checkpoints)

All 228 dirs dated before 2026-08-01 (mostly `Level15*`, `Vanilla*`, `RoPE*`, `TEM*`, `PC*`, `LSTM*`, `MambaLike*`, `CoPE*`, `*_lm200`, `*_goal_*`, `*_dagger`/`*_bc`, `minigrid_*_cached`, `cnav`, plus `route recursive fresh repro bump boundedtest aggtest rgoal rmaze vmaze roomstest hiertest comp_plain hier_multienv torus_transnoise fresh_lm200`). They hold only `.pt` files (nothing tracked, no JSON); their numbers live in the withdrawn lm200 tables (`CORRECTED_LM200_LEADERBOARD.md`, `DETAILED_RESULTS.md`) and `archive/void/`, `archive_stale/`. No script reads them. Largest: `fresh_lm200` 30 MB, `recursive` 17 MB, `route` 13 MB, `Level15EM_broken_init` 10 MB, `rgoal` 9 MB, `aggtest` 9 MB. Class DROP-CANDIDATE, tier 1, saves 0.59 GB.

### DUPLICATE checkpoints

An md5 of all 7,147 regular checkpoint files finds 244 hashes stored more than once: 529 files, of which **0.36 GB
is extra copies**. Almost every copy is deliberate:
- **Bitwise-reproduction evidence** (rules 6 and 19): `*_repro` dirs, `alias_sweep/n16_repro` vs `rope_converge`,
  `dof` vs `recency_em` vs `magonly_repro` (the `test_guards` determinism pairs), and `fresh` / `repro` vs the
  lm200 dirs.
- **Batches that reuse arms**: `dyck_ladder` reuses `dyck_depth` / `dyck_bs128`; `frontier_alg` reuses
  `algorithmic`, `selective`, `hier_parity` and `rope_canon`; `paper2x2` / `sign` / `gated_torus` / `mappope_r4`;
  `l15_ablation` / `l15_loop_2x2` / `noise_refine` / `loop_sampled`; `forget` / `forget_ctrl` / `lamtrace` /
  `popewrap`; `mw_grid_sweep_hier` vs `mw_hier_ablation` (46 MB, the largest); `minigrid_em` vs
  `minigrid_em_fix` (26 MB); `comp_headroom` vs `hier_recheck`.

Each batch's analysis expects its own path. Deleting one copy breaks the other batch, and for the repro pairs it
destroys the evidence. **Do not dedupe by deletion.** If 0.36 GB matters, replace the copies with hard links
(`rdfind -makehardlinks true` or `hardlink`, after a dry run). Not recommended at this size.

Symlinks are not duplicates. `vocab_em_link/` is 120 symlinks into `vocab_em/` (`du -L` / `find -L` count them
twice). `rank_mi/p0/` is 10 absolute symlinks into `rank_matched_e900/p0` and `rank_perhead_pilot/p0`.
`rank_nowrap_pilot/amend2/` has 16 internal symlinks, and `dof` / `n5_phase` each have one `torus_eval_stage/p0`.
No symlink dangles today.

---

## 3. Tiered deletion plan (checkpoints only, unless a line says otherwise)

The rule throughout: **delete checkpoint files, not directories.** The `.json`, `.log`, `code_md5.txt` and
`.done` files are kilobytes, and for many dirs they are untracked (section 4). The legacy dirs are the one
exception: they contain only checkpoints.

### Tier 1: nothing lost (repo: **3.83 GB**; outside the repo: up to 3.3 GB more)

**1a. Pilots / smoke / diagnostics with no reader, and void or withdrawn batches: 40 dirs, 0.68 GB.** Their
numbers are in tracked prereg / pilot / results files. No script reads their checkpoints.
```
runs/{addition_pilot,addition_pilot2,addition_cho}            0.20 GB  ADDITION line (not pursued)
runs/{mw_fixed_diag,mw_grid_pilot,mw_scale_diag,mw_fresh_probe}  0.14 GB  MiniWorld diagnostics
runs/dxr                0.05 GB  DXR_RANK_THRESHOLD (withdrawn); no JSON, nothing tracked
runs/tw_landmark_pilot  0.03 GB  TW_LANDMARK (stopped, prereg Amendment 5; docs/audits/2026-10-10/*.sh ran on it)
runs/torus_t3           0.03 GB  TORUS_T3 (superseded)
runs/looped_pilot       0.03 GB  LOOPED_PILOT (withdrawn); no JSON
runs/recency_t3         0.02 GB  RECENCY_T3 (VOID for ceiling)
runs/dyck_t2            0.01 GB  T2_RESULTS (account withdrawn; probe_dyck_alpha.py reads it, for that account only)
runs/{tem_recency_diag,tem_recency_pilot,hsr_pilot,hs_recipe_pilot,ctxstep_pilot,exptorus_smoke,_sign_smoke,
      robust_pilot,image_torus_pilot,image_torus_pilot2,signsep_pilot,signsep_pilot2,lrsweep_pilot,
      factorial_n8_pilot,torus_exist_pilot,rank_tests_pilot,tw_statechange_pilot,tw_normstep_pilot,
      textworld_pilot,generality_pilot,mappope_pair_pilot,ring_dial_pilot,wrap_cope_pilot,gauge_pilot,
      srope_torus_pilot}                                    0.17 GB  pilots
```
Command shape (after the section 5 checks): `find runs/<dir> -name '*.pt' -print0 | xargs -0 rm --`.
`exptorus_smoke`, `robust_pilot`, `image_torus_pilot*`, `factorial_n8_pilot`, `looped_pilot`, `dxr` and
`torus_t3` have untracked logs / JSON: keep those (or commit them first, section 4).

**1b. Legacy, pre-August 2026: 228 dirs, 0.59 GB.** These are the lm200 era (withdrawn), hier-goal and planner
tasks (withdrawn), Level 1.5 / PC / GSF / Cascade / TEM / DoG / cnav variants (archive/void) and early test dirs.
They hold only `.pt` files (nothing tracked, no JSON, no reader). Whole-dir removal is fine here. They carry no
markers, so `safe_clear.sh` accepts them one prompt at a time. Optional: tar them to cold storage first (0.6 GB),
because they are the raw evidence behind `CORRECTED_LM200_LEADERBOARD.md`.
```
runs/{CoPE_clean,CoPE_lm200,CoPE_noise,GridL15PC_Free_clean,Grid_Free_clean,Grid_clean,Grid_clean_200ep,L15_ConstR_clean}
runs/{L15_ConstR_lm200,L15_DARE_clean,L15_DARE_lm200,L15_NoCorr_clean,L15_NoCorr_lm200,L15_NoMeas_clean,L15_NoMeas_lm200}
runs/{LSTM_clean,LSTM_lm200,LSTM_noise,Level15Beta_clean,Level15Beta_lm200,Level15Beta_noise,Level15CascadeNoSlow_lm200}
runs/{Level15Cascade_clean,Level15Cascade_lm200,Level15Cascade_noise,Level15EM_PerScaleOmega_multisize,Level15EM_b5_lm200}
runs/{Level15EM_broken_init,Level15EM_clean,Level15EM_dagger,Level15EM_dk6_bc,Level15EM_dk6_dagger,Level15EM_dkbc}
runs/{Level15EM_empty_bc,Level15EM_empty_dagger,Level15EM_goal_lm200,Level15EM_goal_multistop,Level15EM_goal_noisy}
runs/{Level15EM_goal_switching,Level15EM_lm200,Level15EM_multisize,Level15EM_noise,Level15EM_vocab256,Level15EM_vocab4096}
runs/{Level15GSF_ModeOmega_multienv,Level15GSF_NoDrop_K16_multiclass,Level15GSF_NoDrop_K16_multisize}
runs/{Level15GSF_NoDrop_K16_topology,Level15GSF_NoDrop_ModeOmega_multienv,Level15GSF_NoDrop_ModeOmega_multisize}
runs/{Level15GSF_NoDrop_clean,Level15GSF_NoDrop_lm10,Level15GSF_NoDrop_lm200,Level15GSF_NoDrop_lm50}
runs/{Level15GSF_NoDrop_multienv,Level15GSF_NoDrop_multienv_clean,Level15GSF_NoDrop_multisize,Level15GSF_NoDrop_noise}
runs/{Level15GSF_NoDrop_topology,Level15GSF_clean,Level15GSF_lm200,Level15GSF_noise,Level15NoDrop_clean}
runs/{Level15NoDrop_dagger,Level15NoDrop_dk6_bc,Level15NoDrop_dk6_dagger,Level15NoDrop_dkbc,Level15NoDrop_empty_bc}
runs/{Level15NoDrop_empty_dagger,Level15NoDrop_goal_lm200,Level15NoDrop_goal_multistop,Level15NoDrop_goal_noisy}
runs/{Level15NoDrop_goal_switching,Level15NoDrop_lm200,Level15NoDrop_noise,Level15PC_NoBypass_clean}
runs/{Level15PC_NoBypass_lm200,Level15PC_clean,Level15PC_lm200,Level15PC_lm200_aux0.0,Level15PC_lm200_aux0.01}
runs/{Level15PC_lm200_aux0.03,Level15PC_lm200_aux0.1,Level15PC_lm200_aux0.3,Level15PC_noise,Level15PC_v3_clean}
runs/{Level15PC_v3_lm200,Level15PC_v4_clean,Level15PC_v4_control_clean,Level15PC_v4_control_lm200,Level15PC_v4_lm200}
runs/{Level15_DoG_clean,Level15_DoG_fixed_clean,Level15_ExtraHead_multisize,Level15_Hopfield_NoMainAP_multisize}
runs/{Level15_Hopfield_multisize,Level15_L2_clean,Level15_L2_lm200,Level15_L4_clean,Level15_L4_lm200}
runs/{Level15_PerScaleOmega_multisize,Level15_SR_lm200,Level15_clean,Level15_dagger,Level15_dk6_bc,Level15_dk6_dagger}
runs/{Level15_dkbc,Level15_empty_bc,Level15_empty_dagger,Level15_goal_lm200,Level15_goal_multistop,Level15_goal_noisy}
runs/{Level15_goal_switching,Level15_lm10,Level15_lm200,Level15_lm50,Level15_multiclass,Level15_multienv}
runs/{Level15_multienv_clean,Level15_multisize,Level15_noise,Level15_numberline,Level15_size16,Level15_size32}
runs/{Level15_topology,Level15_vocab256,Level15_vocab4096,Level1_clean,Level1_lm200,Level1_noise,MambaLike_clean}
runs/{MambaLike_lm200,MambaLike_noise,PC_clean,PC_lm200,PC_noise,RoPE_clean,RoPE_lm10,RoPE_lm200,RoPE_lm50,RoPE_multiclass}
runs/{RoPE_multienv,RoPE_multienv_clean,RoPE_multisize,RoPE_noise,RoPE_topology,TEMFaithful_FFN_clean,TEMFaithful_FFN_lm200}
runs/{TEMFaithful_clean,TEMFaithful_dg128_lm200,TEMFaithful_dg256_lm200,TEMFaithful_dg32_lm200,TEMFaithful_lm10}
runs/{TEMFaithful_lm200,TEMFaithful_lm50,TEMFaithful_multienv,TEMFaithful_multienv_clean,TEMFaithful_multisize}
runs/{TEMFaithful_noise,TEMFaithful_topology,TEM_T_clean,TEM_T_lm200,TEM_T_noise,TEM_clean,TEM_lm200,TEM_noise}
runs/{VanillaEM_clean,VanillaEM_lm200,VanillaEM_noise,VanillaEM_vocab256,VanillaEM_vocab4096,VanillaNoDrop_multienv}
runs/{VanillaNoDrop_multienv_clean,Vanilla_ExtraHead_clean,Vanilla_ExtraHead_lm200,Vanilla_ExtraHead_noise}
runs/{Vanilla_ExtraHead_numberline,Vanilla_L2_clean,Vanilla_L2_lm200,Vanilla_L4_clean,Vanilla_L4_lm200,Vanilla_clean}
runs/{Vanilla_dagger,Vanilla_dk6_bc,Vanilla_dk6_dagger,Vanilla_dkbc,Vanilla_empty_bc,Vanilla_empty_dagger}
runs/{Vanilla_goal_lm200,Vanilla_goal_multistop,Vanilla_goal_noisy,Vanilla_goal_switching,Vanilla_lm10,Vanilla_lm200}
runs/{Vanilla_lm50,Vanilla_multiclass,Vanilla_multienv,Vanilla_multienv_clean,Vanilla_multisize,Vanilla_noise}
runs/{Vanilla_numberline,Vanilla_topology,Vanilla_vocab256,Vanilla_vocab4096,aggtest,boundedtest,bump,cnav,comp_plain,fresh}
runs/{fresh_lm200,hier_multienv,hiertest,minigrid_dk16_clean_cached,minigrid_dk16_noise_cached,minigrid_doorkey_clean}
runs/{minigrid_doorkey_clean_cached,minigrid_doorkey_noise,minigrid_doorkey_noise_cached,minigrid_memory_cached,recursive}
runs/{repro,rgoal,rmaze,roomstest,route,torus_transnoise,vmaze}
```

**1c. Redundant checkpoint variants with no reader: 2.56 GB.** The full-val rescore (`CODE_FULLVAL_RESULTS.md`)
and the ABLATE A1-A3 rescore read `.final.pt`, and `best_val_bpc` is stored in each run JSON. No script reads
these `.best.pt` files:
```
runs/code_decay/*.best.pt                         12 files  1.28 GB
runs/code_ablate/*.best.pt                        11 files  1.17 GB
runs/code_ablate/PoPE-NoSigma_s2.best.pt.stale     1 file   0.11 GB  (the quarantined contaminated checkpoint;
                                                    its iteration 15000 / val_bpc 0.9962 are recorded in
                                                    ABLATE_RESULTS.md and ckpt_guard.py)
```
**Do NOT apply 1c to `runs/code2048/*.best.pt` or `runs/tinystories/p0/*.best.pt`.** Draft 2 App. J3-J4 reads
them (`docs/paper/expansion/scripts/language/usage_tick.py`, `tw_alpha_drift.py`), and so do `alpha_language.py`
and `code_step_table.py`.

**1d. Outside the repo.** `/home/prashr/mm_data/{pilots,pilots3,smoke_a2}` (0.08 GB; the pilot tables are
committed). `/tmp/claude-1002/-home-prashr-mapformer/*` session scratchpads (3.2 GB), but only when no session
is running (another agent's files were being written there during this inventory). `data/enwik8.zip` (35 MB;
the unzipped `data/enwik8` is what the code reads; check before deleting).

### Tier 2: evals stored, named re-analyses lost (**15.81 GB**)

Prerequisites, all of them, before any tier-2 deletion:
- **P1, snapshot.** Write a tarball of every non-checkpoint file under `runs/` (JSON, `.log`, `.txt`, `.md`,
  `code_md5.txt`, `.done`; 37 MB raw) outside the repo. Commit an md5 manifest of every checkpoint about to
  be deleted.
- **P2, extract what lives only in the checkpoint.** For the nav-trainer dirs, dump every non-weight key of each
  `.pt` (`results`, `losses`, `variant`, sizes, `allocentric` / `oracle` flags) to a sidecar `<name>.meta.json`.
  This is CPU work, a few minutes. It is required for the 15 dirs that have **no JSON and nothing tracked**:
  `vocab_em, minigrid_em, minigrid_em_fix, minigrid_2x2x2_allo, l15_loop_2x2, l15_ablation, noise_refine,
  loop_sampled, recipe_power, forget, forget_ctrl, mappope_r4, em_fig4, paper_task_rerun, lamtrace`. It is also
  required for the dirs whose aggregator reads losses from the `.pt`: `alias_sweep, alias_follow, rope_converge,
  g8_converge, visits_test, miniworld*, mw_*, flipflop`. Then point `agg_alias.py` / `agg_visits.py` /
  `agg_miniworld.py` at the sidecars, or accept that they will not re-run.

**2a. Not cited (reported / robustness-only / live negatives): 8.47 GB.**
| dirs | GB | re-analysis lost |
|---|---|---|
| `indirect_t3` | 1.76 | none documented |
| `jsb_len512` | 0.71 | `analyze_t1.py`, `probe_theory_numbers.py` (THEORY_MAPPOPE numbers; that account is withdrawn) |
| `jsb_t3`, `jsb_forced`, `jsb_centered`, `jsb_cross`, `jsb_decay`, `jsb_base{512,8192,32768}` | 1.49 | `analyze_t1.py` (jsb_centered), `probe_theory_numbers.py` (t3 / forced) |
| `mw_grid_sweep`, `mw_grid_sweep_hier`, `mw_hier_ablation`, `miniworld_fixed`, `miniworld_fresh`, `miniworld_oracle`, `miniworld` | 1.94 | `eval_ablation_miniworld.py` (context destruction, fresh maps); loss curves (P2); `experiment_audit.py`'s documented example dir (`mw_grid_sweep`) |
| `flipflop` | 0.57 | loss curves (P2) |
| `frontier_alg`, `algorithmic`, `hier_parity`, `loop_hier`, `rope_canon`, `selective` | 0.75 | `probe_rank_learned.py` / `probe_rank_truncate.py` / `probe_gate.py` / `probe_conv_kernel.py` (selective) |
| `sameblock` | 0.19 | `analyze_sameblock.py` (its registered analysis loads checkpoints) |
| `vocab_em` (+ `vocab_em_link`), `minigrid_em`, `minigrid_em_fix`, `minigrid_2x2x2_allo`, `n5_phase`, `counter_batch`, `recency_budget`, `dyck_t3`, `cmq_curric`, `cmq_stab`, `cmq_sweep`, `recipe_power`, `paper_task_rerun`, `em_fig4` | 0.68 | `analyze_papertask.py` / `probe_collisions.py`, `gain_grain_power.py` (em_fig4); results only via P2 for the no-JSON dirs |
| `mq_noise`, `mq_noise_c2`, `l15_loop_2x2`, `l15_ablation`, `noise_refine`, `loop_sampled`, `forget`, `forget_ctrl`, `lamtrace`, `popewrap`, `mappope_r4`, `mappope_recency` | 0.38 | `ablate_match_noise.py`, `eval_noise_refine.py` (Level 1.5 negatives; "do not re-run") |

**2b. Cited, evals in tracked JSON: 7.34 GB.**
| path | GB | re-analysis lost |
|---|---|---|
| `runs/indirect/**/*.pt` | 3.02 | `docs/audits/2026-10-05/pope_task_step_table.py` (step table of the 100k models). The cited "1/8 at 100k" is in the tracked JSONs |
| `runs/indirect_200k/**/*.pt` | 1.51 | `eval_indirect_ood.py` (INDIRECT_OOD, eval-only) and the step table. The cited 7/8 / 8/8 is in the JSONs |
| `runs/code/*.best.pt` | 1.28 | `probe_code_depth0.py` / `probe_code_accum.py` (CODE_RESULTS, withdrawn OOD) |
| `runs/jsb/**/*.pt` | 0.36 | `probe_theory_numbers.py`, step table. JSB_RESULTS -0.032 is in the JSONs |
| `runs/{alias_sweep,alias_follow,rope_converge,g8_converge,visits_test}/**/*.pt` | 0.80 | the map-extent boundary row: accuracies in JSON; **losses only via P2**; re-evaluation at new lengths |
| `runs/jsb_aug/**/*.pt` | 0.18 | none documented (AUG_RESULTS numbers in JSON) |
| `runs/tinystories/p0/*.final.pt` | 0.20 | none documented (keep the `.best.pt`) |

### Tier 3: anything else (not recommended without a second copy)

| path | GB | what it costs |
|---|---|---|
| `runs/{code,code_decay,code_ablate}/*.final.pt` | 3.84 | the cited CODE_FULLVAL C1/C2 and ABLATE A1-A3 rescore; re-deriving means retraining 36 code runs |
| the 101 KEEP-CHECKPOINTS dirs | 6.93 | registered analyses that re-score from weights (strata, censuses, what-where, dropout rescore, rank probes, `test_guards`). **Keep** |
| `/home/prashr/mm_data/zips/15x15/*.zip` | 12.55 | re-download (~13 GB, public Drive, `hyak/mm_download.sh`) to rebuild arrays or re-run the gate. Record md5s first |
| `data/tinystories/TinyStoriesV2-GPT4-{train,valid}.txt` | 2.10 | re-download to rebuild `train.bin` / `val.bin` (kept). `tinystories_data.py` needs the raw text only to rebuild |
| `/home/prashr/mm_data/arrays/` | 2.15 | **keep**: the frozen encoder and embeddings the registered batch used |

**Totals.** Tier 1: 3.83 GB in the repo (+0.08 GB mm pilots, +3.2 GB scratch). Tier 2: 15.81 GB (8.47 not
cited + 7.34 cited). Tier 3: 3.84 GB (code `.final.pt`) + 14.65 GB re-downloadable data. KEEP-CHECKPOINTS
stays at 6.93 GB. Tiers 1 + 2 take `runs/` from 30.7 GB to about 11.1 GB.

---

## 4. Git hygiene proposals

1. **Commit the small files of the newest registered batches. They are untracked today.** `runs/{exptorus,
   image_torus,signsep,ssm_sign,torus_lrsweep,factorial_n8,rnn_torus}` and their pilots hold about 390 files
   (3 MB): per-run `eval.json`, `code_md5.txt`, smoke JSON, and `runs/srope_torus_pilot/PILOT_SRoPE*.md`. Older
   batches commit exactly these files. `git status` lists them as `??`.
2. **Bring the Memory Maze per-run results into the repo.** `/home/prashr/mm_data/batch_runs/*/results.json`
   (64 files, 17.6 MB, plus `DONE` and 2.2 MB of `log.txt`) is the only local copy behind
   `MEMORY_MAZE_RESULTS.md`, and the cluster copy sits on purgeable scratch. Commit it as `runs/memory_maze/`,
   or as one compressed archive if 17.6 MB of JSON is too noisy.
3. **Logs.** Keep `*.log` ignored (6,770 per-run logs, 21 MB, would be noise in git). Keep them on disk and in
   the P1 tarball: for many runs they are the only loss curve outside the checkpoint.
4. **`.gitignore` tidy-ups.** `*.log` appears twice. The 34 explicit `.<name>_done` lines are redundant with
   `.*_done`. `runs/lap_probe/` names a dir that no longer exists. `.done` is ignored, yet 34 `runs/**/.done` are
   tracked (added before the rule): decide whether to keep or `git rm --cached` them. Add `*.lock.stale*` (an
   untracked `.run_signsep.lock.stale_20261024` sits at the repo root) and `papers/new_*/**/*.pdf` (28 untracked
   PDFs; only `papers/pdf/` is ignored).
5. **Tracked absolute symlinks.** 28 symlinks (`rank_mi/p0/*` x10, `rank_nowrap_pilot/amend2/**` x16,
   `{dof,n5_phase}/torus_eval_stage/p0`) store `/home/prashr/mapformer/...` targets, so they resolve only in a
   clone at that path. Rewrite them as relative links.
6. **Repository size is fine** (pack 74 MB). `git gc` would pack 106 MB of loose objects. History holds repeated
   rebuilds of `docs/paper/pdf/*.pdf` (2.9-4.2 MB each, 7+ versions) and four 8.4 MB `paper_figures/cnav_*.npz`.
   No rewrite is needed.
7. **Repo-root items to confirm.** A personal PDF (`Prashant_Rangarajan_Generals_Document.pdf`, 2.7 MB) and
   `MapFormer.pdf` are tracked at the root; confirm they belong in the repo. `_INIT_GRAD.json` (4.9 MB, tracked)
   is the largest non-PDF blob.

---

## 5. Risks and how to run the cleanup safely

1. **Rule 26: verify before and after.** *Before:* record `du -sk` per dir, write the exact file list with md5
   (the manifest from P1), and confirm `git ls-files -d runs` is empty. *After:* confirm `git ls-files -d runs`
   is still empty (no tracked file removed) and `find runs -xtype l` prints nothing (no dangling symlink). Run
   `cd /home/prashr && python3 -m mapformer.test_guards`, which needs `.pt` under `runs/{dof,warm,unfreeze,
   recency_em,magonly_repro}` (all KEEP). Re-run one registered analysis that reads a pilot repro checkpoint, e.g.
   `analyze_score_rank.py` (reads `runs/score_rank_pilot/repro/...`) or `analyze_gain_grain.py`.
2. **`safe_clear.sh` is the wrong tool for this cleanup, and its refusals must not be bypassed.** It removes whole
   dirs only and fails closed: it refuses any dir with a marker inside, a repo-level `.<name>_done`, a marker
   naming a prefix, or a live `python3` naming the dir. 187 of 436 dirs are refused, including 36 of the tier-1
   pilots. Checkpoint-only deletion goes around it by design, so the manifest and the after-checks above replace
   it. Use it as intended only for the legacy whole-dir removals.
3. **Drivers re-train on a missing checkpoint.** `lib_driver.sh` and most drivers skip a run iff its `.pt` exists
   (`[ -f "$OUT/$V.pt" ] && continue`; 46 drivers test `.pt` directly). After tier 2, re-launching an old driver
   silently re-trains the deleted runs and can overwrite their JSON and logs; `drv_require <...>.pt` checks fail.
   Leave a `CHECKPOINTS_DELETED` note in each touched dir and never re-launch those drivers.
4. **Do not delete a symlink target.** `rank_mi` (cited) reads `rank_matched_e900/p0` and
   `rank_perhead_pilot/p0` through symlinks; both are KEEP. `vocab_em_link` depends on `vocab_em`. Several
   registered analyses read pilot repro checkpoints (`score_rank_pilot`, `gain_grain_pilot`, `rnn_torus_pilot`,
   `exptorus_pilot`, `wrap_existence_pilot`, `*_repro`). These are classed KEEP even though their names say pilot.
5. **Orphaned processes.** 40 `python3 -c ... multiprocessing.spawn` workers have run since 2026-10-26 01:39
   (cwd `/home/prashr`, no live parent), probably orphaned data workers. `safe_clear.sh`'s live check matches
   command lines naming the dir, so it would not see them. Before deleting, confirm with `ls -l /proc/<pid>/fd`
   that none holds a file under `runs/`. They were not touched.
6. **Another session may be active.** The repo root's mtime changed during this inventory, and a parallel agent
   was writing to the shared scratchpad. Check `ps` and `git status` immediately before deleting anything.
7. **Checkpoints that are not on kalman.** SIGNROUTE, RANKCLOCK, RANKWD and RANKRECIPE ran on the koopman clone,
   so their run dirs are not here. Memory Maze runs are on cluster scratch plus `mm_data/batch_runs`. The
   Match-Query run dirs behind a cited row exist nowhere (`docs/paper/review_draft5/REPRO_AUDIT_nonsign.md`: "could
   not run"). That is the precedent for not dropping cited checkpoints without a second copy. Also,
   `docs/RUNBOOK_NEXT_EXPERIMENTS.md` says `runs/` is 45 GB "here"; on kalman it is 30.7 GB today.
8. **Duplicates are evidence** (end of section 2). Hard-link them if anything; do not delete them.
9. **Classification limits.** Readers were found by name: a script that builds the dir name at runtime from a
   variable (rather than a literal `runs/<dir>`) can be missed. Every large dir (>100 MB) was checked by hand.
   The small-dir rows rely on the scan. Of 436 dirs, 266 are DROP-CANDIDATE (1.27 GB in all), so a misclassified
   small dir costs little and is recoverable from P1 + retraining only if its code still runs.
