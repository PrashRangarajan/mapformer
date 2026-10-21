# Runbook: running the next experiments on another machine (2026-10-20)

Start here if you are picking this project up on a different computer. Read `CLAUDE.md` (binding rules 1-29) and
`.claude-memory/project_state.md` first; this file says what is queued, what is ready to launch, and how to set a machine
up so the drivers run unchanged.

## 1. Machine setup

1. **Path.** Code and drivers assume the repository lives at **`/home/prashr/mapformer`** (337 files hard-code it, and
   every driver runs `python3 -m mapformer.<module>` from the parent directory, so the directory must be named
   `mapformer`). Either clone it to exactly that path (create the user / a bind mount / `sudo ln -s $HOME /home/prashr`),
   or on the new machine only run
   `grep -rl /home/prashr/mapformer --include=*.py --include=*.sh . | xargs sed -i "s#/home/prashr/mapformer#$PWD#g"`
   and **do not commit that change**.
2. **Python / torch.** The batches since September ran on torch **2.10.0+cu128** (user site). Install it with CUDA and
   check `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` -- pip can silently install a
   CPU wheel. Other packages: `requirements.txt` (and `HOURGLASS_README.md`). scipy, numpy, matplotlib, markdown and
   weasyprint are used by analyses and the PDF builder.
3. **Sanity.** `cd /home/prashr && python3 -m mapformer.test_guards` (13/13 should pass).
4. **Checkpoints are NOT in git** (`*.pt` is ignored; `runs/` is 45 GB here). Every queued GPU batch trains from scratch, so
   none needs them. The CPU re-analyses do: copy the specific run dirs they read (`rsync -a prashr@<this host>:/home/prashr/
   mapformer/runs/<dir> runs/`), e.g. `runs/sign_matched`, `runs/generality`, `runs/rank_*`, `runs/factorial*`.
5. **Shared GPUs.** Check `nvidia-smi` for other users before launching; do not launch beside another user's jobs without
   the user's approval. Drivers take `DRV_GPUS="0 1"` and a per-GPU cap (`DRV_MAXPG`, set in each driver).

## 2. Protocol for every batch (rules 7, 11, 12, 29 in CLAUDE.md)

Pre-registration file `<NAME>_PREREG.md` with registered decisions and predictions -> code + `analyze_<name>.py
--selftest` reaching every label -> CPU smoke of every arm -> pilot on seeds OUTSIDE the batch (never pooled) -> an
independent, results-blind code-check agent -> its findings appended as Amendment 1 -> commit (drivers refuse a dirty or
uncommitted tree) -> launch under `setsid nohup` -> results file `<NAME>_RESULTS.md` from the registered analysis only.
Templates to copy: `ROBUST_PREREG.md` / `run_robust.sh` / `analyze_robust.py`, `LRSWEEP_*`, `DELTA_FREE_*`,
`SROPE_TORUS_*`. Drivers source `lib_driver.sh` (flock lock, clean-tree check, md5 guard, slot picker, done marker).

## 3. State of the queue

| id | experiment | status | how to run |
|---|---|---|---|
| SROPE_TORUS | faithful Selective RoPE (authors' code reproduced) vs MapWM and RoPE on the T = 1024 torus, 6 arms x 8 seeds | **registered, running on the original machine** (started 2026-10-20 16:02, GPU 1, ~18/48 done at 22:44). Do not re-run elsewhere unless that run fails; if both run, never pool them | `cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_srope_torus.sh > /dev/null 2>&1 &` (~7.5 h on two free 4090s) |
| DELTA_FREE | PoPE peak-pinning test | **done** (`DELTA_FREE_RESULTS.md`) | -- |
| LRSWEEP, ROBUST, GAUGE, ... | sign line | **done** (results files; CLAUDE.md sign row) | -- |

### Not yet built (design in `docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md` Sec. 5.1; detail in the expansion reports)

Ranked by value; "decisive" = the theory and its rival predict opposite outcomes. The cheapest set that registers the
clock theory is G-2 (ring part) + G-1, about 19 h wall on two GPUs (G-6 is done).

| id | experiment | design sketch (full spec in the named report) | cost | decisive? |
|---|---|---|---|---|
| **G-1** | clock-route batch (rank) | torus T = 1024, 900 ep, n = 8 fresh seeds: shared r2; r2 + per-channel clock slot (identical at init, +64 params); r2 drift-forbidden (tie z_W = -z_E, z_S = -z_N); r2 with observation steps zeroed; r4; optional r4 with no weight decay; save snapshots every 50 epochs. `RANK_THEORY.md` G1/G1+, `SEARCH_THEORY.md` 6.1/6.3 | ~14 h | yes: clock slot >= 6/8 supports parking; <= 2/8 supports over-parameterisation |
| **G-2** | sign-route batch | ring, observation-step-free, n = 8: signed; abs; hybrid "non-negative readout, signed latent" (a = omega |W| z: collective route restored, predicted to solve); abs with tied latents (predicted to fail); exp (LRU-style) and softplus steps at their best rates. Torus part: frozen move latents, n = 8. `SEARCH_THEORY.md` 6.2/6.4 | ~5 h ring + ~4 h torus | yes (the cheapest decisive test) |
| G-3 | sign beyond attention | one-layer diagonal complex SSM h_t = exp(i omega Delta(x_t)) h_{t-1} + B x_t on Z_m (m 8, 32; T 128-512), moves +1 / -1 / stay; signed, abs, softplus, exp; n = 8; existence construction as Prop. 1. `PAPER_LEVEL.md` #1 | 5-8 GPU-h + 1-2 days code | generalises the claim beyond attention |
| G-4 | torus best-rate sign sweep | observation-step-free torus, signed vs abs at lr 1e-3 / 3e-3 / 1e-2, n = 8, selection-aware test copied from `analyze_lrsweep.py`; optional r2 vs r4 | ~12-15 h (sign only) | closes the ring-only learning-rate scope |
| G-5 | depth routes | (a) ring T = 128, index 2 layers with / without a BOS token + absolute position channel, n = 4-8 (~3 GPU-h); (b) torus walks with run length fixed at 2 / 5 / 10, index 4L and path 1L, T = 512 (~24 GPU-h). `DEPTH_THEORY.md` E1/E3 | ~27 GPU-h | local-rule vs global-count account |
| G-7..G-11 | text world at 4096 words; rank x sign x passes factorial; rank follow-ups; LSTM/GRU baseline; egocentric actions; scale | see the plan's table | 10-75 GPU-h each | breadth for a main-track venue |

## 4. CPU work that does not need a GPU (also from the plan, Sec. 5.2)

Retrace floor on the TW_NORMSTEP / TW_STATECHANGE batches (CPU-2); harmonise the clock metric between
`RANK_THEORY.md` and `SEARCH_THEORY.md` (CPU-3, CPU-4); exp-step existence construction (CPU-7, needed by G-2 and G-3);
an independent code audit of the ~20 analysis scripts under `docs/paper/expansion/scripts/` whose numbers enter the
paper (CPU-9); new figures (CPU-10); then writing draft 2 (`docs/paper/MAIN_DRAFT1.md` -> plan in
`SYNTHESIS_DRAFT2_PLAN.md`). These need the stored checkpoints (Sec. 1.4).

## 5. Where things are

- Paper: `docs/paper/MAIN_DRAFT1.md` (+ `MAIN_APPENDIX1.md`, `APPENDIX_theory_v3.md`), outline `OUTLINE_main_v2.md`,
  draft-2 plan `expansion/SYNTHESIS_DRAFT2_PLAN.md`, sign workshop paper `DRAFT6_workshop.md`, PDFs in `docs/paper/pdf/`
  (rebuild with `python3 docs/paper/tools/build_pdfs.py docs/paper/pdf`), summary figure `figs/fig_summary.png`.
- Results: `*_RESULTS.md` at the top level (a CORRECTED / AUDIT block at the top supersedes the body); catalogue
  `RESULTS_INDEX.md`; citable and withdrawn claims in `CLAUDE.md`.
- Reviews and audits: `docs/paper/review_draft5/`, `docs/audits/<date>/`.
