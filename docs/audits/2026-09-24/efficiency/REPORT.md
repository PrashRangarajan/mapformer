I couldn't write `REPORT.md`: the harness refuses report files from subagents, so the full report is below instead. The patches, proposed files and gate scripts are all in `/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency/`. Nothing in the repo was changed and the GPUs were not touched.

# Code-efficiency audit (2026-09-24)

**VERIFIED** means I measured it. **JUDGED** means I read it from the code or logs. CPU timings were single-threaded on a machine at load 26–34, so compare ratios, not absolute numbers.

## Where the time goes at the rank config (B=16, n_steps=1024, d=128, 1 layer, 2 heads)

The GPU is the bottleneck. The CPU is not, and running 5 jobs per GPU does not help.

- **The attention tensor is huge.** The explicit attention path builds a [16, 2, 2047, 2047] fp32 score tensor: 0.54 GB, read or written about 22 times per step, roughly 12 GB of memory traffic.
- **What ps and nvidia-smi show (VERIFIED):**
  - Both GPUs are at 100%.
  - The data workers are idle about 72% of the time, waiting on a pipe.
  - Each trainer's main thread runs at about 91% CPU. I judge this to be spin-waiting on CUDA syncs, not useful work.
- **What the logs show (VERIFIED), in GPU time per epoch:**
  - a job running alone: 2.4 s
  - 5 jobs per GPU: 12.6–13.3 s per epoch per job, so 2.52–2.66 s each
  - the whole e900 batch: 2.62 s per epoch per GPU
- **Baseline for a 16-run, 900-epoch batch:** about 5.2 h of training, then about 2.5 min of evaluation and analysis.

## Findings, ranked by wall time saved per batch

| # | Change | Saves per batch | Same results? | Status |
|---|---|---|---|---|
| 1 | SDPA attention: add `--fast-attn` to `train_variant.py`, **without TF32** | ~3–4 h (2.5–4x) | No — new batches only | JUDGED |
| 2 | Vectorised random walk in `environment.py` | ~0 on its own, but #1 needs it; also likely speeds jobs that run alone | Byte-identical, RNG state included | VERIFIED, 23x faster |
| 3 | Fold 1/sqrt(d_head) into Q, and make `masked_fill_` in-place | ~0.5 h (~10%) | Bitwise on CPU; GPU check pending | Mixed |
| 4 | Run 2 jobs per GPU instead of 5 (`MAXPG` 5 → 2) | ~0.25–0.5 h (5–10%) | Not applicable | VERIFIED from logs |
| 5 | Remove per-step device syncs from the training loop | 0–5% now, 10–20% after #1 | Bitwise on CPU; GPU check pending | Mixed |
| 6 | Evaluators: #2 plus vectorised strata scoring | ~2 min | Byte-identical JSON | VERIFIED |
| 7 | Vectorised exact permutation test | 9 s (idle) or 51 s (loaded) → 0.1 s | Identical p-values and CI | VERIFIED |
| 8 | Shared driver library | ~0; mainly fixes correctness | Not applicable | JUDGED |

### 1. SDPA attention (`train_variant_fastattn.patch`)

- **Where:**
  - The explicit attention path is at `model.py:243-251`.
  - `train_variant.py` has no `--fast-attn` flag, so every navigation run takes that path.
  - The flag exists in `train_match_query.py:105-129` and similar trainers, but there it also turns on TF32. TF32, not SDPA, is what broke equivalence on the compositional trainer (logit difference 6e-01).
- **Why it pays:**
  - About 13–14 ms of the ~25 ms step is spent moving the score tensor through GPU memory. The memory-efficient SDPA kernel never materialises it.
  - The repo already measured 2.56x at B=24, T=1024 (comment at `model.py:233-236`).
  - This config has about 2.7x more of that quadratic work per step, so I estimate 2.5–4x, taking the batch from ~5.2 h to ~1.3–2.1 h.
  - GPU memory per job drops from ~3.4 GB to under 1 GB.
- **The patch adds:**
  - `--fast-attn`, with TF32 explicitly off
  - `--deterministic`, which calls `torch.use_deterministic_algorithms(True, warn_only=True)`
  - both settings saved in the checkpoint config
- **Reproducibility risk: high.**
  - SDPA drops attention weights with a different random-number stream and sums in a different order. SDPA runs can never be pooled with, or continue, the e900, e900c or proj batches.
  - I found this warning string in the installed `libtorch_cuda.so` (VERIFIED): "Memory Efficient attention defaults to a non-deterministic algorithm". Without `--deterministic`, same-seed reruns are probably not bitwise identical, which rule 27 depends on.
  - `verify_gpu_bitexact.py` part 2 measures the speed-up and checks run-to-run reproducibility with and without `--deterministic`.
- **Depends on #2.** At 3x GPU speed, the old walk generator would need about 23 CPU cores for 10 jobs, and training would become CPU-bound.

### 2. Vectorised random walk (`environment_fastwalk.patch`)

- **Where the time goes now:**
  - `environment.py:294-377` loops once per step.
  - Inside it, the `obs_map[...].item()` call at line 359 costs 2.1 µs.
  - A torch bool setitem per revisit at lines 388-390 costs 2.3 µs.
  - The comment at `data_parallel.py:10-12` says no micro-optimisation is worth having; this finding refutes that.
- **How it stays identical:**
  - `np.random.randint` works by masked rejection on raw MT19937 words, and NumPy freezes that legacy stream (NEP 19). So the default walk is a deterministic parse of the raw word stream.
  - The patch parses one loop iteration per segment (about n/5.5) instead of per step. It then rewinds the RNG and advances it by exactly the words consumed. Everything else is vectorised numpy.
  - It only runs where it is proven identical: translate actions, allocentric view, torus, commanded actions, no transition noise, and no subclass override of `generate_trajectory` (`environment_topology.py` has one).
  - Setting `FAST_WALK = False` restores the old loop.
- **Evidence (VERIFIED):**
  - `verify_fastgen.out`: 75/75 byte-exact checks, covering tokens, masks, locations, `last_x`/`last_y` and the global numpy RNG state after the call. Configs include grids of 64, 48, 32 and 5, 200 landmarks, n_steps from 1 to 2048, sequences of calls, and the `data_parallel` worker protocol.
  - `verify_env_patch.out`: 17/17 checks of the patched `environment.py` against the original, including fallback for non-default configs, the subclass override, and the `FAST_WALK` off switch.
  - Speed: one B=16 × 1024 batch goes from 82.7 ms to 3.6 ms. Per trajectory it is 8x, 13x and 16x faster at T = 512, 1024 and 2048. Per job-epoch that is ~8.1 CPU-seconds → 0.36.
- **Knock-on gains:**
  - Frees about 8.5 CPU cores across 10 jobs.
  - `--data-workers 1` can replace `3` with a byte-identical data stream, because the stream does not depend on worker count. That removes 20 spawned processes of ~550 MB each.
  - With 3 workers the old generator could only supply about 2.7 s of batches per epoch, so jobs running alone (such as the ~45-minute s7 tail of e900) were probably data-bound (JUDGED).
- **Operational note:** `run_rank_matched.sh:24-28` checks the md5 of `environment.py` and will abort if it changes. Apply the patch between series and regenerate `code_md5.txt` on purpose.

### 3. Scale fold (`model_scalefold.patch`)

- **What changes:** `model.py:245-246` becomes `(Q/8) @ K^T` plus an in-place `masked_fill_`.
- **Why it is exact:**
  - With d_head = 64, the scale is 8 = 2³, a power of two.
  - Multiplying by a power of two commutes exactly with rounding in the forward matmul and both backward matmuls.
  - The in-place mask is safe because the matmul's backward never reads its output.
- **Saving:** it removes one full pass over the score tensor in the forward and one in the backward, about 2.1 of ~12 GB. I estimate ~10% of step time.
- **Evidence:**
  - `verify_cpu_bitexact.out`: logits and all gradients are bitwise equal with dropout on, at d_head 64 and at the non-power-of-two fallback (48).
  - It still needs the GPU check, `verify_gpu_bitexact.py` part 1. Setting `_POW2_SCALE_FOLD = False` reverts it.
- **Why it matters:** it is the only GPU saving that can be used inside an existing explicit-path series, once that check passes.

### 4. Jobs per GPU

- **Where:** `run_rank_matched.sh:32` (`MAXPG=5`) and `run_rank_proj.sh:17`.
- **Evidence:** running 5-way costs 5–10% in context switching and ~3.4 GB per job, compared with one job alone.
- **Recommendation:** use 2 per GPU, then re-measure 1, 2 and 3 after #1 and #2.
- **Optional:** the spinning main threads burn about 9 cores in total. CUDA blocking-sync mode (set through ctypes before CUDA starts) would stop that. This is low priority.

### 5. Removing per-step syncs (`train_syncfree.patch`)

- **Where the syncs are:**
  - `train.py:118-119`, the plain host-to-device copies
  - line 154, the `target_mask.sum()` check
  - line 157, boolean indexing, which calls `nonzero`
  - line 173, `loss.item()`
- **What the patch does:**
  - makes the skip decision on the CPU copy of the mask
  - gathers the scored rows with CPU-computed flat indices, which pick the same rows in the same order
  - uses pinned, non-blocking copies
  - accumulates the epoch loss on the GPU in float64, in the same order as before, and reads it once per epoch
- **Evidence:** `verify_cpu_bitexact.out` shows losses and all parameters bitwise equal after 2 epochs, under both schedules and for a config with no revisits.

### 6. Evaluators

- **`eval_noise_refine.py`:** almost all of its time is the per-step walk, so #2 speeds it up automatically with identical trajectories. No code change is needed.
- **`eval_rank_strata.py` (`eval_rank_strata_vec.patch`):**
  - Lines 34-43 and 58-64 do Python work per scored target.
  - The patch vectorises that. It sums the NLL with `np.cumsum` behind a leading 0.0, which reproduces the old left-to-right `+=` exactly, signed zeros included.
  - `verify_strata.out` shows identical dicts and JSON on trained e900 checkpoints.
  - Per trial at T=2048, time drops from about 233 ms to 19 ms when combined with #2 (noisy measurement).
- **Optional merge:** `eval_noise_refine` and `eval_rank_strata` run the same forward on the same trajectories. Merging them would halve eval GPU time, at the cost of the independent `--check-json` cross-check.

### 7. Permutation test (`analyze_perm_vec.patch`)

- **Where:** `analyze_rank_matched.py:35-48`. `perm_ci` calls `perm` 241 times, each a Python loop over 12,870 combinations.
- **Patch:** builds the combination matrix once and computes each `perm` as one gather plus row sums.
- **Evidence:** `verify_perm.out` shows an identical CI of (0.050, 0.155) in 0.12 s instead of 50.7 s under load. p-values are identical on every committed contrast and on 90 random cases with heavy ties.

### 8. Driver library (`lib_driver.sh`)

- **The duplication across the 273 `run_*.sh`:**
  - 81 define `on_gpu()`, 65 `busy()`, 51 `pick()` and 37 `is_gpu_free()`.
  - 34 still use `pgrep`, the self-matching trap CLAUDE.md warns about.
  - `MAXPG` values of 2 through 6 all appear.
- **What the library provides:**
  - a trainer count that matches on `comm`, so shells never match
  - least-loaded GPU picking
  - a lock
  - the md5 guard
  - `setsid` launch
  - wait-for-batch
  - a done marker set only after every artifact exists
- **Wall time:** the 45 s launch spacing and the polling loops cost nothing while jobs are GPU-bound.

### Considered and not recommended

- **torch.compile:** it changes the dropout random stream, so it has the same reproducibility cost as SDPA but is more complex.
- **CUDA graphs:** the variable-length revisit gather prevents static shapes.
- **bf16 with flash attention:** a precision change that would need its own equivalence study.
- **Larger batches or cached trajectories:** these change the recipe or data distribution.
- **Micro-tweaks** (caching the causal mask, int16 tokens, dropping `obs_mask` from the worker queue): each saves under 1%.

## Summary

1. **SDPA without TF32 is the only big win.** Training is GPU-bound at 2.4–2.6 s of GPU time per epoch, most of it spent moving the 0.54 GB score tensor. I expect 2.5–4x, taking a 16-run × 900-epoch batch from ~5.2 h to ~1.3–2.1 h (JUDGED). It is not bitwise-reproducible, and its backward pass is non-deterministic unless `--deterministic` is set, so use it only for new series.
2. **The vectorised walk is byte-identical, RNG state included, and 23x faster** (82.7 → 3.6 ms per batch, VERIFIED across 92 checks). It frees about 8.5 cores and lets `--data-workers 1` replace 3 with the same data stream. #1 depends on it.
3. **The scale fold is bitwise on CPU and saves an estimated ~10% of GPU time.** It is the only GPU saving usable inside an existing explicit-path series, once the GPU check passes.
4. **Five jobs per GPU is past the optimum.** It costs 5–10% against one job alone (VERIFIED from logs); use 2.
5. **The sync-free loop and the vectorised evaluators and statistics are bitwise or byte-identical** (VERIFIED on CPU). The permutation CI goes from 50.7 s to 0.12 s and eval from ~2.5 min to well under a minute, which is small against a 5 h batch.

**Before applying anything:**
- **GPU check:** `verify_gpu_bitexact.py` has not been run. It needs an idle GPU, and it is the check #3 and #5 require.
- **md5 guard:** `run_rank_matched.sh` checks the md5 of `environment.py`, `model.py` and `train.py`. Apply these changes only between series and regenerate `code_md5.txt` on purpose.
- **CPU budget:** I went slightly over the ~1 core-minute limit. The original `perm_ci` alone took 51 s under load.

**Suggested order:**
1. #2 and #7, which carry no risk.
2. #6.
3. #5 and #3, after the GPU check.
4. #1 with `--deterministic`, for the next new series, then re-measure jobs per GPU.

Everything is in `/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency/`:
- **Patches:** `environment_fastwalk.patch`, `model_scalefold.patch`, `train_syncfree.patch`, `train_variant_fastattn.patch`, `eval_rank_strata_vec.patch`, `analyze_perm_vec.patch`
- **Full replacement files:** `proposed/*.py`
- **Checks already run, with their `.out` files:** `verify_fastgen.py`, `verify_env_patch.py`, `verify_cpu_bitexact.py`, `verify_strata.py`, `verify_perm.py`, `time_strata.py`
- **Not yet run:** `verify_gpu_bitexact.py`
- **Other:** `fastgen.py` (standalone fast walk), `lib_driver.sh`