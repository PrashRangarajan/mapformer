---
name: feedback-scheduler-and-measurement-traps
description: Operational traps behind CLAUDE.md rules 20-27 -- GPU pickers, orphans and duplicate drivers, running-script edits, pkill -f, destructive commands, relative paths -- plus two measurement traps.
metadata:
  type: feedback
---

Each cost real time and was caught only by looking at something other than the log.

## Scheduling (rules 21, 24)

- **A fill-first GPU picker is a single-GPU scheduler when jobs <= MAXPG.** After `--fast-attn` cut
  memory I raised MAXPG 3 -> 6; with exactly 6 jobs GPU 0 never filled and **a 4090 sat at 0% for
  three hours** while the log reported six launches. Pick the LESS LOADED device, and do not then
  interleave job types: alternating types against an alternating picker phase-locks every long job
  onto one card (I reproduced the bug I was fixing). Check `nvidia-smi`, never the launch log.
- **`cuda:$((i%2))` is not a scheduler.** Runs finish out of order; two 13.5-14.7 GiB runs landed on
  one 24 GiB card and OOM-died. Pick by actual per-card occupancy (`gpu_busy` in
  `run_code2048_fill.sh`).
- **CPU threads.** Ten GPU jobs with uncapped threads: load 52 on 32 cores, 3x slow.
  `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` in every launcher: 24 -> 6 s/epoch.

## Drivers and processes (rules 20-23)

- **Killing a supervisor does not kill its children.** The launcher survived as an orphan and the next
  supervisor started a second copy: 16 launches for 12 runs, ~10 GPU-hours. Every driver takes
  `exec 9>/tmp/.mapformer_<name>.lock; flock -n 9 || exit 0`.
- **Killing a trainer orphans its `--data-workers` (2026-10-03).** 76 orphaned multiprocessing processes
  (tracker + 3 workers per killed run, ~30 GB RSS) had piled up. After stopping runs, check `ps` for
  python3 processes with parent 1 and kill them by PID (never `pkill -f`).
- **Duplicate drivers (2026-09-20).** I started one batch three times; three drivers raced on the same
  seeds. Check `ps` for the driver by name before relaunching; kill extras by PID.
- **A checkpoint can be stale and load cleanly.** A duplicate launch overwrote a finished best.pt with
  an iter-15000 one, read as "an anomalous unstable seed". Evaluators must cross-check the stored val
  metric against the run JSON and refuse on mismatch (`eval_code_long.py` does).
- **Never edit a running bash script** (bash reads by byte offset; an insert makes it resume
  mid-token). Write a new file and `mv` it over; the running copy keeps its inode.
- **Never `pkill -f` / `pgrep -f`.** They match your own shell (2026-09-06 it killed my command
  mid-task) and the author's shells (a waiter sat 2 h on zero jobs). Use
  `ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/'`.
- **Check the `.done` marker before calling a batch unfinished.** C1 was reported unfinished when done;
  C2 sat complete and unread for a day.

## Destructive commands (rule 26)

Twice I acted on an assumption the terminal had already contradicted.
- **A failed `git add` stages NOTHING.** `git add -A a b c 2>/dev/null` with one bad pathspec fails
  the whole call silently; two commits shipped a 40-paper corpus without its document. ` M` with a
  leading space in `git status --short` is UNSTAGED. Never `2>/dev/null` a `git add`; verify with
  `git show --stat HEAD` or `git show HEAD:<file>`.
- **`rm -rf` on a batch that had finished 40 min earlier** (forget-clock, 48/48, ~3.3 GPU-h). The
  `.forget_clock_done` marker existed, the log said `finished`, and the previous command printed
  "48 checkpoints"; I read "no workers" as early, then invented "nothing lost". When output
  contradicts the plan, the output wins. `safe_clear.sh` is meant to refuse a dir whose marker exists
  but FAILS OPEN from /home/prashr (relative marker path) until patched -- do not rely on it.

## Paths (rule 25)

`python3 -m mapformer.X` runs from `/home/prashr`, so every relative path inside the module resolves
there and silently matches nothing (globs empty, aggregators print `--`, no error). Four debugging
rounds. Use `REPO = "/home/prashr/mapformer"` in modules; `cd "$REPO"` in heredocs. Tell: an
aggregator that runs cleanly with zero rows or n=0 everywhere; `ls` one path before believing it.

## Measurement traps

- **Do not infer held-out accuracy from training loss.** An index arm at 0.03 loss scored 0.674
  held-out against 0.949; the loss-accuracy relation holds in some regimes only. Wait for the eval.
- **A wide pre-registered band is not a pre-registration.** "Between -0.010 and +0.374" fired on
  +0.015, 0.025 from the null against a 0.150 floor. Set disjoint branches against the measured floor.
- **LaTeX**: a scripted mid-line `% src:` silently deleted two sentences from report.pdf. Comments on
  their own lines; grep for text after `%`; pdftotext the result.
- **Fast paths**: a vectorised generator was wrong on 92/3200 rows (carry overwrote the top digit).
  Verify row-exact; for torch.compile compare eager vs compiled losses per arm
  (`verify_addition_compile.py`).

Related: [[feedback-convergence-first]], [[feedback-probe-verification]].
