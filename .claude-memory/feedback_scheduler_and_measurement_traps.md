---
name: feedback-scheduler-and-measurement-traps
description: Four traps bought in one week — fill-first GPU pickers, inferring accuracy from loss, wide pre-registered bands, editing running scripts.
metadata:
  type: feedback
---

Four ways I wasted real time on the MapFormer runs, each caught only by looking at
something other than the log.

**1. A fill-first GPU picker is a single-GPU scheduler when jobs <= MAXPG.**
`if on_gpu(0) < MAXPG -> 0 elif on_gpu(1) < MAXPG -> 1` was fine at MAXPG=3. After
`--fast-attn` cut memory to ~2.1 GiB/job I raised it to 6, and with exactly 6 jobs
GPU 0 never filled: **one 4090 sat at 0% for three hours** while the log happily
reported six successful launches. **Why:** the optimisation created the bottleneck.
**How to apply:** pick the LESS LOADED device. And do NOT then interleave job types
to "balance" — alternating types against an alternating picker phase-locks and puts
every long job on one device. I did exactly that and reproduced the bug I was fixing.
Check `nvidia-smi`, never the launch log.

**2. Do not infer held-out accuracy from training loss.** I predicted a condition
would be null because its index arm reached 0.03 training loss; it scored **0.674
held-out against 0.949**. **Why:** the r=-0.996 affine relation between loss and
accuracy holds in some regimes and not others, and both arms being at low loss says
nothing about the gap. **How to apply:** wait for the eval. Never trail a prediction
off a loss curve.

**3. A wide pre-registered band is not a pre-registration.** My three outcomes were
"between -0.010 and +0.374", "near -0.010", "at/above +0.374". The middle case
mechanically fired on +0.015, which is 0.025 from the null reference against a 0.150
noise floor — i.e. identical to it. **How to apply:** set branch boundaries against
the MEASURED NOISE FLOOR, not against the endpoints, and make the branches disjoint.

**4. Editing a running bash script is unsafe.** Bash reads by byte offset; an insert
before the read point makes it resume mid-token. Kill and relaunch — cheap when the
script is parked in a wait loop, and children survive killing the parent.

Related: [[feedback_convergence_first]], [[feedback_cwd_aggregator_bug]].


## The `pkill -f` self-match trap, hit AGAIN 2026-09-06

`pkill -u $USER -f "_resolve.py"` matched the shell that typed it and killed my own
command mid-task. This is documented twice in CLAUDE.md and I still did it. The rule
is not "split the pattern" — that only protects a script from itself. **Do not use
`pkill -f` at all.** Filter by `ps -o comm=` and require a real interpreter, or just
let the background job finish, which is what should have happened here (it already had).


## CPU thread oversubscription and mid-line LaTeX comments (2026-09-14)

- **CPU threads.** Ten concurrent GPU jobs with uncapped CPU threads gave load average 52 on 32 cores and ran 3x
  slow. `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` in every launcher fixed it: 24 -> 6 s/epoch.
- **Mid-line `%`.** A scripted edit put `% src:` mid-line in report.tex and silently deleted two sentences from
  the PDF. The build log showed nothing. **How to apply:** comments go on their own lines. After scripted .tex
  edits, grep for text after `%` on the same line, and pdftotext the result.
- **Row-exact verification.** A vectorised generator looked right and was wrong on 92/3200 rows (the carry
  overwrote the top digit). Verify fast paths row-exact against the reference code before using them.
- **Compile check.** For torch.compile, compare eager vs compiled loss curves on identical batches per arm before
  a batch uses it (`verify_addition_compile.py`).


**DUPLICATE DRIVERS (2026-09-20).** A launcher whose log you cannot find may already be running. I
started the same batch three times; three drivers raced on the same seeds, and only a `ps` check
before they collided prevented duplicated runs. Check `ps` for the driver by name before relaunching,
and kill extras by PID (never `pkill -f`, which matches your own shell).


## Orphans, blind round-robin, stale checkpoints, unread .done (2026-09-21..23, code batches)

- **Killing a supervisor does not kill its children.** The launcher survived as an orphan and the next
  supervisor started a second copy: 16 launches for 12 runs, ~10 GPU-hours lost. **How to apply:** every
  driver takes `exec 9>/tmp/.mapformer_<name>.lock; flock -n 9 || exit 0` (see `run_code2048_fill.sh`).
- **`cuda:$((i%2))` is not a scheduler.** Runs finish out of order, the counter desynchronises from which
  card is free, and two 13.5-14.7 GiB runs landed on one 24 GiB card and OOM-died. Pick by ACTUAL per-card
  occupancy (`gpu_busy` over `ps comm=,args=` in `run_code2048_fill.sh`). Rule 13 again, reintroduced by me.
- **A checkpoint can be stale and load cleanly.** A duplicate launch overwrote a finished run's best.pt with
  an iter-15000 one; it was read as "an anomalous unstable seed". Evaluators must cross-check the stored
  val metric against the run JSON and refuse on mismatch (`eval_code_long.py` does).
- **Check the `.done` marker before calling a batch unfinished.** C1 was reported unfinished when done; C2
  sat complete and unread for a day.
