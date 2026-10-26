# Running batches on a Slurm GPU cluster (Hyak): lessons

What we know about running this project's work on the university's Slurm cluster. Connection details (host, user,
SSH configuration, account and storage paths) are deliberately NOT in this repository: on the machine that has
access they are in a local, uncommitted file outside the repo. Ask the user for them on a new machine.

## Connecting from an agent session

- Logins need an interactive second factor (Duo) that an agent cannot do. The workstation shares one SSH master
  connection; while it is alive, non-interactive `ssh` works. Check it with `ssh -O check <alias>`, and always run
  remote commands with `ssh -o BatchMode=yes` so a dead connection fails at once instead of waiting for a password.
- If a command fails with `Permission denied (gssapi-keyex,gssapi-with-mic,keyboard-interactive)`, stop and ask the
  user to re-authenticate in their own terminal. Do not retry in a loop (repeated failures can block the IP). Expect
  to need re-authentication about every 8 hours.
- Login nodes are for editing and submitting only: no training, and never import torch with CUDA there (it hangs).
- Run `rsync` on the workstation, never on the cluster (it copies from the machine it runs on). Fetch results, not
  checkpoints: the workstation's disk is nearly full.

## Choosing where jobs run

- The free, preemptible checkpoint partition is the default. Jobs there are requeued on preemption (`--requeue`)
  but NOT on timeout, and requested limits above about 9 h were clamped to about 9 h. Plan for 8-9 h windows: make
  runs resumable (save and resume every epoch) and chain jobs with `--dependency=afterany:<jobid>` if longer.
- The GPU pool is mixed. Always restrict to modern cards (e.g. `--constraint="a40|a100|l40"`); older nodes have no
  kernels for current torch and fail with `no kernel image is available for execution on the device`.
- A priority (non-preemptible) partition exists for short urgent jobs; it draws on a shared departmental allocation.
- Array jobs are the cheap way to parallelise (one run per task). Packing several jobs on one GPU contends rather
  than overlaps there (utilisation is already high).
- The free queue can be busy: a 1-hour job waited about 30-55 min to start (2026-10-25).

## Storage

- Every shared filesystem is at 96-99% (2026-10-25). A full allocation kills a job with exit 1 and no traceback (a
  truncated log, or `Disk quota exceeded`). Check `df -h` on the target directory before writing anything large.
- The large scratch space is purged when idle (about 30 days) and has silently lost checkpoints before: keep anything
  you need to reload elsewhere, or copy it back. The lab's persistent allocation is shared: ask the user before
  writing tens of GB there.
- The cluster filesystem (GPFS) is slow at many small files: `pip install` takes many minutes, and datasets should
  be stored as a few large arrays, not many small files.
- Slurm does not create the `--output` directory; create `logs/` first or the job fails before the script runs.

## Software

- System `python3` is old (3.6); build environments with `python3.12 -m venv` in a project directory on the cluster
  storage, never in the small home directory. conda is not on PATH; a failed `conda activate` in a pasted block lets
  `pip` install into home.
- Torch there (2.13 at last check) differs from the workstation's (2.10): treat cluster batches as separate batches,
  never pooled with workstation runs (rule 12), and record the version.
- This repository hard-codes its workstation path; on the cluster rewrite paths in the cluster copy only (see
  `docs/RUNBOOK_NEXT_EXPERIMENTS.md` Sec. 1) and never commit that change.
- Memory Maze (2026-10-25): `memory-maze` 1.0.3 with `dm_control` 1.0.50, `mujoco` 3.17 and `gym` 0.26.2 (needed
  even for the dm_env interface) works with `MUJOCO_GL=egl` on an A40 compute node; OSMesa is not available.
  Throughput is CPU-bound: about 36 env steps/s per process at 9x9 and 21 at 15x15; 8 parallel processes ran at
  about 33 each.

## Watching jobs

- `squeue -u $USER`, `sacct -j <id> --format=JobID%20,State,Elapsed,ExitCode`, `scontrol show job <id>` (granted time
  limit, node, pending reason).
- A poller must wait for a positive success condition (the expected output files or a completion line), not for an
  empty `squeue`, which can be transient. `sacct -X` shows only the last attempt of a requeued job.
- A requeued job runs the script captured at submission; fixing the file afterwards does not change it.
- "Skip if already trained" must test for a file written only at the end, not for the output directory (a
  preempted run leaves a partial one).
- tqdm logs use `\r`: read them with `tr '\r' '\n' < log`.
