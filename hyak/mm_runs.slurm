#!/bin/bash
# Memory Maze runs as a Slurm array (MEMORY_MAZE_PREREG.md). One run per array task, from a task table file with one
# line per task: "<ARM> <SEED> [tag=<name>] [extra train_mm args]" (no comments; line i+1 = task i). The run dir is
# <MM_RUNS>/<ARM>_s<SEED>[_<name>]. A run resumes from its per-epoch
# ckpt.pt and is skipped when its DONE marker exists (DONE is written only after results.json). Requeued on preemption;
# a TIMEOUT is not requeued, so submit a chain with --dependency=afterany:<previous array id> (resume continues).
#   sbatch --array=0-N --gpus=1 --constraint="a40|a100|l40" --cpus-per-task=4 --mem=48G --time=08:00:00 --requeue \
#     --open-mode=append --export=ALL,MM_PROJ=..,MM_ARR=..,MM_RUNS=..,MM_TASKS=..,MM_PY=.. mm_runs.slurm
set -uo pipefail
: "${MM_PROJ:?}" "${MM_ARR:?}" "${MM_RUNS:?}" "${MM_TASKS:?}" "${MM_PY:?}"
cd "$MM_PROJ" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONPATH="$MM_PROJ"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MM_TASKS")
[ -n "$LINE" ] || { echo "no task line $SLURM_ARRAY_TASK_ID"; exit 2; }
read -r ARM SEED EXTRA <<< "$LINE"
TAG=""
if [[ "$EXTRA" == tag=* ]]; then TAG="_${EXTRA%% *}"; TAG="_${TAG#_tag=}"; EXTRA="${EXTRA#* }"; [[ "$EXTRA" == tag=* ]] && EXTRA=""; fi
OUT="$MM_RUNS/${ARM}_s${SEED}${TAG}"
echo "=== $(date -Is) task $SLURM_ARRAY_TASK_ID: $ARM s$SEED [$EXTRA] on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$OUT/DONE" ] && { echo "DONE exists; skip"; exit 0; }
[ -f "$MM_ARR/PREP.done" ] || { echo "arrays not prepared"; exit 3; }
# shellcheck disable=SC2086
"$MM_PY" -u -m mapformer.train_mm --arm "$ARM" --seed "$SEED" --data "$MM_ARR" --out "$OUT" --device cuda $EXTRA
echo "=== $(date -Is) end rc=$? ==="
