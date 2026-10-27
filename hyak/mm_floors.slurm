#!/bin/bash
# Gate floors on the held-out secondary scoring split (train-part0 1600-2899), CPU only (MEMORY_MAZE_PREREG.md).
#   sbatch --cpus-per-task=8 --mem=48G --time=03:00:00 --requeue --export=ALL,MM_PROJ=..,MM_ZIPS=..,MM_OUT=..,MM_PY=.. mm_floors.slurm
set -uo pipefail
: "${MM_PROJ:?}" "${MM_ZIPS:?}" "${MM_OUT:?}" "${MM_PY:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$MM_OUT"
[ -f "$MM_OUT/floors_held.done" ] && { echo "done"; exit 0; }
"$MM_PY" -I "$MM_PROJ/mapformer/mm_floors.py" --zip "$MM_ZIPS/15x15/train-part0.zip" --out "$MM_OUT/floors_held.json" \
  > "$MM_OUT/floors_held.log" 2>&1 && touch "$MM_OUT/floors_held.done"
echo "rc=$?"
