#!/bin/bash
# Memory Maze CPU gate (MEMORY_MAZE_GATE.md). CPU only; account, partition, output path and the data/venv locations
# are given on the sbatch command line (--export=ALL,MM_DATA=...,MM_PY=...,MM_OUT=...), never in this file.
#   sbatch --cpus-per-task=8 --mem=48G --time=03:00:00 --requeue --export=ALL,MM_DATA=..,MM_PY=..,MM_OUT=.. mm_gate.slurm
# Needs <MM_DATA>/15x15/{eval,train-part0}.ok (and 9x9/eval.ok for the descriptive 9x9 run). Writes
# <MM_OUT>/gate_<size>.json and .log, then <MM_OUT>/gate_<size>.done (artifact first, then marker).
set -uo pipefail
: "${MM_DATA:?}" "${MM_PY:?}" "${MM_OUT:?}"
cd "$(dirname "$0")" 2>/dev/null || true
GATE="${MM_GATE:-$PWD/mm_gate.py}"
mkdir -p "$MM_OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "=== $(date -Is) on $(hostname) ==="
"$MM_PY" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
TRAIN="$MM_DATA/15x15/train-part0.zip"
[ -f "$MM_DATA/15x15/train-part0.ok" ] || { echo "missing 15x15 train-part0 (no .ok)"; exit 3; }
for size in 15x15 9x9; do
  [ -f "$MM_OUT/gate_$size.done" ] && { echo "skip $size (done)"; continue; }
  [ -f "$MM_DATA/$size/eval.ok" ] || { echo "missing $size eval (no .ok)"; continue; }
  "$MM_PY" -I "$GATE" --eval-zip "$MM_DATA/$size/eval.zip" --train-zip "$TRAIN" --n-eval "${MM_NEVAL:-1000}" \
      --n-train "${MM_NTRAIN:-300}" --label "$size" --out "$MM_OUT/gate_$size.json" > "$MM_OUT/gate_$size.log" 2>&1
  rc=$?
  echo "$(date -Is) $size rc=$rc"; tail -3 "$MM_OUT/gate_$size.log"
  [ $rc -eq 0 ] && [ -s "$MM_OUT/gate_$size.json" ] && touch "$MM_OUT/gate_$size.done"
done
echo "=== $(date -Is) end ==="
