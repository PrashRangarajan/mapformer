#!/bin/bash
# Memory Maze CPU gate v2, blind stretches (MEMORY_MAZE_GATE_V2.md). CPU only; account, partition, output path and
# data/venv locations are given on the sbatch command line (--export=ALL,MM_DATA=...,MM_PY=...,MM_OUT=...,MM_GATE_DIR=...).
# MM_GATE_DIR must hold mm_gate_v2.py AND mm_gate.py (v2 imports v1's helpers).
# Writes <MM_OUT>/gate_v2_15x15.{json,log}, then gate_v2_15x15.done (artifact first, then marker).
set -uo pipefail
: "${MM_DATA:?}" "${MM_PY:?}" "${MM_OUT:?}" "${MM_GATE_DIR:?}"
mkdir -p "$MM_OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "=== $(date -Is) on $(hostname) ==="
"$MM_PY" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
for f in 15x15/eval 15x15/train-part0; do [ -f "$MM_DATA/$f.ok" ] || { echo "missing $f (no .ok)"; exit 3; }; done
[ -f "$MM_OUT/gate_v2_15x15.done" ] && { echo "already done"; exit 0; }
"$MM_PY" -I "$MM_GATE_DIR/mm_gate_v2.py" --eval-zip "$MM_DATA/15x15/eval.zip" --train-zip "$MM_DATA/15x15/train-part0.zip" \
    --n-eval "${MM_NEVAL:-1000}" --n-train "${MM_NTRAIN:-300}" --out "$MM_OUT/gate_v2_15x15.json" > "$MM_OUT/gate_v2_15x15.log" 2>&1
rc=$?
echo "$(date -Is) rc=$rc"; tail -3 "$MM_OUT/gate_v2_15x15.log"
[ $rc -eq 0 ] && [ -s "$MM_OUT/gate_v2_15x15.json" ] && touch "$MM_OUT/gate_v2_15x15.done"
echo "=== $(date -Is) end ==="
