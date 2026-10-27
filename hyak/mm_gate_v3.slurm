#!/bin/bash
# Memory Maze CPU gate v3 (MEMORY_MAZE_GATE_V3.md): cell B16/S16 on fresh train-part0 episodes. CPU only; account,
# partition, output path and data/venv locations on the sbatch command line
# (--export=ALL,MM_DATA=...,MM_PY=...,MM_OUT=...,MM_GATE_DIR=...). MM_GATE_DIR holds mm_gate.py, mm_gate_v2.py, mm_gate_v3.py.
# Writes <MM_OUT>/gate_v3_15x15.{json,log}, then gate_v3_15x15.done (artifact first, then marker).
set -uo pipefail
: "${MM_DATA:?}" "${MM_PY:?}" "${MM_OUT:?}" "${MM_GATE_DIR:?}"
mkdir -p "$MM_OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "=== $(date -Is) on $(hostname) ==="
"$MM_PY" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
[ -f "$MM_DATA/15x15/train-part0.ok" ] || { echo "missing 15x15 train-part0 (no .ok)"; exit 3; }
[ -f "$MM_OUT/gate_v3_15x15.done" ] && { echo "already done"; exit 0; }
"$MM_PY" -I "$MM_GATE_DIR/mm_gate_v3.py" --zip "$MM_DATA/15x15/train-part0.zip" --out "$MM_OUT/gate_v3_15x15.json" \
    > "$MM_OUT/gate_v3_15x15.log" 2>&1
rc=$?
echo "$(date -Is) rc=$rc"; tail -3 "$MM_OUT/gate_v3_15x15.log"
[ $rc -eq 0 ] && [ -s "$MM_OUT/gate_v3_15x15.json" ] && touch "$MM_OUT/gate_v3_15x15.done"
echo "=== $(date -Is) end ==="
