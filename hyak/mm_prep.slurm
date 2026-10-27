#!/bin/bash
# Memory Maze prep (MEMORY_MAZE_PREREG.md): frame autoencoder on train-part0 0-299, embeddings of train-part0 and eval
# as a few large arrays, E map fit, encoder pose-leak checks. One GPU. Resumable (each stage skips if its marker exists).
# Account, partition, output path, data locations on the sbatch command line:
#   sbatch --gpus=1 --constraint="a40|a100|l40" --cpus-per-task=8 --mem=64G --time=08:00:00 --requeue \
#     --open-mode=append --export=ALL,MM_PROJ=..,MM_ZIPS=..,MM_ARR=..,MM_PY=.. mm_prep.slurm
set -uo pipefail
: "${MM_PROJ:?}" "${MM_ZIPS:?}" "${MM_ARR:?}" "${MM_PY:?}"
cd "$MM_PROJ" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONPATH="$MM_PROJ"
echo "=== $(date -Is) on $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$MM_ARR/PREP.done" ] && { echo "prep already done"; exit 0; }
"$MM_PY" -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)"
"$MM_PY" -u mapformer/mm_encoder.py prep --zips "$MM_ZIPS" --out "$MM_ARR" --device cuda
echo "=== $(date -Is) end rc=$? ==="
