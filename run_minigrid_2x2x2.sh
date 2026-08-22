#!/usr/bin/env bash
# The full factorial on MiniGrid: {RoPE, PoPE} x {index, path-integrated}
# x {flat, hierarchical}, n=3, one batch.
#
# WHY EVERYTHING IS RE-RUN AT n_layers=3 rather than adding hierarchical arms to
# the existing flat grid: the Hourglass variants IGNORE --n-layers and are always
# the 3-block scaffold. At n_layers=1 that is 614K params against the flat arms'
# 218K -- a 2.8x capacity mismatch, which is the exact confound
# EXTRAHEAD_CONTROL.md used to overturn the Hopfield claim. At n_layers=3 flat
# and hierarchical match to the parameter:
#     Vanilla/Hourglass_k2       614,538 / 614,538
#     MapPoPE-Flat/MapPoPE-Hier  615,114 / 615,114
#     RoPE/PlainHourglass        614,090 / 614,090
#
# SEVEN of eight cells. There is no "PoPE + index + hierarchy" variant in
# VARIANT_MAP; PoPE-Flat is included so the flat row is complete and the missing
# cell is named rather than quietly dropped.
#
# The 25K trajectory buffer is already cached on disk from run_minigrid_2x2.sh.
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/minigrid_2x2x2
mkdir -p "$R"
VARS="Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat"
# GPU 1 ONLY. Another user (mmattb) is on GPU 0 and his job already saturates
# it at 100% util on its own; CUDA time-slices between processes with no
# fair-share guarantee, so anything we put there costs him throughput. GPU 1
# is ours alone, so the whole grid goes there even though it means our own
# jobs contend with each other.
GPU=1

for SEED in 0 1 2; do
  for V in $VARS; do
    D="$R/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-cached-buffer 25000 \
      --device "cuda:$GPU" --output-dir "$D" \
      > "$R/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done"
done

python3 -u -m mapformer.eval_minigrid_2x2 --runs-dir "$R" \
  --variants Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat \
  --seeds 0 1 2 --lengths 128 512 1024 --device cuda:1 \
  --out mapformer/MINIGRID_2X2X2.md > "$R/eval.log" 2>&1
echo DONE; touch "$R/.done"
