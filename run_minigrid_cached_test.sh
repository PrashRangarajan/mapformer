#!/bin/bash
# Validate the MiniGridWorld_Cached path with a single training run.
# Compares speed against the slow live trainings already running.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

# Run on GPU 0 (currently 4% utilisation, 22 GB free — current trainings
# are CPU-bound on gym.step so they won't compete with us once buffer is built)
echo "[$(date)] Starting cached-buffer validation: Level15 on DoorKey clean s0"
mkdir -p "$REPO/runs/minigrid_doorkey_clean_cached/Level15/seed0"
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Level15 --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 \
    --env minigrid_doorkey --minigrid-tokenization obj_color \
    --minigrid-cached-buffer 25000 \
    --device cuda \
    --output-dir mapformer/runs/minigrid_doorkey_clean_cached/Level15/seed0 \
    > "$LOGS/mg_doorkey_cached_Level15_s0.log" 2>&1
echo "[$(date)] Cached training done."
tail -10 "$LOGS/mg_doorkey_cached_Level15_s0.log"
