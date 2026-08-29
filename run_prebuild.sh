#!/usr/bin/env bash
# Serial buffer prebuild for the aliasing sweep. See prebuild_buffers.py for why.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
LOG="$REPO/prebuild.log"; echo "prebuild start $(date)" > "$LOG"
python3 -u -m mapformer.prebuild_buffers --grid-size 32 --n-obs 256 64 \
    --seeds 0 1 2 3 4 --n-steps 512 --buffer-size 24000 --eval-trials 128 \
    --n-workers 24 --oracle >> "$LOG" 2>&1
python3 -u -m mapformer.prebuild_buffers --grid-size 32 --n-obs 16 \
    --seeds 3 4 --n-steps 512 --buffer-size 24000 --eval-trials 128 \
    --n-workers 24 --oracle >> "$LOG" 2>&1
touch "$REPO/.prebuild_done"; echo "$(date) DONE" >> "$LOG"
