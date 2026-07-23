#!/usr/bin/env bash
# Waits for BOTH GPUs to be idle (free mem > 20 GB, i.e. the other user's jobs
# and our seq=512 enwik8 runs have finished), then launches:
#   GPU0 chain: long-sequence enwik8 efficiency confirmation (seq=2048)
#               hourglass(shorten=4) then flat10  -> hourglass_enwik8_long/
#   GPU1 chain: compositional Phase 1 (Vanilla, VanillaEM, Hourglass_k2,
#               HourglassFlat3) then eval          -> runs/comp_phase1/
# Produces COMPOSITIONAL_RESULTS.md and hourglass_enwik8_long/*.json.
# Does NOT commit/push (left to the user).
set -u
# Repo root = directory this script lives in (portable across servers).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run as a module from the repo's PARENT so `python3 -m mapformer.X` resolves.
cd "$REPO/.."
LOG="$REPO/hourglass_experiments.log"
echo "launcher start $(date)" > "$LOG"

free_mb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

wait_idle () {
  # wait until both GPUs have > 20000 MiB free
  until [ "$(free_mb 0)" -gt 20000 ] && [ "$(free_mb 1)" -gt 20000 ]; do
    echo "$(date +%H:%M) waiting: gpu0_free=$(free_mb 0) gpu1_free=$(free_mb 1)" >> "$LOG"
    sleep 120
  done
  echo "$(date +%H:%M) GPUs idle -> launching" >> "$LOG"
}

wait_idle

# ---- GPU0: long-sequence enwik8 efficiency confirmation ----
(
  OUT="$REPO/hourglass_enwik8_long"
  python3 -u -m mapformer.train_hourglass_enwik8 --model hourglass --shorten 4 \
      --seq-len 2048 --batch-size 6 --iters 8000 --eval-every 400 \
      --device cuda:0 --out "$OUT" >> "$LOG" 2>&1
  python3 -u -m mapformer.train_hourglass_enwik8 --model flat10 \
      --seq-len 2048 --batch-size 6 --iters 8000 --eval-every 400 \
      --device cuda:0 --out "$OUT" >> "$LOG" 2>&1
  echo "$(date +%H:%M) GPU0 enwik8-long chain DONE" >> "$LOG"
) &
PID0=$!

# ---- GPU1: compositional Phase 1 ----
(
  OUT="$REPO/runs/comp_phase1"
  for V in Vanilla VanillaEM Hourglass_k2 HourglassFlat3; do
    python3 -u -m mapformer.train_compositional --variant "$V" --target motif \
        --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 \
        --device cuda:1 --output-dir "$OUT" --seed 0 >> "$LOG" 2>&1
  done
  python3 -u -m mapformer.eval_compositional \
      --checkpoints "$OUT"/Vanilla.pt "$OUT"/VanillaEM.pt \
                    "$OUT"/Hourglass_k2.pt "$OUT"/HourglassFlat3.pt \
      --lengths 256 512 1024 2048 --n-traj 200 --device cuda:1 \
      --out "$REPO/COMPOSITIONAL_RESULTS.md" >> "$LOG" 2>&1
  echo "$(date +%H:%M) GPU1 compositional chain DONE" >> "$LOG"
) &
PID1=$!

wait $PID0 $PID1
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
touch "$REPO/.hourglass_experiments_done"
