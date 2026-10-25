#!/usr/bin/env bash
# IMAGE_TORUS pilot (IMAGE_TORUS_PREREG.md, "Pilot"): timing / memory, seeds 3180-3181 OUTSIDE the batch (rule 29).
# Run only after .ssm_sign_a2_done and .factorial_n8_done exist. One GPU, given explicitly:
#   cd /home/prashr/mapformer && DRV_GPUS=1 setsid nohup bash run_image_torus_pilot.sh > /dev/null 2>&1 &
# Optional: PILOT_B=1 adds I_RoPE_L4 to the solo stage; PILOT_LEARN=1 adds stage 4 (150-epoch learning check, ~1 h).
# Output: runs/image_torus_pilot/, image_torus_pilot.log, IMAGE_TORUS_PILOT.txt (read by hand; no decision is automatic).
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/image_torus_pilot.log"; P="$REPO/runs/image_torus_pilot"; OUTF="$REPO/IMAGE_TORUS_PILOT.txt"
G="${DRV_GPUS:-}"
[[ "$G" =~ ^[0-9]$ ]] || { echo "set DRV_GPUS to ONE device index" >> "$LOG"; exit 1; }
source "$REPO/lib_driver.sh"
drv_lock "$REPO/.run_image_torus_pilot.lock" || exit 1
for mk in .ssm_sign_a2_done .factorial_n8_done; do
  [ -f "$REPO/$mk" ] || [ "${DRV_SKIP_MARKERS:-0}" = 1 ] || { _drv_log "REFUSED: $mk missing"; exit 1; }
done
[ "$(drv_ntrain "$G")" -eq 0 ] || { _drv_log "REFUSED: trainers already on cuda:$G"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$P"; cd "$REPO/.."
_drv_log "pilot start at $(cd "$REPO" && git rev-parse HEAD); GPU $G"
COMMON=(--lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --schedule cosine --data-workers 3 --fast-attn)
memmon(){ while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" >> "$1"; sleep 2; done; }
peak(){ sort -n "$1" | tail -1; }
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G")
echo "# IMAGE_TORUS pilot $(date '+%F %T'), GPU $G ($(nvidia-smi --query-gpu=name --format=csv,noheader -i "$G")), baseline ${base} MiB" > "$OUTF"
sepoch(){ grep -o '| [0-9.]*s$' "$1" | tr -d '|s ' | tail -n +2 | sort -n | awk '{a[NR]=$1} END{print (NR? a[int((NR+1)/2)] : "NA")}'; }
# stage 1: solo, 10 epochs each
SOLO=("T_MapWM:MapWM:token:1:" "I_MapWM:MapWM:image:1:" "I_MapWM_bench:MapWM:image:1:--cudnn-benchmark" "I_RoPE:RoPE:image:1:")
[ "${PILOT_B:-0}" = 1 ] && SOLO+=("I_RoPE_L4:RoPE:image:4:" "T_RoPE_L4:RoPE:token:4:")
echo "## stage 1: solo, 10 epochs (median s/epoch over the 5-epoch prints after the first; peak MiB above baseline)" >> "$OUTF"
for C in "${SOLO[@]}"; do
  IFS=: read -r NAME ARM OBS NL XF <<< "$C"; O="$P/solo/$NAME"; mkdir -p "$O"; : > "$O/mem.txt"
  memmon "$O/mem.txt" & MM=$!
  OMP_NUM_THREADS=3 python3 -u -m mapformer.train_image --arm "$ARM" --obs "$OBS" --n-layers "$NL" --seed 3180 --epochs 10 \
    "${COMMON[@]}" $XF --device "cuda:$G" --output-dir "$O" > "$O/train.log" 2>&1
  kill "$MM" 2>/dev/null; wait "$MM" 2>/dev/null
  echo "$NAME: $(sepoch "$O/train.log") s/epoch (whole run incl. start-up $(grep -o '[0-9.]* s/epoch' "$O/train.log")); peak +$(( $(peak "$O/mem.txt") - base )) MiB; $(grep -o 'params=[^ ]*' "$O/train.log")" >> "$OUTF"
done
# stage 2: block A's four arms at once (the batch's packing), 15 epochs, seed 3181
echo "## stage 2: contended, block A's four arms at once, 15 epochs" >> "$OUTF"
: > "$P/cont_mem.txt"; memmon "$P/cont_mem.txt" & MM=$!
PIDS=()
for C in "I_MapWM:MapWM:image" "I_ActOnly:ActOnly:image" "I_RoPE:RoPE:image" "T_MapWM:MapWM:token"; do
  IFS=: read -r NAME ARM OBS <<< "$C"; O="$P/cont/$NAME"; mkdir -p "$O"
  OMP_NUM_THREADS=3 python3 -u -m mapformer.train_image --arm "$ARM" --obs "$OBS" --seed 3181 --epochs 15 "${COMMON[@]}" \
    --device "cuda:$G" --output-dir "$O" > "$O/train.log" 2>&1 & PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done                 # wait on the trainers only, never the monitor
kill "$MM" 2>/dev/null; wait "$MM" 2>/dev/null
for NAME in I_MapWM I_ActOnly I_RoPE T_MapWM; do
  echo "$NAME: $(sepoch "$P/cont/$NAME/train.log") s/epoch (whole run $(grep -o '[0-9.]* s/epoch' "$P/cont/$NAME/train.log"))" >> "$OUTF"
done
echo "contended peak +$(( $(peak "$P/cont_mem.txt") - base )) MiB total" >> "$OUTF"
# stage 3: evaluator wall time on one image checkpoint and one token checkpoint
echo "## stage 3: evaluator wall time per run (all readouts + probe)" >> "$OUTF"
for NAME in I_MapWM T_MapWM; do
  t0=$(date +%s)
  OMP_NUM_THREADS=3 python3 -u -m mapformer.eval_image_torus --device "cuda:$G" --ckpt "$P/solo/$NAME/$NAME.pt" > "$P/solo/$NAME/eval.log" 2>&1
  echo "$NAME eval: $(( $(date +%s) - t0 )) s" >> "$OUTF"
done
# stage 4 (optional): does the image path learn at all? 150 epochs, seed 3181; training loss only
if [ "${PILOT_LEARN:-0}" = 1 ]; then
  echo "## stage 4: learning check, 150 epochs (training loss at epochs 50 / 100 / 150; eval NOT read)" >> "$OUTF"
  PIDS=()
  for C in "I_MapWM:MapWM:image" "I_RoPE:RoPE:image" "T_MapWM:MapWM:token"; do
    IFS=: read -r NAME ARM OBS <<< "$C"; O="$P/learn/$NAME"; mkdir -p "$O"
    OMP_NUM_THREADS=3 python3 -u -m mapformer.train_image --arm "$ARM" --obs "$OBS" --seed 3181 --epochs 150 "${COMMON[@]}" \
      --device "cuda:$G" --output-dir "$O" > "$O/train.log" 2>&1 & PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  for NAME in I_MapWM I_RoPE T_MapWM; do
    echo "$NAME: $(grep -E 'Epoch +(50|100|150)/' "$P/learn/$NAME/train.log" | sed 's/.*Loss: \([0-9.]*\).*/\1/' | tr '\n' ' ')" >> "$OUTF"
  done
fi
_drv_log "pilot done"; echo "done $(date '+%F %T')" >> "$OUTF"
