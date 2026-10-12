#!/usr/bin/env bash
# CLOCK_SIDE_PREREG.md calibration pilot: Signed_r4 and CARoPE_r4 at seed 100 on three (k_max, T) settings, GPU 0 only.
set -uo pipefail
REPO=/home/prashr/mapformer; R=$REPO/runs/clock_side_calib; mkdir -p "$R"; LOG=$R/calib.log
exec 9>"$REPO/.clock_side_calib.lock"; flock -n 9 || exit 1
echo "$(date '+%F %T') code $(cd $REPO && git rev-parse HEAD) $(cd $REPO && git diff --quiet && echo clean || echo DIRTY)" >> "$LOG"
cd /home/prashr
for cfg in "128 1024" "128 2048" "256 2048"; do set -- $cfg; K=$1; T=$2
  for V in Signed_r4 CARoPE_r4; do
    OUT="$R/k${K}_T${T}/${V}_s100"; mkdir -p "$OUT"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_recency --variant "$V" --seed 100 --k-max "$K" --T "$T" \
      --eval-T "$T" $((2 * T)) --epochs 300 --n-batches 48 --batch-size 16 --schedule cosine --lr 1e-3 --n-layers 1 \
      --d-model 128 --n-heads 2 --fast-attn --device cuda:0 --output-dir "$OUT" > "$OUT.log" 2>&1 < /dev/null &
    echo "$(date '+%F %T') launched k$K T$T $V" >> "$LOG"; sleep 20
  done
done
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_recency/ && /clock_side_calib/' | grep -q .; do sleep 60; done
echo "$(date '+%F %T') done" >> "$LOG"; touch "$REPO/.clock_side_calib_done"
