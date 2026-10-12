#!/usr/bin/env bash
# RING_DIAL pilot: Signed_r4 and Abs_r4 at p_plus 0.5 and 0.9, seed 130 (outside the batch), GPU 0. Timing and pipeline.
set -uo pipefail
REPO=/home/prashr/mapformer; R=$REPO/runs/ring_dial_pilot; mkdir -p "$R"; LOG=$R/pilot.log
echo "$(date '+%F %T') code $(cd $REPO && git rev-parse HEAD) $(cd $REPO && git diff --quiet && echo clean || echo DIRTY)" >> "$LOG"
cd /home/prashr
for P in 0.5 0.9; do for A in Signed_r4 Abs_r4; do
  OUT="$R/${A}_p${P}_s130"; mkdir -p "$OUT"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_ring_dial --arm "$A" --p-plus "$P" --seed 130 --device cuda:0 \
    --output-dir "$OUT" > "$OUT.log" 2>&1 < /dev/null &
  sleep 15
done; done
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_ring_dial/ && /ring_dial_pilot/' | grep -q .; do sleep 30; done
for f in "$R"/*_s130.log; do echo "$(basename "$f"): $(grep "fresh maps" "$f" | tr '\n' ' ') $(grep -c Epoch "$f") epoch lines, last $(grep Epoch "$f" | tail -1 | grep -o '[0-9.]*s$')"; done >> "$LOG"
touch "$REPO/.ring_dial_pilot_done"
