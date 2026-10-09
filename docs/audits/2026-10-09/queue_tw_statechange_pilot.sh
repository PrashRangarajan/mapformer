#!/usr/bin/env bash
# Queued on the user's request (2026-10-09): after the GAIN_PHASE driver exits, run the TW_STATECHANGE pilot exactly as
# TW_STATECHANGE_PREREG.md "Pilot" specifies -- (a) GPU bitwise reproduction check, (b) the four arms at seed 150, 900
# epochs, 2 per GPU, (c) the pilot readouts end to end. It does NOT launch the batch: the pilot is reviewed and written
# up as an amendment first (rule 29). Log: runs/tw_statechange_pilot/queue.log; marker .tw_statechange_pilot_done.
set -u
REPO=/home/prashr/mapformer; P=$REPO/runs/tw_statechange_pilot; mkdir -p "$P/p0"; LOG=$P/queue.log
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
log "queued; waiting for the GAIN_PHASE driver"
while ps -u "$USER" -o args= | grep -qx "bash run_gain_phase.sh"; do sleep 300; done
# also wait until no GAIN_PHASE trainer or eval remains on a GPU
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /runs\/gain_phase\//' | grep -q .; do sleep 120; done   # rule 23: comm-matched, never grep the pattern
log "GAIN_PHASE finished; marker present: $([ -e $REPO/.gain_phase_done ] && echo yes || echo NO)"
cd /home/prashr
log "(a) reproduction check"
PYTHONPATH=/home/prashr python3 -u "$REPO/docs/audits/2026-10-08/tw_statechange_repro.py" 5 cuda:0 \
  > "$REPO/docs/audits/2026-10-08/tw_statechange_repro_gpu_out.txt" 2>&1; log "(a) exit $?"
log "(b) four arms at seed 150"
i=0; for ARM in MapWM NormStep DirOnly RoPE; do G=$((i % 2)); i=$((i+1))
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_tw_statechange --arm "$ARM" --seed 150 --device "cuda:$G" \
    --output-dir "$P/p0/${ARM}_s150" > "$P/p0/${ARM}_s150.log" 2>&1 < /dev/null &
  log "launched $ARM s150 on cuda:$G"; sleep 20
done
sleep 120
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_tw_statechange/ && /--seed 150/' | grep -q .; do sleep 120; done
log "(b) done: $(ls $P/p0/*/*.pt 2>/dev/null | wc -l) checkpoints"
log "(c) pilot readouts"
PYTHONPATH=/home/prashr python3 -u -m mapformer.analyze_tw_statechange --readouts --runs-dir "$P/p0" --seeds 150 \
  --out "$P/TWSC_PILOT.json" > "$P/pilot_readouts_out.txt" 2>&1; log "(c) exit $?"
touch "$REPO/.tw_statechange_pilot_done"; log "pilot complete; batch NOT launched (review + amendment first)"
