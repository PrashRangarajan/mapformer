#!/usr/bin/env bash
# GAIN_PHASE pilot (seeds 110, 111: outside the batch's 8-15 and LEAK's 0-7 / 100). Short schedule (30 epochs of
# cosine, everything else the batch's flags) to measure s/epoch at the batch's concurrency (lib_driver picker, 4 jobs
# per GPU, counting every mapformer.train_ job on the machine) and to run the eval / analysis / secondary pipeline end
# to end on real checkpoints. Launch: setsid nohup bash docs/audits/2026-10-06/gain_phase_pilot.sh > /dev/null 2>&1 &
set -uo pipefail
REPO=/home/prashr/mapformer
LOG="$REPO/docs/audits/2026-10-06/gain_phase_pilot.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_SPACING=5
drv_lock "$REPO/.gain_phase_pilot.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/gain_phase_pilot"; mkdir -p "$R/p0"
_drv_log "pilot start; code version $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY)"
for S in 110 111; do for ARM in MapWM NormStep GainRaw GainPhase; do
  OUT="$R/p0/${ARM}_s${S}"; [ -f "$OUT/${ARM}.pt" ] && continue
  mkdir -p "$OUT"; G=$(drv_wait_slot)
  _drv_log "$ARM s$S -> cuda:$G (mapformer.train_ jobs now: gpu0 $(drv_ntrain 0), gpu1 $(drv_ntrain 1))"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_gain_phase --variant "$ARM" --seed "$S" \
    --epochs 30 --n-steps 1024 --batch-size 16 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
REQ=(); for S in 110 111; do for ARM in MapWM NormStep GainRaw GainPhase; do REQ+=("$R/p0/${ARM}_s${S}/${ARM}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "pilot checkpoints missing"
T0=$(date +%s)
python3 -u -c "
import sys; sys.path.insert(0, '/home/prashr')
from mapformer.eval_gain_phase import main
main('cuda:0', runs='$R/p0', seeds=[110, 111], out='$R/GAIN_PHASE_PILOT_EVAL.json')" > "$R/eval_out.txt" 2>&1 || drv_fail "pilot eval"
_drv_log "pilot eval took $(( $(date +%s) - T0 )) s for 8 runs"
python3 -u -c "
import sys; sys.path.insert(0, '/home/prashr')
import mapformer.analyze_gain_phase as A
A.SEEDS = [110, 111]
D = A.load_runs('$R/GAIN_PHASE_PILOT_EVAL.json', '$R/p0')
A.analyse(D, seeds=[110, 111], E=30)" > "$R/analysis_out.txt" 2>&1 || drv_fail "pilot analysis"
python3 -u "$REPO/docs/audits/2026-10-06/gain_phase_secondary.py" "$R/GAIN_PHASE_PILOT_EVAL.json" "$R/p0" 110 111 \
  > "$R/secondary_out.txt" 2>&1 || drv_fail "pilot secondary"
drv_done "$R/.pilot_done" "$R/GAIN_PHASE_PILOT_EVAL.json" "$R/analysis_out.txt" "$R/secondary_out.txt"
