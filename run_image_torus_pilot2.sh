#!/usr/bin/env bash
# IMAGE_TORUS diagnostic pilot 2 (after pilot 1's stop rule fired; IMAGE_TORUS_PREREG.md "Pilot 2"). Seeds 3182-3183,
# OUTSIDE the batch (rule 29); never pooled with the batch. 150 epochs each, pilot 1 stage-4 recipe; training loss at
# 50 / 100 / 150 and, because this is a pilot, the full held-out evaluation at the end.
#   P2a I_ActOnly  gain 30.37  s3182 lr 1e-3   does the action-only step learn with the init-matched gain?
#   P2b I_MapWM    gain 1.0    s3182 lr 1e-3   does MapWM learn when images enter the step map weakly?
#   P2c I_MapWM    gain 30.37  s3183 lr 1e-3   does pilot 1's stuck run replicate on another seed?
#   P2d I_MapWM    gain 30.37  s3182 lr 3e-4   is it an optimisation-speed (step-size) issue?
# Two jobs per GPU on GPUs 0 and 1:
#   cd /home/prashr/mapformer && setsid nohup bash run_image_torus_pilot2.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/image_torus_pilot2.log"; P="$REPO/runs/image_torus_pilot2"; OUTF="$REPO/IMAGE_TORUS_PILOT2.txt"
source "$REPO/lib_driver.sh"
drv_lock "$REPO/.run_image_torus_pilot2.lock" || exit 1
for g in 0 1; do [ "$(drv_ntrain "$g")" -eq 0 ] || { _drv_log "REFUSED: trainers already on cuda:$g"; exit 1; }; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$P"; cd "$REPO/.."
_drv_log "pilot2 start at $(cd "$REPO" && git rev-parse HEAD) (image files untracked; md5 below)"
( cd "$REPO" && md5sum model_image.py train_image.py eval_image_torus.py image_bank.py model_codes.py ) > "$P/code_md5.txt"
COMMON=(--epochs 150 --n-batches 98 --batch-size 16 --n-steps 1024 --schedule cosine --data-workers 3 --fast-attn)
RUNS=("P2a:ActOnly:30.3729:3182:1e-3:0" "P2b:MapWM:1.0:3182:1e-3:0" "P2c:MapWM:30.3729:3183:1e-3:1" "P2d:MapWM:30.3729:3182:3e-4:1")
PIDS=()
for R in "${RUNS[@]}"; do
  IFS=: read -r TAG ARM GAIN S LR G <<< "$R"; O="$P/$TAG"; mkdir -p "$O"
  ( OMP_NUM_THREADS=3 python3 -u -m mapformer.train_image --arm "$ARM" --obs image --enc-gain "$GAIN" --seed "$S" --lr "$LR" \
      "${COMMON[@]}" --device "cuda:$G" --output-dir "$O" > "$O/train.log" 2>&1 &&
    OMP_NUM_THREADS=3 python3 -u -m mapformer.eval_image_torus --device "cuda:$G" --ckpt "$O"/*.pt > "$O/eval.log" 2>&1 ) &
  PIDS+=($!); _drv_log "$TAG $ARM gain $GAIN s$S lr $LR -> cuda:$G (pid $!)"
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
{
  echo "# IMAGE_TORUS pilot 2 (diagnostic), $(date '+%F %T'); seeds 3182-3183 outside the batch; 150 epochs, cosine; never pooled"
  echo "memoryless loss level (blank 0.5, 16 classes): ln 2 + 0.5 ln 16 = 2.079"
  echo "| run | arm | gain | seed | lr | loss ep 50 / 100 / 150 | held-out acc T=1024 (test images) | zeroobj | shuffle | probe | step obj/act, inst/act |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|"
  for R in "${RUNS[@]}"; do
    IFS=: read -r TAG ARM GAIN S LR G <<< "$R"; O="$P/$TAG"
    L=$(grep -E 'Epoch +(50|100|150)/' "$O/train.log" | sed 's/.*Loss: \([0-9.]*\).*/\1/' | tr '\n' ' ')
    E=$(python3 -c "
import json,glob,sys
f=glob.glob('$O/eval_image.json')
if not f: print('| EVAL MISSING | | | | |'); sys.exit()
r=json.load(open(f[0])); a=r['acc']; st=r.get('step') or {}
print('| %.4f | %s | %.4f | %.4f | %.3f, %.3f |' % (a['test|1024'], ('%.4f' % a['zeroobj|1024']) if 'zeroobj|1024' in a else '--',
      a['shuffle|1024'], r['probe']['test_acc'], st.get('object_over_action', float('nan')), st.get('instance_over_action', float('nan'))))")
    echo "| $TAG | $ARM | $GAIN | $S | $LR | $L $E"
  done
  echo "pilot 1 (s3181, gain 30.37, lr 1e-3): I_MapWM 2.0488 2.0472 2.0458; I_RoPE 1.2883 1.2283 1.1996; T_MapWM 0.9118 0.7810 0.5178"
} > "$OUTF"
_drv_log "pilot2 done"
