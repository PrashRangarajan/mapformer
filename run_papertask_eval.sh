#!/usr/bin/env bash
# Re-run the PAPERTASK evaluation, which OOM'd because SPREAD2 had already started on the same
# GPUs. Training is complete; only the eval is missing. Waits for real GPU headroom, keeps the
# ORIGINAL protocol layout (8 x 32, matching PAPER_TASK_FLOORS.md), and verifies the artifact
# before declaring success.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

echo "$(date +%H:%M) waiting for training to clear"
while [ "$(busy)" -gt 0 ]; do sleep 60; done
for g in 1 0; do
  if [ "$(free_mib $g)" -gt 12000 ]; then GPU=$g; break; fi
done
GPU=${GPU:-1}
echo "$(date +%H:%M) using cuda:$GPU with $(free_mib $GPU) MiB free"

python3 -u -m mapformer.eval_paper_ood --runs-dir "$REPO/runs/paper_task_rerun" \
  --variants Vanilla VanillaEM_P0 MapPoPE-Flat --seeds 0 1 2 3 4 5 6 7 \
  --extended --n-batches 8 --batch-size 32 --device "cuda:$GPU" \
  --out "$REPO/PAPER_OOD_RERUN.md" > "$REPO/runs/paper_task_rerun/eval_ood.log" 2>&1
echo "eval exit $?"
[ -s "$REPO/PAPER_OOD_RERUN.json" ] || { echo "EVAL PRODUCED NO JSON -- not analysing"; tail -5 "$REPO/runs/paper_task_rerun/eval_ood.log"; exit 1; }
python3 -u -m mapformer.analyze_papertask > "$REPO/runs/paper_task_rerun/analysis.log" 2>&1
echo "analysis exit $?"
tail -45 "$REPO/runs/paper_task_rerun/analysis.log"
