#!/bin/bash
# Linear-probe frozen lm200 prediction-trained models for goal-directed info.
# No backbone training — just freeze + linear head. Quick (~10 min total).
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

is_gpu_free() {
    local mem util
    while IFS=', ' read -r mem util; do
        mem=${mem//[^0-9]/}; util=${util//[^0-9]/}
        [ -z "$mem" ] && mem=0; [ -z "$util" ] && util=0
        if [ "$mem" -ge 5000 ] || [ "$util" -ge 50 ]; then return 1; fi
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 0
}

echo "[$(date)] [Probe] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [Probe] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_goal_linear \
    --output-md "$REPO/PROBE_GOAL_RESULTS.md" \
    > "$LOGS/probe_goal.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add PROBE_GOAL_RESULTS.md probe_goal_linear.py run_probe_goal.sh
git commit -m "Linear probe of frozen lm200 models for goal-directed action info

Tests whether prediction-trained cognitive maps encode goal-directed
action info BEFORE any goal-directed training. Frozen backbone, single
linear head, BFS-optimal action target. Result in PROBE_GOAL_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [Probe] Done."
