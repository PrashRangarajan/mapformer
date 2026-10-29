#!/bin/bash
# Closed-loop goal-directed eval on torus. No new training; evaluates the
# Cluster D experiment 8 goal-directed BC checkpoints in closed-loop.
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

echo "[$(date)] [goal-closed] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [goal-closed] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.eval_goal_closedloop \
    --output-md "$REPO/GOAL_CLOSEDLOOP_RESULTS.md" \
    --n-episodes 200 \
    > "$LOGS/goal_closedloop_eval.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add GOAL_CLOSEDLOOP_RESULTS.md eval_goal_closedloop.py run_goal_closedloop.sh
git commit -m "Closed-loop goal-directed eval on torus

Evaluates Cluster D experiment 8 BC-trained goal-directed checkpoints in
closed loop: model picks actions, env executes, agent must reach goal.
No new training. Compares success rate to open-loop match-acc.

Tests whether the cognitive map supports re-planning on the fly when the
model's actions are executed in the env, vs just predicting the expert's
next action open-loop.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [goal-closed] Done."
