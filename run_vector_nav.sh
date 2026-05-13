#!/bin/bash
# Vector navigation probe — Tolman cognitive-map test.
# Eval-only on existing prediction-trained lm200 checkpoints (multi-seed).
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

echo "[$(date)] [vector-nav] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [vector-nav] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_vector_nav \
    --output-md "$REPO/VECTOR_NAV_RESULTS.md" \
    --n-train-trajs 100 --n-eval-trajs 50 --n-landmarks-per-traj 5 \
    > "$LOGS/vector_nav.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add VECTOR_NAV_RESULTS.md probe_vector_nav.py run_vector_nav.sh
git commit -m "Vector navigation probe (Tolman cognitive-map test)

Probes whether prediction-trained representations support navigation to
arbitrary landmarks without goal-directed training. Frozen backbone, linear
head with learnable landmark embedding, predicts BFS-optimal next action
given (hidden state, target landmark id).

Key split: test accuracy on landmarks that were vs were not visited during
the random walk. If similar, the model has integrated landmark positions
even before encountering them — a true cognitive map.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [vector-nav] Done."
