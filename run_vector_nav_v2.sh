#!/bin/bash
# Vector navigation v2: pair-based probe (h_t, h_s) -> BFS direction.
# Proper Tolman test with shortcut filter.
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

echo "[$(date)] [vector-nav-v2] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [vector-nav-v2] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_vector_nav_v2 \
    --output-md "$REPO/VECTOR_NAV_V2_RESULTS.md" \
    --n-train-trajs 100 --n-eval-trajs 50 --pairs-per-traj 30 \
    > "$LOGS/vector_nav_v2.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add VECTOR_NAV_V2_RESULTS.md probe_vector_nav_v2.py run_vector_nav_v2.sh
git commit -m "Vector navigation probe v2: paired-hidden-states Tolman test

Linear probe over [h_t; h_s] (two hidden states from the same random-walk
trajectory) -> BFS-optimal next action from pos_t to pos_s.

Following Banino 2018 / Whittington 2020 / Tolman 1948: tests whether the
model's representation supports linear extraction of relative spatial
position. Shortcut filter: pairs where the agent visited both cells but
never traversed the direct path between them (the proper Tolman test).

Strong weight decay (1e-2) on the probe to prevent memorization.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [vector-nav-v2] Done."
