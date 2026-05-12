#!/bin/bash
# SR readout: train a linear successor-representation head from frozen
# prediction-trained representations, evaluate reward-conditional planning.
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

echo "[$(date)] [sr-probe] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [sr-probe] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_sr \
    --output-md "$REPO/SR_PROBE_RESULTS.md" \
    --n-train-trajs 100 \
    --n-eval-episodes 100 \
    --T-explore 64 \
    --T-navigate 128 \
    --sr-samples 20 \
    --sr-future 20 \
    --sr-gamma 0.92 \
    --seeds 0 1 2 \
    > "$LOGS/sr_probe.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add SR_PROBE_RESULTS.md probe_sr.py run_probe_sr.sh
git commit -m "Successor representation (SR) readout from frozen prediction-trained models

Trains a linear SR head from frozen lm200 prediction-trained representations,
then evaluates reward-conditional planning: plant random one-hot reward,
compute V = SR · r, derive policy via one-step lookahead, roll out closed-loop.

This is the strongest cognitive-map content claim possible: the prediction-
trained representation supports navigation to arbitrary rewards without any
goal-directed training of the backbone.

Variants: RoPE / Vanilla / Level15 / Level15GSF_NoDrop. TEMFaithful skipped
(its out_proj is called per-step in a Python loop; requires custom hidden
extraction).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [sr-probe] Done."
