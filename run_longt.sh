#!/bin/bash
# Tier 1: Long-T eval on existing lm200 checkpoints.
# Tests the cognitive-map necessity claim: does the gap RoPE → MapFormer
# grow with sequence length? No new training, eval-only.
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

echo "[$(date)] [longT] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [longT] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.eval_long_t \
    --output-md "$REPO/LONGT_EVAL_RESULTS.md" \
    --T-values 512 1024 2048 \
    --n-trials 30 \
    > "$LOGS/longt_eval.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add LONGT_EVAL_RESULTS.md eval_long_t.py run_longt.sh
git commit -m "Long-T eval: does RoPE → MapFormer gap grow with sequence length?

Tier 1 cognitive-map necessity test. Eval-only on existing lm200
checkpoints at T ∈ {512, 1024, 2048}. Minimal model set: RoPE, Vanilla,
Level15, Level15GSF_NoDrop, TEMFaithful. Result in LONGT_EVAL_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [longT] Done."
