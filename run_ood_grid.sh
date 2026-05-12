#!/bin/bash
# OOD-d / OOD-s grid-size + density generalization eval.
# Uses existing clean-trained checkpoints (5 variants × 3 seeds), no new training.
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

echo "[$(date)] [ood-grid] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [ood-grid] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.eval_ood_grid \
    --output-md "$REPO/OOD_GRID_RESULTS.md" \
    --n-trials 30 \
    --seeds 0 1 2 \
    > "$LOGS/ood_grid_eval.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add OOD_GRID_RESULTS.md eval_ood_grid.py run_ood_grid.sh
git commit -m "OOD-d / OOD-s grid-size + density generalization (paper-faithful)

Evaluates clean-trained checkpoints on the MapFormer paper's Table 2 OOD
configs: OOD-d (32×32, p_empty=0.2, T=64) and OOD-s (128×128, p_empty=0.8,
T=512). MapFormer variants use omega-rescaling (×N/N'); TEMFaithful gets
no rescaling. 5-variant minimal set × 3 seeds, no new training. Result in
OOD_GRID_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [ood-grid] Done."
