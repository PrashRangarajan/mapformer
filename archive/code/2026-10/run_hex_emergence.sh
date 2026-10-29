#!/bin/bash
# Hex / place cell emergence probe on trained MapFormer + TEMFaithful models.
# Eval-only. ~10-15 min compute.
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

echo "[$(date)] [hex-probe] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [hex-probe] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_hex_emergence \
    --output-md "$REPO/HEX_EMERGENCE_RESULTS.md" \
    --n-trajectories 100 --T 256 \
    > "$LOGS/hex_emergence.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add HEX_EMERGENCE_RESULTS.md probe_hex_emergence.py run_hex_emergence.sh
git commit -m "Hex / place cell emergence probe: do trained models show spatial firing?

Classical neuroscience test: take prediction-trained models, compute per-unit
rate maps across cell visits, score for hex (grid cell) and peak (place cell)
firing patterns. Compares MapFormer family (RoPE, Vanilla, Level15, GSF)
to TEMFaithful.

Tests whether cognitive-map-like representations emerge naturally from
prediction training, or require specific training objectives (Sorscher 2019:
non-negativity + DoG targets).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [hex-probe] Done."
