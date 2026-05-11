#!/bin/bash
# GSF mode-weight diagnostic — quick probe of trained Level15GSF checkpoints.
# Reports entropy / effective-mode-count / winner-mode statistics over time.
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

echo "[$(date)] [GSF-modes] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [GSF-modes] GPUs free."

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_gsf_modes \
    --output-md "$REPO/GSF_MODES_DIAGNOSTIC.md" \
    --n-trials 50 --T 512 > "$LOGS/gsf_modes.log" 2>&1

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add GSF_MODES_DIAGNOSTIC.md probe_gsf_modes.py run_gsf_modes.sh
git commit -m "GSF mode-weight diagnostic: is Level15GSF actually multi-modal?

Reports entropy and effective-mode-count of the K=8 mixture over time
on lm200 trajectories. Tells us whether K=8 is justified or collapses
to ~1 mode. Result in GSF_MODES_DIAGNOSTIC.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [GSF-modes] Done."
