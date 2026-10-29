#!/usr/bin/env bash
# Orchestrate the GridL15PC sweep: 3 configs × 3 seeds = 9 runs across 2 GPUs.
# s0 clean is already running on GPU 0 (started separately); this script
# launches the remaining 8 in two parallel queues.

set -euo pipefail

cd "$(dirname "$0")/.."        # repo root above mapformer/
LOGDIR=mapformer/logs
RUNDIR=mapformer/runs
mkdir -p "$LOGDIR"

run_one() {
    local gpu="$1" config="$2" seed="$3"
    local extra_args=""
    case "$config" in
        clean)  extra_args="" ;;
        noise)  extra_args="--p-action-noise 0.10" ;;
        lm200)  extra_args="--n-landmarks 200" ;;
        *) echo "Unknown config: $config" >&2; return 1 ;;
    esac
    local outdir="${RUNDIR}/GridL15PC_${config}/s${seed}"
    local logfile="${LOGDIR}/GridL15PC_${config}_s${seed}.log"
    mkdir -p "$outdir"
    echo "[$(date +%H:%M:%S)] GPU $gpu  ${config}  seed=$seed → $logfile"
    CUDA_VISIBLE_DEVICES="$gpu" python3 -m mapformer.train_variant \
        --variant GridL15PC \
        --d-model 132 \
        --aux-coef 0.1 \
        --epochs 50 --n-batches 156 \
        --seed "$seed" \
        --output-dir "$outdir" \
        $extra_args \
        > "$logfile" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  GPU $gpu  ${config}  seed=$seed"
}

# GPU 0 queue (s0 clean is already running externally — start with seed 1 for clean)
gpu0_queue() {
    run_one 0 clean 1
    run_one 0 clean 2
    run_one 0 noise 0
    run_one 0 noise 1
}

# GPU 1 queue
gpu1_queue() {
    run_one 1 noise 2
    run_one 1 lm200 0
    run_one 1 lm200 1
    run_one 1 lm200 2
}

# Launch both queues in parallel
gpu0_queue > "${LOGDIR}/orchestrator_gpu0.log" 2>&1 &
PID0=$!
gpu1_queue > "${LOGDIR}/orchestrator_gpu1.log" 2>&1 &
PID1=$!

echo "GPU 0 queue: PID $PID0"
echo "GPU 1 queue: PID $PID1"
wait $PID0 $PID1
echo "All queues finished at $(date +%H:%M:%S)"
