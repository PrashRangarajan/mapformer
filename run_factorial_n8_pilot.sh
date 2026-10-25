#!/usr/bin/env bash
# FACTORIAL_N8_PREREG.md pilot (timing / memory only; seeds 2890-2891, OUTSIDE the batch 2800-2803; rule 29).
# (a) solo: the slowest cell (RoPE --n-layers 4) alone on the GPU, 10 epochs.
# (b) contended: the four cells at once (the batch's DRV_MAXPG=4 packing), 15 epochs, seed 2891.
# Exact batch flags except --epochs. (2026-10-24 run: a bare `wait` hung on the monitor loop until killed by hand, so
# its 'contended wall' line is inflated; the per-epoch times in the logs are the measurement. Fixed below.) Per-process GPU memory sampled every 5 s with nvidia-smi.
#   cd /home/prashr/mapformer && PILOT_GPU=1 setsid nohup bash run_factorial_n8_pilot.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
G="${PILOT_GPU:?set PILOT_GPU}"
P="$REPO/runs/factorial_n8_pilot"; mkdir -p "$P"
LOG="$P/pilot.log"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"$REPO/.run_factorial_n8_pilot.lock"; flock -n 9 || { echo "REFUSED: lock held" >> "$LOG"; exit 1; }
cd "$REPO/.."
echo "start $(date) code $(cd "$REPO" && git rev-parse HEAD) GPU $G" >> "$LOG"
train(){ local V=$1 NL=$2 S=$3 E=$4 OUT=$5
  OMP_NUM_THREADS=3 python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" --epochs "$E" --lr 1e-3 --n-batches 98 \
    --batch-size 16 --n-steps 1024 --n-layers "$NL" --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --fast-attn --device "cuda:$G" --output-dir "$OUT" > "$OUT.log" 2>&1; }
mon(){ while :; do echo "$(date +%s) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$G" | tr '\n' ';')" >> "$1"; sleep 5; done; }
mon "$P/mem_solo.txt" & M=$!
t0=$(date +%s); train RoPE 4 2890 10 "$P/solo_L4_RoPE_s2890"; t1=$(date +%s)
kill $M; echo "solo L4 RoPE 10 epochs: wall $((t1 - t0)) s" >> "$LOG"
mon "$P/mem_contended.txt" & M=$!
t0=$(date +%s)
PIDS=()
train RoPE 4 2891 15 "$P/c_L4_RoPE_s2891" & PIDS+=($!)
train RoPELooped 1 2891 15 "$P/c_LOOP_RoPELooped_s2891" & PIDS+=($!)
train RoPE 1 2891 15 "$P/c_L1_RoPE_s2891" & PIDS+=($!)
train Vanilla_r4 1 2891 15 "$P/c_L1_Vanilla_r4_s2891" & PIDS+=($!)
wait "${PIDS[@]}"; t1=$(date +%s); kill $M     # wait on the trainers only (a bare `wait` also waits on the monitor loop)
echo "contended 4 cells 15 epochs: wall $((t1 - t0)) s" >> "$LOG"
for f in "$P"/solo_*.log "$P"/c_*.log; do echo "$(basename "$f"): $(grep -c 'Epoch' "$f") epoch lines; last: $(grep 'Epoch' "$f" | tail -1)" >> "$LOG"; done
touch "$P/.pilot_done"; echo "done $(date)" >> "$LOG"
