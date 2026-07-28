#!/usr/bin/env bash
# PoPE + combo + hierarchy: train 3 PoPE variants on both tasks (3 seeds), then
# re-aggregate. Completes the position-source x combination x structure factorial.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/pope.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
VS=(PoPE MapPoPE MapPoPE_Hier)
hg(){ local g=$1 v=$2 s=$3 o="$REPO/runs/hiergoal_multiseed/seed$3"
  [ -f "$o/${v}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] hg $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
cp(){ local g=$1 v=$2 s=$3 o="$REPO/runs/comp_multiseed/seed$3"
  [ -f "$o/${v}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] cp $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
JOBS=()
for v in "${VS[@]}"; do for s in 0 1 2; do JOBS+=("hg:$v:$s" "cp:$v:$s"); done; done
run_job(){ local g=$1 j=$2; IFS=: read -r t v s <<<"$j"; [ "$t" = hg ] && hg "$g" "$v" "$s" || cp "$g" "$v" "$s"; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run_job 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run_job 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier PlainFlat PlainHourglass \
  --lengths 64 128 192 256 --out "$REPO/HIERGOAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier HourglassFlat3 PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.pope_done"
