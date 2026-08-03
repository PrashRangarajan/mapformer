#!/usr/bin/env bash
# Dimensionality sweep testing the SSP discriminability floor 3/sqrt(D_head).
#   d=128 -> 32 bands/head, floor 0.375   (already trained: runs/hiergoal_multiseed)
#   d=256 -> 64 bands/head, floor 0.265
#   d=512 -> 128 bands/head, floor 0.188
# 4 variants x {256,512} x 3 seeds. Plain variants are the PARAM CONTROL: if gains
# are generic capacity they rise too; if positional precision, MapFormer gains more.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/dimsweep.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
VS=(Vanilla Hourglass_k2 PlainFlat PlainHourglass)
job(){ local g=$1 v=$2 d=$3 s=$4 o="$REPO/runs/dimsweep/d$3/seed$4"
  [ -f "$o/${v}_hiergoal.pt" ] && { echo "$(date +%H:%M) skip $v d$d s$s" >> "$LOG"; return; }
  echo "$(date +%H:%M) [gpu$g] $v d=$d seed=$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --d-model "$d" --n-heads 2 --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
JOBS=()
for d in 256 512; do for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$d:$s"); done; done; done
rj(){ local g=$1 j=$2; IFS=: read -r v d s <<<"$j"; job "$g" "$v" "$d" "$s"; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && rj 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && rj 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants "${VS[@]}" --lengths 64 128 192 256 --out "$REPO/DIMSWEEP_d128.md" >> "$LOG" 2>&1
for d in 256 512; do
  python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/dimsweep/d$d" --seeds 0 1 2 \
    --variants "${VS[@]}" --lengths 64 128 192 256 --out "$REPO/DIMSWEEP_d$d.md" >> "$LOG" 2>&1
done
cd "$REPO"; git add DIMSWEEP_d128.md DIMSWEEP_d256.md DIMSWEEP_d512.md \
  DIMSWEEP_d128.json DIMSWEEP_d256.json DIMSWEEP_d512.json 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Dimensionality sweep: d in {128,256,512} on hier-goal

Tests the SSP discriminability floor 3/sqrt(D_head): 0.375 / 0.265 / 0.188.
4 variants x 3 seeds; plain index-RoPE variants act as the parameter control
(634K/2.4M/9.6M params) to separate positional precision from generic capacity.
Auto-committed by run_dimsweep.sh; interpretation pending review."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.dimsweep_done"
