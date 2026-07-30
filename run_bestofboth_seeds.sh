#!/usr/bin/env bash
# Tighten the error bars on the length<->content frontier claim. Adds seeds 3-7
# (-> n=8) for the two variants that decide it, on the CONTENT axis only
# (hier-goal is already +-0.001, the variance problem is compositional):
#   MapPoPE_CoarseIdx   (best-of-both, faithful PoPE)
#   Hourglass_CoarseIdx (the RoPE content king it must match)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/bob_seeds.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
cp(){ local g=$1 v=$2 s=$3 o="$REPO/runs/comp_multiseed/seed$3"
  [ -f "$o/${v}.pt" ] && { echo "$(date +%H:%M) skip $v s$s" >> "$LOG"; return; }
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
JOBS=(); for s in 3 4 5 6 7; do for v in MapPoPE_CoarseIdx Hourglass_CoarseIdx; do JOBS+=("$v:$s"); done; done
rj(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"; cp "$g" "$v" "$s"; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && rj 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && rj 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating (n up to 8)" >> "$LOG"
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 3 4 5 6 7 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier MapPoPE_CoarseIdx HourglassFlat3 PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.bob_seeds_done"
