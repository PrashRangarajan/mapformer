#!/usr/bin/env bash
# Re-run the PoPE arm with FAITHFUL PoPE (learnable delta_c). Removes the
# PoPE-lite checkpoints (incompatible with the delta layer), retrains all PoPE
# variants on hier-goal + compositional (3 seeds) and the PoPE clock variants
# (3 seeds), re-aggregates all three tables.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/pope_faithful.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
for s in 0 1 2; do
  rm -f "$REPO/runs/hiergoal_multiseed/seed$s"/{PoPE,MapPoPE,MapPoPE_Hier,MapPoPE_CoarseIdx}_hiergoal.pt
  rm -f "$REPO/runs/comp_multiseed/seed$s"/{PoPE,MapPoPE,MapPoPE_Hier,MapPoPE_CoarseIdx}.pt
  rm -f "$REPO/runs/clock_scan/seed$s"/{PoPE,MapPoPE_Hier,MapPoPE_CoarseIdx}_clock.pt
done
hg(){ local g=$1 v=$2 s=$3 o="$REPO/runs/hiergoal_multiseed/seed$3"
  [ -f "$o/${v}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] hg $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
cp(){ local g=$1 v=$2 s=$3 o="$REPO/runs/comp_multiseed/seed$3"
  [ -f "$o/${v}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] cp $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
ck(){ local g=$1 v=$2 s=$3 o="$REPO/runs/clock_scan/seed$3"
  [ -f "$o/${v}_clock.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] ck $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_clock --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
JOBS=()
for s in 0 1 2; do for v in PoPE MapPoPE MapPoPE_Hier MapPoPE_CoarseIdx; do JOBS+=("hg:$v:$s" "cp:$v:$s"); done; done
for s in 0 1 2; do for v in PoPE MapPoPE_Hier MapPoPE_CoarseIdx; do JOBS+=("ck:$v:$s"); done; done
run_job(){ local g=$1 j=$2; IFS=: read -r t v s <<<"$j"; case $t in hg) hg $g $v $s;; cp) cp $g $v $s;; ck) ck $g $v $s;; esac; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run_job 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run_job 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier MapPoPE_CoarseIdx PlainFlat PlainHourglass \
  --lengths 64 128 192 256 --out "$REPO/HIERGOAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier MapPoPE_CoarseIdx HourglassFlat3 PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_clock --runs-dir "$REPO/runs/clock_scan" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarsePI PoPE MapPoPE_Hier MapPoPE_CoarseIdx PlainFlat \
  --lengths 64 128 192 256 --out "$REPO/CLOCK_SCAN.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.pope_faithful_done"
