#!/usr/bin/env bash
# (1) best-of-both MapPoPE-Hier-CoarseIdx on both tasks (3 seeds); (2) PoPE-on-clock
# scan (key variants, seed 0) to test whether the length-extrapolation win transfers
# to the symbolic modular domain. Re-aggregates all three tables.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/bestofboth.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
BB=MapPoPE_CoarseIdx
hg(){ local g=$1 s=$2 o="$REPO/runs/hiergoal_multiseed/seed$2"
  [ -f "$o/${BB}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] hg $BB s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$BB" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
cp(){ local g=$1 s=$2 o="$REPO/runs/comp_multiseed/seed$2"
  [ -f "$o/${BB}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] cp $BB s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$BB" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
ck(){ local g=$1 v=$2 o="$REPO/runs/clock_scan/seed0"
  [ -f "$o/${v}_clock.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] clock $v" >> "$LOG"
  python3 -u -m mapformer.train_clock --variant "$v" --seed 0 --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( cp 0 0; cp 0 1; hg 0 0; ck 0 Vanilla; ck 0 Hourglass_CoarsePI; ck 0 MapPoPE_Hier; ck 0 PlainFlat
  echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) & P0=$!
( cp 1 2; hg 1 1; hg 1 2; ck 1 Hourglass_k2; ck 1 PoPE; ck 1 MapPoPE_CoarseIdx
  echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier MapPoPE_CoarseIdx PlainFlat PlainHourglass \
  --lengths 64 128 192 256 --out "$REPO/HIERGOAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE MapPoPE_Hier MapPoPE_CoarseIdx HourglassFlat3 PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_clock --runs-dir "$REPO/runs/clock_scan" --seeds 0 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarsePI PoPE MapPoPE_Hier MapPoPE_CoarseIdx PlainFlat \
  --lengths 64 128 192 256 --out "$REPO/CLOCK_SCAN.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.bestofboth_done"
